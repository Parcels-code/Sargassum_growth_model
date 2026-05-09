import parcels
import numpy as np
from scipy.special import erfc

SargassumParticle = parcels.Particle.add_variable(
        [
            parcels.Variable('temperature', dtype=np.float32, to_write=True, initial=0),
            parcels.Variable('lim_temperature', dtype=np.float32, to_write=True, initial=0),
            parcels.Variable('nitrogen', dtype=np.float32, to_write=True, initial=0),
            parcels.Variable('lim_nitrogen', dtype=np.float32, to_write=True, initial=0),
            parcels.Variable('salinity', dtype=np.float32, to_write=True, initial=0),
            parcels.Variable('lim_salinity', dtype=np.float32, to_write=True, initial=0),
            parcels.Variable('limitation', dtype=np.float32, to_write=True, initial=1),
            parcels.Variable('biomass', dtype=np.float32, to_write=True, initial=1),
            parcels.Variable('stranded', dtype=np.float32, to_write=True, initial=0),
        ]
    )

def DepthIntegratedStokesDriftRK2(particles, fieldset):
    """Depth-integrated Stokes drift kernel:

    Description
    ----------
    Using the approach in [1] (assuming a Phillips wave spectrum),
    equation A.6 and A.7 of [2] are used to determine the Stokes drift velocity
    integrated over depth between upper extent and lower extent of particle.

    Kernel Requirements
    ----------
    fieldset :
        - UVStokes: Zonal and meridional Stokes drift velocity at surface [m s-1]
        - VTPK: Peak wave period field [s]
        - z_upper: Upper depth extent of particle [m]
        - z_lower: Lower depth extent of particle [m]

    References
    ----------
    [1] Breivik (2016) - https://doi.org/10.1016/j.ocemod.2016.01.005
    [2] Li et al. (2017) - http://dx.doi.org/10.1016/j.ocemod.2017.03.016
    """

    for rk in range(2):
        if rk == 0:
            stokes_U, stokes_V = fieldset.UVStokes[particles]
            T_p = np.maximum(fieldset.VTPK[particles], 1E-14)
        else:
            (lon1, lat1) = (particles.lon + stokes_U * 0.5 * particles.dt, particles.lat + stokes_V * 0.5 * particles.dt)
            stokes_U, stokes_V = fieldset.UVStokes[particles.time + 0.5 * particles.dt, particles.z, lat1, lon1, particles]
            T_p = np.maximum(fieldset.VTPK[particles.time + 0.5 * particles.dt, particles.z, lat1, lon1, particles], 1E-14)

        omega_p = 2. * np.pi / T_p
        k_p = (omega_p ** 2) / 9.81

        #Decay function lower extent, based on Equation A.6 of Li et al. (2017)
        decay_function_lower = 1/(2*k_p) * (
                    1 - np.exp(-2.0*k_p*fieldset.z_lower)
                    - (2.0/3.0) * (1 + np.sqrt(np.pi) * (2.0*k_p*fieldset.z_lower)**(3.0/2.0) * erfc(np.sqrt(2.0*k_p*fieldset.z_lower))
                    - (1 + 2.0*k_p*fieldset.z_lower) * np.exp(-2.0*k_p*fieldset.z_lower))
                    )

        #Decay function upper extent, based on Equation A.6 of Li et al. (2017)
        decay_function_upper = 1/(2*k_p) * (
                    1 - np.exp(-2.0*k_p*fieldset.z_upper)
                    - (2.0/3.0) * (1 + np.sqrt(np.pi) * (2.0*k_p*fieldset.z_upper)**(3.0/2.0) * erfc(np.sqrt(2.0*k_p*fieldset.z_upper))
                    - (1 + 2.0*k_p*fieldset.z_upper) * np.exp(-2.0*k_p*fieldset.z_upper))
                    )

        #Integration function between surface and lower level based on Equation A.7 of Li et al. (2017)
        stokes_U_integrated = (stokes_U * decay_function_lower - stokes_U * decay_function_upper) / (fieldset.z_lower - fieldset.z_upper)
        stokes_V_integrated = (stokes_V * decay_function_lower - stokes_V * decay_function_upper) / (fieldset.z_lower - fieldset.z_upper)

    #Compute particle displacement based on depth-integrated Stokes velocity
    particles.dlon += stokes_U_integrated * particles.dt
    particles.dlat += stokes_V_integrated * particles.dt


def WindageRK2(particles, fieldset):
    """Leeway windage kernel using an RK2 scheme.

    Description
    ----------
    A simple windage kernel that applies a linear relative windage

    Kernel Requirements
    ----------
    fieldset :
        - wind_coeff: Windage coefficient [fraction]
        - UV: Ocean velocities [m s-1]
        - UVWind: Wind velocity field at 10m height above sea surface [m s-1]

    """

    ocean_U, ocean_V = fieldset.UV[particles]
    wind_U, wind_V = fieldset.UVWind[particles]
    u = ocean_U + fieldset.wind_coeff * (wind_U - ocean_U)
    v = ocean_V + fieldset.wind_coeff * (wind_V - ocean_V)

    lon1, lat1 = (particles.lon + u * 0.5 * particles.dt, particles.lat + v * 0.5 * particles.dt)

    ocean_U, ocean_V = fieldset.UV[particles.time + 0.5 * particles.dt, particles.z, lat1, lon1, particles]
    wind_U, wind_V = fieldset.UVWind[particles.time + 0.5 * particles.dt, particles.z, lat1, lon1, particles]
    u = ocean_U + fieldset.wind_coeff * (wind_U - ocean_U)
    v = ocean_V + fieldset.wind_coeff * (wind_V - ocean_V)

    particles.dlon += u * particles.dt
    particles.dlat += v * particles.dt


def Stranding(particles, fieldset):
    """Data-based stranding kernel.

    Description
    ----------
    Kernel that determines which particles are stranded under the condition
    that U or V == 0. Stranded particles are deleted.

    Kernel Requirements
    ----------
    particle :
        - stranded: Boolean whether particle is straned (1) or not (0)
    fieldset :
        - UV: Ocean velocities [m s-1]

    Order of Operations:
    ----------
        At the end of physical kernels. Otherwise dlon and dlat will be updated again.

    """

    u, v = fieldset.UV[particles]

    particles.stranded = np.where((u == 0.0) | (v == 0.0), 1, particles.stranded)
    particles[particles.stranded == 1].state = parcels.StatusCode.Delete


def SargassumBiologicalGrowthModel(particles, fieldset):
    """Sargassum biological growth kernel.

    Description
    ----------
    A kernel that implements a simple biological growth model for Sargassum

    Kernel Requirements
    ----------
    particle :
        - temperature: Temperature at particle locations [degC]
        - lim_temperature: Temperature limitation [fraction, range 0-1]
        - nitrogen: Nitrogen at particle locations [mmol m-3]
        - lim_nitrogen: Nitrogen limitation [fraction, range 0-1]
        - salinity: Temperature at particle locations [psu]
        - lim_salinity: Salinity limitation [fraction, range 0-1]

    fieldset :
        - mu_max: Maximum_growth_rate [doublings/day]
        - mort: Mortality relative loss/day
        - T_min: Minimum temperature [degC]
        - T_opt: Optimal temperature [degC]
        - T_max: Maximum temperature [degC]
        - S_opt: Optimal salinity [psu]
        - k_N: Nitrogen half saturation constant [mmol/m3]

    """

    # Growth limitation function for temperature, based on Jouanno et al. (2025).
    particles.temperature = fieldset.thetao[particles]
    limT_left = np.exp(-2 * ( (particles.temperature - fieldset.T_opt)/ (fieldset.T_min - fieldset.T_opt))**2 )
    limT_right = np.exp(-2 * ( (particles.temperature - fieldset.T_opt)/ (fieldset.T_max - fieldset.T_opt))**2 )
    particles.lim_temperature = np.where(particles.temperature < fieldset.T_opt, limT_left, limT_right)

    # Growth limitation function for nitrogen, based on Bonner et al. (2024)
    particles.nitrogen = fieldset.no3[particles]
    particles.lim_nitrogen = particles.nitrogen / ( fieldset.k_N + particles.nitrogen )

    # Growth limitation function for salinity, based on Jouanno et al. (2025)
    particles.salinity = fieldset.so[particles]
    particles.lim_salinity = np.exp(-0.02 * (fieldset.S_opt - particles.salinity)**2 )

    # Compute total growth limitation as the product of the three limitation functions
    particles.limitation = particles.lim_temperature * particles.lim_nitrogen * particles.lim_salinity

    # Update biomass of floating particles with maximum specific growth rate and mortality rate converted from day-1 to s-1
    ptcls_afloat = particles[particles.stranded == 0]
    ptcls_afloat.biomass *= 2 ** ((ptcls_afloat.limitation * fieldset.mu_max - fieldset.mort) * ptcls_afloat.dt  / (24*60*60))


def DeleteOutOfBounds(particles, fieldset):
    out_of_bounds = particles.state == parcels.StatusCode.ErrorOutOfBounds
    particles[out_of_bounds].state = parcels.StatusCode.Delete