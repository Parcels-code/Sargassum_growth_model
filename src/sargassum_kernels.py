import parcels
import numpy as np
from scipy.special import erfc

SargassumParticle = parcels.Particle.add_variable(
        [
            parcels.Variable('temperature', dtype=np.float32, to_write=True, initial=0),
            parcels.Variable('lim_temp', dtype=np.float32, to_write=True, initial=0),
            parcels.Variable('nitrogen', dtype=np.float32, to_write=True, initial=0),
            parcels.Variable('lim_no3', dtype=np.float32, to_write=True, initial=0),
            parcels.Variable('salinity', dtype=np.float32, to_write=True, initial=0),
            parcels.Variable('lim_salinity', dtype=np.float32, to_write=True, initial=0),
            parcels.Variable('limitation', dtype=np.float32, to_write=True, initial=1),
            parcels.Variable('biomass', dtype=np.float32, to_write=True, initial=1),
            parcels.Variable('stranded', dtype=np.float32, to_write=True, initial=0),
        ]
    )

def di_Stokes_drift(particles, fieldset):
    """Depth-integrated Stokes drift kernel:

    Description
    ----------
    Using the approach in [1] (assuming a Phillips wave spectrum), equation A.6 and A.7 of [2] are used to determine
    the Stokes drift velocity integrated over depth between upper extent and lower extent of particle.

    Stokes drift is treated as a linear addition to the velocity field.

    Parameter Requirements
    ----------
    fieldset :
        - fieldset.UVStokes: zonal and meridional Stokes drift velocity at surface [m s-1]
        - fieldset.VTPK: the peak wave period field [s].

    References
    ----------
    [1] Breivik (2016) - https://doi.org/10.1016/j.ocemod.2016.01.005
    [2] Li et al. (2017) - http://dx.doi.org/10.1016/j.ocemod.2017.03.016
    """

    #Sampling the U / V components of Stokes drift at upper level
    stokes_U, stokes_V = fieldset.UVStokes[particles]

    #Sampling the peak wave period and wave number at the particle locations
    T_p = np.maximum(fieldset.VTPK[particles], 1E-14)
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


def windage_drift(particles, fieldset):
    """Leeway windage kernel.

    Description
    ----------
    A simple windage kernel that applies a linear relative 'wind velocity' to the particle.
    Slightly adapted for the usage for Sargassum.

    Kernel Requirements
    ----------
    - `fieldset.wind_coeff`, the particle windage coefficient in decimals.
    - `fieldset.UV`, the ocean velocities. Units [m s-1].
    - `fieldset.UVWind`, the wind velocity field at 10m height above sea surface. Units [m s-1].

    """
    # Sample ocean velocities
    (ocean_U, ocean_V) = fieldset.UV[particles]

    # Use a basic approach to only apply windage to particle in the ocean
    wind_U, wind_V = fieldset.UVWind[particles]

    # Compute particle displacement
    particles.dlon += fieldset.wind_coeff * (wind_U - ocean_U) * particles.dt
    particles.dlat += fieldset.wind_coeff * (wind_V - ocean_V) * particles.dt

def stranding(particles, fieldset):
    """Data-based stranding kernel.

    Description
    ----------
    Kernel that determines if a particle is stranded under the condition that U or V == 0.
    When a particle is stranded, tt also makes sure that physical transport is set to 0 (so no updates of particles' dlon and dlat)

    Parameter Requirements
    ----------
    particle :
        - stranded - initially 0, and becomes 1 when stranded.
    fieldset :
        - `fieldset.UV`, the ocean velocities. Units [m s-1].

    Kernel Requirements
    ----------
    Order of Operations:
        At the end of physical kernels. Otherwise dlon and dlat will be updated again.

    """

    u, v = fieldset.UV[particles]

    particles.stranded = np.where((u == 0.0) | (v == 0.0), 1, particles.stranded)

    particles.dlon = np.where(particles.stranded == 1, 0.0, particles.dlon)
    particles.dlat = np.where(particles.stranded == 1, 0.0, particles.dlat)


#Kernel that determines the new weight of the particle
#Based on the maximum growth rates of morphotypes and multiple limitation curves
def sargassum_biological_growth_model(particles, fieldset):

    # Growth limitation function for temperature, based on Jouanno et al. (2025).
    particles.temperature = fieldset.thetao[particles]
    limT_left = np.exp(-2 * ( (particles.temperature - fieldset.T_opt)/ (fieldset.T_min - fieldset.T_opt))**2 )
    limT_right = np.exp(-2 * ( (particles.temperature - fieldset.T_opt)/ (fieldset.T_max - fieldset.T_opt))**2 )
    particles.lim_temp = np.where(particles.temperature < fieldset.T_opt, limT_left, limT_right)

    # Growth limitation function for nitrogen, based on Bonner et al. (2024)
    particles.nitrogen    = fieldset.no3[particles]
    particles.lim_no3 = particles.nitrogen / ( fieldset.k_N + particles.nitrogen )

    # Growth limitation function for salinity, based on Jouanno et al. (2025)
    particles.salinity    = fieldset.so[particles]
    particles.lim_salinity = np.exp(-0.02 * (fieldset.S_opt - particles.salinity)**2 )

    # Compute total growth limitation as the product of the three limitation functions
    particles.limitation = particles.lim_temp * particles.lim_no3 * particles.lim_salinity

    # Update biomass of floating particles with maximum specific growth rate and mortality rate converted from day-1 to s-1
    ptcls_afloat = particles[particles.stranded == 0]
    ptcls_afloat.biomass *= 2 ** ((ptcls_afloat.limitation * fieldset.mu_max - fieldset.mort) * ptcls_afloat.dt  / (24*60*60))


def DeleteOutOfBounds(particles, fieldset):
    out_of_bounds = particles.state == parcels.StatusCode.ErrorOutOfBounds
    particles[out_of_bounds].state = parcels.StatusCode.Delete