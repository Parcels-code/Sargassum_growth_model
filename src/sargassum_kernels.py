import parcels
import numpy as np
from scipy.special import erfc

SargassumParticle = parcels.Particle.add_variable(
        [
            parcels.Variable('temperature', dtype=np.float32, to_write=True, initial=0),
            parcels.Variable('salinity', dtype=np.float32, to_write=True, initial=0),
            parcels.Variable('depth_extent', dtype=np.float32, to_write=True, initial=1),
            parcels.Variable('nitrogen', dtype=np.float32, to_write=True, initial=0),
            parcels.Variable('biomass_SF3', dtype=np.float32, to_write=True, initial=1),
            parcels.Variable('biomass_SN1', dtype=np.float32, to_write=True, initial=1),
            parcels.Variable('biomass_SN8', dtype=np.float32, to_write=True, initial=1),
            parcels.Variable('biomass_loss', dtype=np.float32, to_write=True, initial=0),
            parcels.Variable('stranded', dtype=np.float32, to_write=True, initial=0),
            parcels.Variable('limitation', dtype=np.float32, to_write=True, initial=1),
            parcels.Variable('lim_salinity', dtype=np.float32, to_write=True, initial=0),
            parcels.Variable('lim_temp', dtype=np.float32, to_write=True, initial=0),
            parcels.Variable('lim_no3', dtype=np.float32, to_write=True, initial=0),
            parcels.Variable('speed_currents', dtype=np.float32, to_write=True, initial=0),
            parcels.Variable('speed_stokes', dtype=np.float32, to_write=True, initial=0),
            parcels.Variable('speed_wind', dtype=np.float32, to_write=True, initial=0),
            parcels.Variable('decay_factor', dtype=np.float32, to_write=True, initial=0),
            parcels.Variable('decay_averaged', dtype=np.float32, to_write=True, initial=0),
            parcels.Variable('decay_integrated_lower', dtype=np.float32, to_write=True, initial=0),
            parcels.Variable('decay_integrated_upper', dtype=np.float32, to_write=True, initial=0),
            parcels.Variable('wind_coefficient', dtype=np.float32, to_write=True, initial=0.01),
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

    delta_z = particles.depth_extent - particles.z
    z_up = particles.z
    z_low = z_up + particles.depth_extent

    #Sampling the U / V components of Stokes drift at upper level
    stokes_U, stokes_V = fieldset.UVStokes[particles]

    #Sampling the peak wave period and wave number at the particle locations
    T_p = np.maximum(fieldset.VTPK[particles], 1E-14)
    omega_p = 2. * np.pi / T_p
    k_p = (omega_p ** 2) / fieldset.G

    #Decay function lower extent, based on Equation A.6 of Li et al. (2017)
    decay_function_lower = 1/(2*k_p) * (
                1 - np.exp(-2.0*k_p*z_low)
                - (2.0/3.0) * (1 + np.sqrt(np.pi) * (2.0*k_p*z_low)**(3.0/2.0) * erfc(np.sqrt(2.0*k_p*z_low))
                - (1 + 2.0*k_p*z_low) * np.exp(-2.0*k_p*z_low)   )
                )

    #Decay function upper extent, based on Equation A.6 of Li et al. (2017)
    decay_function_upper = 1/(2*k_p) * (
                1 - np.exp(-2.0*k_p*z_up)
                - (2.0/3.0) * (1 + np.sqrt(np.pi) * (2.0*k_p*z_up)**(3.0/2.0) * erfc(np.sqrt(2.0*k_p*z_up))
                - (1 + 2.0*k_p*z_up) * np.exp(-2.0*k_p*z_up)   )
                )

    #Integration function between surface and lower level based on Equation A.7 of Li et al. (2017)
    stokes_U_integrated = (stokes_U * decay_function_lower - stokes_U * decay_function_upper) / delta_z
    stokes_V_integrated = (stokes_V * decay_function_lower - stokes_V * decay_function_upper) / delta_z

    #Saving lower and upper decay function and total Stokes decay factor as particle variables
    particles.decay_integrated_lower = decay_function_lower
    particles.decay_integrated_upper = decay_function_upper
    particles.decay_factor = (decay_function_lower - decay_function_upper) / delta_z

    #Compute particle displacement based on depth-integrated Stokes velocity
    particles.dlon += stokes_U_integrated * particles.dt
    particles.dlat += stokes_V_integrated * particles.dt


def windage_drift(particles, fieldset):
    """Leeway windage kernel.

    Description
    ----------
    A simple windage kernel that applies a linear relative 'wind velocity' to the particle.
    Slightly adapted for the usage for Sargassum.

    We treat the windage drift as a linear addition to the velocity field
        :math:`u(x,t) = u_c(x,t) + C_w * (u_w(x,t)-u_c(x,t))`
    where :math:`u_c` is the ocean current velocity, :math:`u_w` is the wind velocity
    at 10m height, and :math:`C_w` is the windage coefficient.

    For further description, see https://plastic.parcels-code.org/en/latest/physicskernels.html#wind-induced-drift-leeway

    Parameter Requirements
    ----------
    particle :
        - wind_coefficient - the particle windage coefficient in decimals.
    fieldset :
        - `fieldset.UV`, the ocean velocities. Units [m s-1].
        - `fieldset.UVWind`, the wind velocity field at 10m height above sea surface. Units [m s-1].

    Kernel Requirements
    ----------
    Order of Operations:
        None - can be applied at any time.

    """
    # Sample ocean velocities
    (ocean_U, ocean_V) = fieldset.UV[particles]

    # Use a basic approach to only apply windage to particle in the ocean
    wind_U, wind_V = fieldset.UVWind[particles]

    # Compute particle displacement
    particles.dlon += particles.wind_coefficient * (wind_U - ocean_U) * particles.dt
    particles.dlat += particles.wind_coefficient * (wind_V - ocean_V) * particles.dt

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

    #We start by sampling temperature field, salinity field and nitrogen field at particle location
    particles.temperature =  fieldset.thetao[particles]
    particles.salinity =     fieldset.so[particles]

    #Selecting depth at which nitrogen field is defined
    # = particle.depth
    #if z_for_n <= 0.49402538:

    # TODO check is z_for_n is needed
    # z_for_n = 0.49402538
    # particles.nitrogen =     fieldset.no3[particles.time, z_for_n, particles.lat, particles.lon]
    particles.nitrogen =     fieldset.no3[particles]

    #RATES
    maximum_growth_rate = 0.095 #doublings/day
    mortality_rate = 0.02     #relative loss/day

    #Minimum, maximum and optimal temperature
    T_min = 20      #degC
    T_max = 31      #degC
    T_opt = 27.5    #degC

    #Optimal salinity
    S_opt = 36 #psu

    #GROWTH LIMITATION FUNCTION DEPENDENT ON TEMPERATURE
    #Formulation from Jouanno et al. (2025).
    limT_left = np.exp(-2 * ( (particles.temperature - T_opt)/ (T_min - T_opt))**2 )
    limT_right = np.exp(-2 * ( (particles.temperature - T_opt)/ (T_max - T_opt))**2 )
    limitation_factor_T = np.where(particles.temperature < T_opt, limT_left, limT_right)

    #GROWTH LIMITATION FUNCTION DEPENDENT ON NITROGEN AVAILABILTIY
    #Formulation from Bonner et al. (2024)
    limitation_factor_N = particles.nitrogen / ( fieldset.k_N + particles.nitrogen )

    #GROWTH LIMITATION FUNCTION DEPENDENT ON SALINITY
    #Formula from Jouanno et al. (2025)
    limitation_factor_S = np.exp(-0.02 * (S_opt - particles.salinity)**2 )

    ###################################

    #Save particle total limitation and seperate limitations as variables
    LIMITATION = limitation_factor_T * limitation_factor_N * limitation_factor_S
    particles.limitation = LIMITATION
    particles.lim_salinity = limitation_factor_S
    particles.lim_temp = limitation_factor_T
    particles.lim_no3 = limitation_factor_N

    #UPDATE OF PARTICLE WEIGHT with maximum specific growth rate and mortality rate converted from day-1 to s-1
    #If particle is stranded, biomass is not updated.
    #mortality_rate = 0.025 #loss/day  # TODO why a different value here?

    # TODO check in Elena's original code what to do with stranding here
    # ptcls_stranded = particles[particles.stranded == 1]

    particles.biomass_SF3 *= 2 ** ((LIMITATION * (maximum_growth_rate / (24*60*60)) - mortality_rate / (24*60*60) ) * particles.dt )
    particles.biomass_loss = particles.biomass_SN1 - particles.biomass_SF3


def DeleteOutOfBounds(particles, fieldset):
    out_of_bounds = particles.state == parcels.StatusCode.ErrorOutOfBounds
    particles[out_of_bounds].state = parcels.StatusCode.Delete