#This file contains the kernels with additional transport mechanisms acting on Sargassum 
#Stokes and windage kernels are taken from Darshika Manral's project: https://github.com/OceanParcels/woc_sargassum_transport/blob/depth_avg/src/simulations/custom_kernels.py

import math

#Simple implementation of Stokes drift
def direct_Stokes_drift(particle, fieldset, time):
    # Sample the U / V components of Stokes drift
    stokes_U = fieldset.U_wave_Stokes[time, particle.depth, particle.lat, particle.lon]
    stokes_V = fieldset.V_wave_Stokes[time, particle.depth, particle.lat, particle.lon]

    # compute particle displacement
    particle_dlon += stokes_U * particle.dt
    particle_dlat += stokes_V * particle.dt 


def avg_Stokes_drift(particle, fieldset, time):
    """Stokes drift kernel: taken from https://github.com/OceanParcels/plasticparcels/kernels.py

    Description
    ----------
    Using the approach in [1] assuming a Phillips wave spectrum to determine
    the depth dependent Stokes drift. Specifically, the 'Stokes drift velocity'
    :math:`u_s` is computed as per Eq. (19) in [1].

    We treat the Stokes drift as a linear addition to the velocity field
        :math:`u(x,t) = u_c(x,t) + C_s * u_s(x,t)`
    where :math:`u_c` is the current velocity, :math:`u_s` is the Stokes drift velocity,
    and :math:`C_s` is the depth-varying decay factor.

    For further description, see https://plastic.oceanparcels.org/en/latest/physicskernels.html#stokes-drift

    Parameter Requirements
    ----------
    fieldset :
        - `fieldset.Stokes_U` and `fieldset.Stokes_V`, the Stokes drift velocity fields. Units [m s-1]
        - `fieldset.wave_Tp`, the peak wave period field (:math:`T_p`). Units [s].

    References
    ----------
    [1] Breivik (2016) - https://doi.org/10.1016/j.ocemod.2016.01.005

    """
    # Sample the U / V components of Stokes drift
    stokes_U = fieldset.U_wave_Stokes[time, particle.depth, particle.lat, particle.lon]
    stokes_V = fieldset.V_wave_Stokes[time, particle.depth, particle.lat, particle.lon]

    # Sample the peak wave period
    T_p = fieldset.wave_Tp[time, particle.depth, particle.lat, particle.lon]

    # Compute the local bathymetry / water depth with a margin of error
    # local_bathymetry = 0.99*fieldset.bathymetry[time, particle.depth, particle.lat, particle.lon]

    # Only compute displacements if the peak wave period is large enough and the particle is in the water
    if T_p > 1E-14: #and particle.depth < local_bathymetry:
        # Peak wave frequency
        omega_p = 2. * math.pi / T_p

        # Peak wave number
        k_p = (omega_p ** 2) / fieldset.G

        # Repeated inner term of Eq. (19) - note depth is negative in this formulation, but model depths are positive by convention
        # kp_z_2 = 2. * k_p * particle.depth
        kp_z_2 = 2. * k_p * particle.depth_extent / 2


        # Decay factor in Eq. (19) -- Where beta=1 for the Phillips spectrum
        decay = math.exp(-kp_z_2) - math.sqrt(math.pi * kp_z_2) * math.erfc(math.sqrt(kp_z_2))

        # Apply Eq. (19) and compute particle displacement
        particle_dlon += stokes_U * decay * particle.dt  # noqa
        particle_dlat += stokes_V * decay * particle.dt  # noqa


def di_Stokes_drift(particle, fieldset, time):
    """Depth-integrated Stokes drift kernel:

    Description
    ----------
    Using the approach in [1] assuming a Phillips wave spectrum,equation A.6 and A.7 are used to determine
    the Stokes drift velocity integrated over depth between upper extent en lower extent of particle. 

    Stokes drift is treated as a linear addition to the velocity field. 

    Parameter Requirements
    ----------
    fieldset :
        - `fieldset.Stokes_U` and `fieldset.Stokes_V`, the Stokes drift velocity fields. Units [m s-1]
        - `fieldset.wave_Tp`, the peak wave period field (:math:`T_p`). Units [s].

    References
    ----------
    [1] Li et al. (2017) - http://dx.doi.org/10.1016/j.ocemod.2017.03.016  """

    delta_z = particle.depth_extent - particle.depth
    z = particle.depth

    #Sampling the U / V components of Stokes drift at upper level
    stokes_U = fieldset.U_wave_Stokes[time, particle.depth, particle.lat, particle.lon]
    stokes_V = fieldset.V_wave_Stokes[time, particle.depth, particle.lat, particle.lon]

    #Sampling the peak wave period
    T_p = fieldset.wave_Tp[time, particle.depth, particle.lat, particle.lon]

    #Only computing displacements if the peak wave period is large enough and the particle is in the water
    if T_p > 1E-14: #and particle.depth < local_bathymetry:
        #Peak wave frequency
        omega_p = 2. * math.pi / T_p

        #Peak wave number
        k_p = (omega_p ** 2) / fieldset.G

        #Decay function lower extent, based on Equation A.6 of Li et al. (2017) 
        decay_function_lower = 1/(2*k_p) * ( 
                    1 - math.exp(-2*k_p*(particle.depth_extent)) 
                    - 2/3 * (1 + math.sqrt(math.pi) * (2*k_p*(particle.depth_extent))**(3/2) * math.erfc(math.sqrt(2*k_p*(particle.depth_extent)))  
                    - (1 + 2*k_p*(particle.depth_extent)) * math.exp(-2*k_p*(particle.depth_extent))   )
                    )
        
        #Decay function upper extent, based on Equation A.6 of Li et al. (2017) 
        decay_function_upper = 1/(2*k_p) * ( 
                    1 - math.exp(-2*k_p*(z)) 
                    - 2/3 * (1 + math.sqrt(math.pi) * (2*k_p*(z))**(3/2) * math.erfc(math.sqrt(2*k_p*(z)))  
                    - (1 + 2*k_p*(z)) * math.exp(-2*k_p*(z))   )
                    )
        
        #Integration function between surface and lower level based on Equation A.7 of Li et al. (2017)
        stokes_U_av = (stokes_U * decay_function_lower - stokes_U * decay_function_upper) / ( delta_z)
        stokes_V_av = (stokes_V * decay_function_lower - stokes_V * decay_function_upper) / ( delta_z)
        
        #Compute particle displacement based on depth-integrated Stokes velocity
        particle_dlon += stokes_U_av  * particle.dt  
        particle_dlat += stokes_V_av  * particle.dt 

def wind_drag(particle, fieldset, time):

    (curr_U, curr_V) = fieldset.UV[particle]
    ocean_speed = math.sqrt(curr_U**2 + curr_V**2)

    if ocean_speed > 1E-14:
        # Sample the U / V components of wind
        wind_U = fieldset.U_wind[time, particle.depth, particle.lat, particle.lon]
        wind_V = fieldset.V_wind[time, particle.depth, particle.lat, particle.lon]

        # compute particle displacement
        particle_dlon += wind_U* particle.dt
        particle_dlat += wind_V * particle.dt 



#Kernel that samples temperature field at particle location
def temperature_from_field(particle, fieldset, time):
    particle.temperature = fieldset.T[time, particle.depth, particle.lat, particle.lon]

#Kernel that samples nitrogen concentration field at particle location 
def nitrogen_from_field(particle, fieldset, time):
    #Selecting depth at which nitrogen field is defined
    z = particle.depth
    if z <= 0.49402538:
        z = 0.49402538
    
    particle.nitrogen = fieldset.no3[time, z, particle.lat, particle.lon] 

#Kernel that determines the new weight of the particleSS 
#Based on the growth rate of morphotype SF3 and SN8 and limitation curve from Jouanno et al. (2021)
def growth_temp_based(particle, fieldset, time):

    #Sargassum model parameters based on Jouanno et al. (2021) in deg Celsius
    T_opt_J = 26
    Tmin_J = 10.5
    Tmax_J = 43.8
    #Growth limitation function dependent on temperature, formula from Jouanno et al. (2021).
    if particle.temperature < T_opt_J:
        limitation_factor = math.exp(-0.5 * ( (particle.temperature - T_opt_J)/ (Tmin_J - particle.temperature))**2 )
    else:
        limitation_factor = math.exp(-0.5 * ( (particle.temperature - T_opt_J)/ (Tmax_J - particle.temperature))**2 )

    #Update particle weight with doubling rate converted from day-1 to s-1
    particle.weight_SF3 *= 2 ** (limitation_factor * (fieldset.MGR_SF3 / (24*60*60)) * particle.dt )
    particle.weight_SN8 *= 2 ** (limitation_factor * (fieldset.MGR_SN8 / (24*60*60)) * particle.dt )
    particle.weight_SN1 *= 2 ** (limitation_factor * (fieldset.MGR_SN1 / (24*60*60)) * particle.dt )

#Kernel that determines the new weight of the particle 
#Based on the maximum growth rates of morphotypes and limitation curve from Jouanno et al. (2025)
def growth_J25(particle, fieldset, time):

    #Sargassum model parameters based on Jouanno et al. (2025) in deg Celsius
    T_opt_J25 = 27.5    #Temperature growth optimum [degC]
    Tmin_J25 = 20       #Temperature growth minimum [degC]
    Tmax_J25 = 31       #Temperature growth maximum [degC]
    
    #Growth limitation function  dependent on temperature, formula from Jouanno et al. (2025).
    if particle.temperature < T_opt_J25:
        limitation_factor = math.exp(-2 * ( (particle.temperature - T_opt_J25)/ (Tmin_J25 - T_opt_J25))**2 )
    else:
        limitation_factor = math.exp(-2 * ( (particle.temperature - T_opt_J25)/ (Tmax_J25 - T_opt_J25))**2 )

    #Update particle weight with doubling rate converted from day-1 to s-1
    particle.weight_SF3 *= 2 ** (limitation_factor * (fieldset.MGR_SF3 / (24*60*60)) * particle.dt )
    particle.weight_SN8 *= 2 ** (limitation_factor * (fieldset.MGR_SN8 / (24*60*60)) * particle.dt )
    particle.weight_SN1 *= 2 ** (limitation_factor * (fieldset.MGR_SN1 / (24*60*60)) * particle.dt )

#Kernel that determines the new weight of the particle 
#Based on the maximum growth rates of morphotypes and limitation curves from Bonner et al. (2024)
def growth_Bonner(particle, fieldset, time):

    #Sargassum model parameters based on Bonner et al. (2024) 
    Tmin_B = 10     #Temperature growth minimum [degC]
    Tmax_B = 40     #Temperature growth maximum [degC]

    T_o_B = ( Tmin_B + Tmax_B ) / 2     #Temperature growth optimum [degC]

    k_N = 0.000129 #Nitrogen uptake half saturation [mmol/m3]

    #Growth limitation function dependent on temperature, formula from Bonner et al. (2024).
    if particle.temperature < T_o_B:
        limitation_factor_T = math.exp(-0.5 * ( (T_o_B - particle.temperature) / (particle.temperature - Tmin_B) )**2 ) 
    else:
        limitation_factor_T = math.exp(-0.5 * ( (T_o_B - particle.temperature) / (particle.temperature - Tmax_B) )**2 )

    #Growth limitation function dependent on nitrogen availability, formula from Bonner et al. (2024).
    limitation_factor_N = 1 / ( k_N / particle.nitrogen + 1 )

    #Update particle weight with doubling rate converted from day-1 to s-1
    particle.weight_SF3 *= 2 ** (limitation_factor_T * limitation_factor_N * (fieldset.MGR_SF3 / (24*60*60)) * particle.dt )
    particle.weight_SN8 *= 2 ** (limitation_factor_T * limitation_factor_N * (fieldset.MGR_SN8 / (24*60*60)) * particle.dt )
    particle.weight_SN1 *= 2 ** (limitation_factor_T * limitation_factor_N * (fieldset.MGR_SN1 / (24*60*60)) * particle.dt )