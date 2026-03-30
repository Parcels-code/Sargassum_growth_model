##################################################################################################
#Importing relevent packages
import sys; print(sys.executable)
import parcels
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import click

#Importing file in which grid initialization and kernels are defined
import CUSTOM_KERNELS as ck
import grid_initialization as gi

#################################################################################################3
#Options for differnt simulations
@click.command(context_settings=dict(ignore_unknown_options=True))
@click.option('--year', default=2023, help= 'start year')
@click.option('--month', default=4, help= 'start month')
@click.option('--day', default=1, help= 'start day')

def run_multiple_simulations(year, month, day):

    ##################################################################################################
    #SETTINGS
    #Time-related settings
    CHANGEABLE_START = datetime(year, month, day)
    starttime = CHANGEABLE_START #datetime(2024,5,1)
    release_times = np.array([CHANGEABLE_START])
    dtime_data = timedelta(days=1)    #je stelt hier handmatig in wat de tijdsresolutie is van je data #deze kan niet naar minuten (13/5/2025)
    simulation_days = 30  
    endtime = release_times[-1] + timedelta(days=simulation_days) + dtime_data
    dtime_execute = timedelta(minutes=10)

    #To check:
    print(f"Checking correct timesettings: starttime = {starttime}, endtime = {endtime}")

    #Selecting physical variables to add
    add_advection = True
    add_stokes = True
    add_wind = True
    add_growth = True

    #Select type of Stokes drift (default is surface Stokes drift)
    avg_stokes = False          #depth-averaged
    di_stokes = True            #depth-integrated
    di_stokes_biomass = False    #depth-integrated including a dependency of depth-extent on biomass 

    #Selecting specific type of growth 
    add_biological_growth = True

    #Selecting other kernels
    add_stranding = True

    #Output storing
    output_folder = "/storage/shared/oceanparcels/output_data/data_Elena/"
    output_file_name = "SATELLITE_OUT/SIM2_westcar_LA_{year_start:04d}_{month_start:02d}_{day_start:02d}.zarr"
    #output_file_name = "Simulations_part2/SIM_lg_notransport_{year_start:04d}_{month_start:02d}.zarr"

    ##################################################################################################
    #Loading the physical dataset and matching the grid
    directory_phy =  '/storage/shared/oceanparcels/input_data/MOi/'
    phy_base_file =  directory_phy + 'GLO12/psy4v3r1-daily_{vector:s}_{y:04d}-{m:02d}-{d:02d}.nc' 
    phy_files_U = []
    phy_files_V = []
    phy_files_T = []
    phy_files_S = []

    time = starttime
    phy_file_W = directory_phy + 'GLO12/psy4v3r1-daily_{vector:s}_{y:04d}-{m:02d}-{d:02d}.nc'.format(vector = 'W', y = time.year, m = time.month, d = time.day)
    while(time < endtime):
        phy_files_U.append(phy_base_file.format(vector = 'U', y = time.year, m = time.month, d = time.day))
        phy_files_V.append(phy_base_file.format(vector = 'V', y = time.year, m = time.month, d = time.day))
        phy_files_T.append(phy_base_file.format(vector = 'T', y = time.year, m = time.month, d = time.day))
        phy_files_S.append(phy_base_file.format(vector = 'S', y = time.year, m = time.month, d = time.day))
        time+=dtime_data

    mesh_file_h = directory_phy + "domain_ORCA0083-N006/PSY4V3R1_mesh_hgr.nc"   #Hiermee converteren we het grid!
    mesh_file_z = directory_phy + "domain_ORCA0083-N006/PSY4V3R1_mesh_zgr.nc"
    filenames_phy = {
        'U': {
            'lon': mesh_file_h, 'lat': mesh_file_h, 'depth':phy_file_W,
            'data': phy_files_U,
        },'V': {
            'lon': mesh_file_h, 'lat': mesh_file_h, 'depth':phy_file_W,
            'data': phy_files_V,
        },'T': {
            'lon': mesh_file_h, 'lat': mesh_file_h, 'depth':phy_file_W,
            'data': phy_files_T,
        },'S': {
            'lon': mesh_file_h, 'lat': mesh_file_h, 'depth':phy_file_W,
            'data': phy_files_S,
        }
        }
    variables_phy = {'U': 'vozocrtx',
                    'V': 'vomecrty',
                    'T': 'votemper',
                    'S': 'vosaline'}

    c_grid_dimensions = {   'lat': 'gphif',
                            'lon': 'glamf',
                            'depth':'depthw',
                            'time': 'time_counter'}

    dimensions_phy = {'U': c_grid_dimensions,
                    'V': c_grid_dimensions,
                    'T': c_grid_dimensions,
                    'S': c_grid_dimensions}

    #Defining a range of indices to load as fieldset to reduce computational time
    indices = { 'lat':  range(1370,1950),
                'lon':  range(2260,3500),
                'depth':range(0,2)}
    #Creating fieldset
    fieldset = parcels.FieldSet.from_nemo(filenames=filenames_phy,variables=variables_phy,dimensions=dimensions_phy, indices=indices)
    print('fieldset of U, V, T and S = made')

    #Setting interpolation method for T and S to be the inverse distance weighting interpolation method 
    #To account for T and S values being 0
    fieldset.T.interp_method = "linear_invdist_land_tracer"
    fieldset.S.interp_method = "linear_invdist_land_tracer"

    #Adding the wave/stokes data
    if add_stokes:
        #depth_extent = 1 
        directory_stokes = '/storage/shared/oceanparcels/input_data/CopernicusMarineService/GLOBAL_ANALYSISFORECAST_WAV_001_027/'
        stokes_base_file =  directory_stokes + 'cmems_mod_glo_wav_anfc_0.083deg_PT3H-i_VSDX-VSDY-VTPK_180.00W-179.92E_80.00S-90.00N_{y:04d}-{m:02d}-{d:02d}-{y:04d}-{m:02d}-{d:02d}.nc' 
        stokes_base_file =  directory_stokes + 'cmems_mod_glo_wav_anfc_0.083deg_PT3H-i_VSDX-VSDY-VTPK_180.00W-179.92E_80.00S-90.00N_{y:04d}-{m:02d}-{d:02d}-{y:04d}-{m:02d}-{d:02d}.nc'
        
        stokes_files = []

        time = starttime
        while(time < endtime):
            stokes_files.append(stokes_base_file.format(y = time.year, m = time.month, d = time.day ))
            time+=dtime_data
        
        filenames_stokes = {'U_wave_Stokes': stokes_files,
                        'V_wave_Stokes': stokes_files,
                        'wave_Tp': stokes_files,
                        }
        variables_stokes = {'U_wave_Stokes': 'VSDX',        #Sea surface wave stokes drift x velocity
                        'V_wave_Stokes': 'VSDY',            #Sea surface wave stokes drift y velocity
                        'wave_Tp': 'VTPK'                   #Wave peak period
                        }
        dimensions_stokes = {'lat': 'latitude',
                            'lon': 'longitude',
                            'time': 'time'}
        
        #Defining a range of indices to load as fieldset to reduce computational time
        indices_stokes = {'lat': range(800,1370),
                          'lon': range(970,2250)
                         }
        
        fieldset_stokes = parcels.FieldSet.from_netcdf(filenames=filenames_stokes, variables=variables_stokes, dimensions=dimensions_stokes, indices=indices_stokes)

        #Converting units
        fieldset_stokes.U_wave_Stokes.units = parcels.tools.converters.GeographicPolar()
        fieldset_stokes.V_wave_Stokes.units = parcels.tools.converters.Geographic()

        #Adding fields to initial fields
        fieldset.add_field(fieldset_stokes.U_wave_Stokes)
        fieldset.add_field(fieldset_stokes.V_wave_Stokes)

        #Only required if more depth-averaged of depth-integrated Stokes drift is implemented
        if avg_stokes or di_stokes or di_stokes_biomass:
            fieldset.add_field(fieldset_stokes.wave_Tp)
            fieldset.add_constant('G', 9.81)  # Gravitational constant [m s-1]
            #fieldset.add_constant('depth_extent', depth_extent)  # depth extent of the sargassum [in m]

    if add_wind:

        ds = xr.open_mfdataset(f'/storage/shared/oceanparcels/input_data/ERA5/reanalysis-era5-single-level_wind10m_{year:04d}*.nc')

        filenames_wind = ds.sel(valid_time=slice(starttime, endtime))

        variables_wind = {'U_wind': 'u10', 
                        'V_wind':'v10'}

        dimensions_wind = {'lon':'longitude', 
                        'lat': 'latitude', 
                        'time': 'valid_time'}

        fieldset_wind= parcels.FieldSet.from_xarray_dataset(filenames_wind, variables_wind, dimensions_wind, mesh='spherical')

        fieldset_wind.add_periodic_halo(zonal=True)

        # windage_factor = 0.01

        # fieldset_wind.U_wind.set_scaling_factor(windage_factor)
        # fieldset_wind.V_wind.set_scaling_factor(windage_factor)

        fieldset_wind.U_wind.units = parcels.tools.converters.GeographicPolar()
        fieldset_wind.V_wind.units = parcels.tools.converters.Geographic()

        fieldset.add_field(fieldset_wind.U_wind)
        fieldset.add_field(fieldset_wind.V_wind)

    if add_growth:

        #Overall maximal growth rate (Corbin & Oxenford)
        fieldset.add_constant('MGR_SF3', 0.124) 
        fieldset.add_constant('MGR_SN1', 0.083)
        fieldset.add_constant('MGR_SN8', 0.053)
        #Set initial weight
        fieldset.add_constant('initial_weight', 50) #grams

        nitrogen_file_path = '/nethome/6903894/testing/Input_data_test/'
        #nitrogen_file = 'cmems_mod_glo_bgc-nut_anfc_0.25deg_P1D-m_1748337531153.nc'
        nitrogen_file = '2023_cmems_mod_glo_bgc-nut_anfc_0.25deg_P1D-m_1763121983211.nc'
        ds_N = xr.open_dataset(nitrogen_file_path + nitrogen_file)

        filename_N = ds_N.sel(time=slice(starttime, endtime))

        variables_N = {'no3': 'no3'}

        dimensions_N = {'lon':'longitude', 
                        'lat': 'latitude',
                        'depth' : 'depth',
                        'time': 'time'}

        fieldset_nitrogen = parcels.FieldSet.from_xarray_dataset(filename_N, variables=variables_N, dimensions=dimensions_N)
        
        #Setting interpolation method for no3 to be the inverse distance weighting interpolation method 
        #To account for no3 values being 0
        fieldset_nitrogen.no3.interp_method = "linear_invdist_land_tracer"

        #Adding fields to initial fields
        fieldset.add_field(fieldset_nitrogen.no3)

    ##################################################################################################
    #Defining particle set that we can advect on fieldset
    
    #image_name_path_C = '/storage/shared/oceanparcels/output_data/data_Elena/SATELLITE/C20241772024183.1KM.C_ATLANTIC.7DAY.L3D.FA_UNET_DENSITY.png'
    image_name_path_WC = "/nethome/6903894/testing/Input_data_test/C20230852023091.1KM.GCOOS.7DAY.L3D.FA_UNET_DENSITY.png"

    #sarg_lon_grid_C, sarg_lat_grid_C, amount_C = gi.sarg_grid_from_sat(image_name_path_C, 22.0, 0.0, -38.0, -63.0, coarse=True, as_pset=True)
    sarg_lon_grid_WC, sarg_lat_grid_WC, amount_WC = gi.sarg_grid_from_sat(image_name_path_WC, 31.0, 18.0, -79.0, -98.0, coarse=True, as_pset=True)
    
    # sarg_lon_grid = np.append(sarg_lon_grid_WC)
    # sarg_lat_grid = np.append(sarg_lat_grid_WC)
    sarg_lon_grid = sarg_lon_grid_WC
    sarg_lat_grid = sarg_lat_grid_WC

    sarg_depth_grid = np.zeros_like(sarg_lon_grid)

    #NUMBER OF PARTICLES
    #nparticles = len(grid_lon.tolist())
    #nparticles = len(sarg_lon_grid.tolist())
    nparticles=amount_WC

    class SargassumParticle(parcels.JITParticle):
        temperature =   parcels.Variable('temperature', dtype=np.float32, to_write=True, initial=0)
        salinity    =   parcels.Variable('salinity', dtype=np.float32, to_write=True, initial=0)
        depth_extent =  parcels.Variable('depth_extent', dtype=np.float32, to_write=True, initial=1)
        nitrogen =      parcels.Variable('nitrogen', dtype=np.float32, to_write=True, initial=0)
        biomass_SF3 =    parcels.Variable('biomass_SF3', dtype=np.float32, to_write=True, initial=1)
        biomass_SN1 =    parcels.Variable('biomass_SN1', dtype=np.float32, to_write=True, initial=1)
        biomass_SN8 =    parcels.Variable('biomass_SN8', dtype=np.float32, to_write=True, initial=1)
        biomass_loss =   parcels.Variable('biomass_loss', dtype=np.float32, to_write=True, initial=0)
        stranded =      parcels.Variable('stranded', dtype=np.float32, to_write=True, initial=0)
        limitation =    parcels.Variable('limitation', dtype=np.float32, to_write=True, initial=1)
        lim_salinity =   parcels.Variable('lim_salinity', dtype=np.float32, to_write=True, initial=0)
        lim_temp    =    parcels.Variable('lim_temp', dtype=np.float32, to_write=True, initial=0)
        lim_no3     =    parcels.Variable('lim_no3', dtype=np.float32, to_write=True, initial=0)
        speed_currents = parcels.Variable('speed_currents', dtype=np.float32, to_write=True, initial=0)
        speed_stokes =   parcels.Variable('speed_stokes', dtype=np.float32, to_write=True, initial=0)
        speed_wind =     parcels.Variable('speed_wind', dtype=np.float32, to_write=True, initial=0)
        decay_factor =   parcels.Variable('decay_factor', dtype=np.float32, to_write=True, initial=0)
        decay_averaged = parcels.Variable('decay_averaged', dtype=np.float32, to_write=True, initial=0 )
        decay_integrated_lower = parcels.Variable('decay_integrated_lower', dtype=np.float32, to_write=True, initial=0 )
        decay_integrated_upper = parcels.Variable('decay_integrated_upper', dtype=np.float32, to_write=True, initial=0 )
        wind_coefficient = parcels.Variable('wind_coefficient', dtype=np.float32, to_write=True, initial=0.01) 

    pset = parcels.ParticleSet.from_list(
        fieldset=fieldset,                  #the fields on which the particles are advected
        pclass = SargassumParticle,

            lon = sarg_lon_grid,            #vector of release longitudes
            lat = sarg_lat_grid,            #vector of release latitudes
            depth = sarg_depth_grid         #vector of release depths
    )

    ##################################################################################################
    #Creating output file
    for release_time in release_times:
        print(release_time)
        sim_file = output_file_name.format(year_start=release_time.year,
                                           month_start=release_time.month,
                                           day_start=release_time.day )
        output_file = pset.ParticleFile(
            name=  output_folder + sim_file,         #the file name
            outputdt=timedelta(hours=2),             #the time step of the outputs
            chunks = (nparticles, 50))               #per hoeveel tijdstappen je data wordt opgeslagen

    ##################################################################################################
    #Selecting kernels 
    kernels = [ ]

    if add_advection:
        kernels+= [parcels.AdvectionRK4]

    if add_stokes:
        if avg_stokes:
            kernels+= [ck.avg_Stokes_drift]
        elif di_stokes: 
            kernels+= [ck.di_Stokes_drift]
        elif di_stokes_biomass:
            kernels+= [ck.di_Stokes_drift_biomass_extent_dependency]
        else:
            kernels+= [ck.direct_Stokes_drift]

    if add_wind:
        kernels+= [ck.windage_drift]

    if add_growth:
        kernels+= [ck.sampling_from_field, ck.velocity_contribution]
        if add_biological_growth:
            kernels+= [ck.sargassum_biological_growth_model]

    if add_stranding:
        kernels+= [ck.stranding]

    ##################################################################################################
    #Executing the simulation
    pset.execute(
        kernels,                                    # the kernels (which define how particles move)
        runtime=timedelta(days=simulation_days),    # the total length of the run
        dt=dtime_execute,                           # the timestep of the kernel
        output_file=output_file,
    )
    print(f'Lagrangian simulation with starting month {month} is finished')
    ##################################################################################################

if __name__ == "__main__":
    run_multiple_simulations()