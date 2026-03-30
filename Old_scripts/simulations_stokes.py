##################################################################################################
#Importing relevent packages
import sys; print(sys.executable)
import parcels
import math
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature
from datetime import datetime, timedelta

#Importing file in which kernels are defined
import custom_kernels as ck

##################################################################################################
#SETTINGS
#Time-related settings
starttime = datetime(2024,7,1)
endtime = datetime(2024,8,2)
dtime_data = timedelta(days=1)    #je stelt hier handmatig in wat de tijdsresolutie is van je data #deze kan niet naar minuten (13/5/2025)
dtime_execute = timedelta(minutes=10)
simulation_days = 31  

#Selecting physical variables to add
add_stokes = True
add_wind = True
add_growth = True

#Select type of Stokes drift (default is surface Stokes drift)
avg_stokes = False
di_stokes = True

#Selecting specific type of growth (default is growth according to weak temperature limitation of Jouanno et al. 2021)
add_temp_limitation_J25 = True
add_growth_according_to_Bonner_2024 = False

#Output storing
output_folder = "/storage/shared/oceanparcels/output_data/data_Elena/"
output_file_name = "Simulations_part2/SIM_11sept_di_check3.zarr"

##################################################################################################
#Loading the physical dataset and matching the grid
directory_phy =  '/storage/shared/oceanparcels/input_data/MOi/'
phy_base_file =  directory_phy + 'GLO12/psy4v3r1-daily_{vector:s}_{y:04d}-{m:02d}-{d:02d}.nc' 
phy_files_U = []
phy_files_V = []
phy_files_T = []

time = starttime
phy_file_W = directory_phy + 'GLO12/psy4v3r1-daily_{vector:s}_{y:04d}-{m:02d}-{d:02d}.nc'.format(vector = 'W', y = time.year, m = time.month, d = time.day)
while(time < endtime):
    phy_files_U.append(phy_base_file.format(vector = 'U', y = time.year, m = time.month, d = time.day))
    phy_files_V.append(phy_base_file.format(vector = 'V', y = time.year, m = time.month, d = time.day))
    phy_files_T.append(phy_base_file.format(vector = 'T', y = time.year, m = time.month, d = time.day))
    time+=dtime_data

mesh_file_h = directory_phy + "domain_ORCA0083-N006/PSY4V3R1_mesh_hgr.nc"   #Hiermee converteren we het grid!
mesh_file_z = directory_phy + "domain_ORCA0083-N006/PSY4V3R1_mesh_zgr.nc"
filenames_phy = {'U': {
        'lon': mesh_file_h,
        'lat': mesh_file_h,
       'depth':phy_file_W,
        'data': phy_files_U,
    }, 'V': {
        'lon': mesh_file_h,
        'lat': mesh_file_h,
       'depth':phy_file_W,
        'data': phy_files_V,
    }, 'T': {'lon': mesh_file_h,
        'lat': mesh_file_h,
       'depth':phy_file_W,
        'data': phy_files_T,
    }
}
variables_phy = {'U': 'vozocrtx',
                'V': 'vomecrty',
                'T': 'votemper'}

c_grid_dimensions = {   'lat': 'gphif',
                        'lon': 'glamf',
                        'depth':'depthw',
                        'time': 'time_counter'}

dimensions_phy = {'U': c_grid_dimensions,
                 'V': c_grid_dimensions,
                 'T': c_grid_dimensions}

#Defining a range of indices to load as fieldset to reduce computational time
indices = {'lat': range(800,2000),
           'lon': range(2300,3500)
           }

#Creating fieldset
fieldset = parcels.FieldSet.from_nemo(filenames=filenames_phy,variables=variables_phy,dimensions=dimensions_phy, indices=indices)
print('fieldset = made')

#Adding the wave/stokes data
if add_stokes:
    depth_extent = 1 
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
    indices_stokes = {'lat': range(650,1750),
                       'lon': range(800,2500)
                     } 
    
    fieldset_stokes = parcels.FieldSet.from_netcdf(filenames=filenames_stokes, variables=variables_stokes, dimensions=dimensions_stokes, indices=indices_stokes)

    #Converting units
    fieldset_stokes.U_wave_Stokes.units = parcels.tools.converters.GeographicPolar()
    fieldset_stokes.V_wave_Stokes.units = parcels.tools.converters.Geographic()

    #Adding fields to initial fields
    fieldset.add_field(fieldset_stokes.U_wave_Stokes)
    fieldset.add_field(fieldset_stokes.V_wave_Stokes)

    #Only required if more depth-averaged of depth-integrated Stokes drift is implemented
    if avg_stokes or di_stokes:
        fieldset.add_field(fieldset_stokes.wave_Tp)
        fieldset.add_constant('G', 9.81)  # Gravitational constant [m s-1]
        fieldset.add_constant('depth_extent', depth_extent)  # depth extent of the sargassum [in m]

if add_wind:

    ds = xr.open_mfdataset('/storage/shared/oceanparcels/input_data/ERA5/reanalysis-era5-single-level_wind10m_2024*.nc')

    filenames_wind = ds.sel(valid_time=slice(starttime, endtime))

    variables_wind = {'U_wind': 'u10', 
                      'V_wind':'v10'}

    dimensions_wind = {'lon':'longitude', 
                       'lat': 'latitude', 
                       'time': 'valid_time'}

    fieldset_wind= parcels.FieldSet.from_xarray_dataset(filenames_wind, variables_wind, dimensions_wind, mesh='spherical')

    fieldset_wind.add_periodic_halo(zonal=True)

    windage_factor = 0.01

    fieldset_wind.U_wind.set_scaling_factor(windage_factor)
    fieldset_wind.V_wind.set_scaling_factor(windage_factor)

    fieldset_wind.U_wind.units = parcels.tools.converters.GeographicPolar()
    fieldset_wind.V_wind.units = parcels.tools.converters.Geographic()

    fieldset.add_field(fieldset_wind.U_wind)
    fieldset.add_field(fieldset_wind.V_wind)

if add_growth:
    #Overall average growth rate (Corbin & Oxenford)
    fieldset.add_constant('RGR_SF3', 0.077) 
    fieldset.add_constant('RGR_SN1', 0.046)
    fieldset.add_constant('RGR_SN8', 0.032)
    #Overall maximal growth rate (Corbin & Oxenford)
    fieldset.add_constant('MGR_SF3', 0.124) 
    fieldset.add_constant('MGR_SN1', 0.083)
    fieldset.add_constant('MGR_SN8', 0.053)
    #Set initial weight
    fieldset.add_constant('initial_weight', 50) #grams

    nitrogen_file_path = '/nethome/6903894/testing/Input_data_test/'
    nitrogen_file = 'cmems_mod_glo_bgc-nut_anfc_0.25deg_P1D-m_1748337531153.nc'
    ds_N = xr.open_dataset(nitrogen_file_path + nitrogen_file)

    filename_N = ds_N.sel(time=slice(starttime, endtime))

    variables_N = {'no3': 'no3'}

    dimensions_N = {'lon':'longitude', 
                       'lat': 'latitude',
                       'depth' : 'depth',
                       'time': 'time'}

    fieldset_nitrogen = parcels.FieldSet.from_xarray_dataset(filename_N, variables=variables_N, dimensions=dimensions_N)

    #Adding fields to initial fields
    fieldset.add_field(fieldset_nitrogen.no3)

##################################################################################################
#Defining particle set that we can advect on fieldset
#nparticles = 16   #how many particles will be advected

#Defining longitudes and latitudes for particle release
# release_lon1 = np.linspace(-56, -55, 6)  # 6 points in longitude
# release_lat1 = np.linspace(15, 16, 6)    # 6 points in latitude

# release_lon2 = np.linspace(-56, -55, 6)
# release_lat2 = np.linspace(9, 10, 6)

# #Creating meshgrids for areas
# lon_grid1, lat_grid1 = np.meshgrid(release_lon1, release_lat1)
# lon_grid2, lat_grid2 = np.meshgrid(release_lon2, release_lat2)

# #Flattening and combining
# lon_array = np.concatenate([lon_grid1.ravel(), lon_grid2.ravel()])
# lat_array = np.concatenate([lat_grid1.ravel(), lat_grid2.ravel()])

#Defining longitudes and latitudes for particle release
# release_lon1 = np.linspace(-57, -48, 9*5+1)  # 6 points in longitude
# release_lat1 = np.linspace(7, 12, 5*5+1)    # 6 points in latitude

# release_lon2 = np.linspace(-47.8, -39, 9*5)
# release_lat2 = np.linspace(3, 8, 5*5+1)

# release_lon3 = np.linspace(-38.8, -30, 9*5)
# release_lat3 = np.linspace(-1, 4, 5*5+1)

# #Creating meshgrids for areas
# lon_grid1, lat_grid1 = np.meshgrid(release_lon1, release_lat1)
# lon_grid2, lat_grid2 = np.meshgrid(release_lon2, release_lat2)
# lon_grid3, lat_grid3 = np.meshgrid(release_lon3, release_lat3)

# #Flattening and combining
# lon_array = np.concatenate([lon_grid1.ravel(), lon_grid2.ravel(), lon_grid3.ravel()])
# lat_array = np.concatenate([lat_grid1.ravel(), lat_grid2.ravel(), lat_grid3.ravel()])

#print("lon =", lon.tolist())
#print("lat =", lat.tolist())

#nparticles = len(lon_array.tolist())

nparticles = 16

class SargassumParticle(parcels.JITParticle):
    temperature =   parcels.Variable('temperature', dtype=np.float32, to_write=True, initial=0)
    depth_extent =  parcels.Variable('depth_extent', dtype=np.float32, to_write=True, initial=1)
    nitrogen =      parcels.Variable('nitrogen', dtype=np.float32, to_write=True, initial=0)
    weight_SF3 =    parcels.Variable('weight_SF3', dtype=np.float32, to_write=True, initial=50)
    weight_SN1 =    parcels.Variable('weight_SN1', dtype=np.float32, to_write=True, initial=50)
    weight_SN8 =    parcels.Variable('weight_SN8', dtype=np.float32, to_write=True, initial=50)
    k_p         =   parcels.Variable('k_p', dtype=np.float32, to_write=True, initial=0)
    decay_averaged = parcels.Variable('decay_averaged', dtype=np.float32, to_write=True, initial=0 )
    decay_integrated_lower = parcels.Variable('decay_integrated_lower', dtype=np.float32, to_write=True, initial=0 )
    decay_integrated_upper = parcels.Variable('decay_integrated_upper', dtype=np.float32, to_write=True, initial=0 )

pset = parcels.ParticleSet.from_list(
      fieldset=fieldset,  # the fields on which the particles are advected
      pclass = SargassumParticle,
      lon=[-43, -43, -40, -40, 
           -20, -20, -17, -17,
           -55, -55, -52, -52,
           -23, -23, -20, -20], #vector of release longitudes
      
      lat=[2,  5,  2, 5,
           -2, 1, -2, 1,
           13, 16, 13, 16,
           9, 12, 9, 12],       #vector of release latitudes

        #lon = lon_array,            #vector of release longitudes
        #lat = lat_array,            #vector of release latitudes

        depth = [0] * nparticles    #vector of release depths
  )

##################################################################################################
#Creating output file
output_file = pset.ParticleFile(
    name=  output_folder + output_file_name, # the file name
    outputdt=timedelta(hours=2),             # the time step of the outputs
    chunks = (nparticles, 50))               #per hoeveel tijdstappen je data wordt opgeslagen

##################################################################################################
#Selecting kernels with parcels.AdvectionRK4 as default
kernels = [ parcels.AdvectionRK4 ]

if add_stokes:
    if avg_stokes:
        kernels+= [ck.avg_Stokes_drift]
    elif di_stokes: 
        kernels+= [ck.di_Stokes_drift]
    else:
        kernels+= [ck.direct_Stokes_drift]

if add_wind:
    kernels+= [ck.wind_drag]

if add_growth:
    kernels+= [ck.temperature_from_field, ck.nitrogen_from_field]
    if add_temp_limitation_J25:
        kernels+= [ck.growth_J25]
    elif add_growth_according_to_Bonner_2024:
        kernels+= [ck.growth_Bonner]
    else:
        kernels+= [ck.growth_temp_based]

##################################################################################################
#Executing the simulation
pset.execute(
    kernels,                                    # the kernels (which define how particles move)
    runtime=timedelta(days=simulation_days),    # the total length of the run
    dt=dtime_execute,                                   # the timestep of the kernel
    output_file=output_file,
)
##################################################################################################
