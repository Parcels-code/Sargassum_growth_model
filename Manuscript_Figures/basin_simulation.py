import os

import numpy as np
import xarray as xr

import parcels

import src.load_copernics_fieldset as load_copernics_fieldset  # noqa: E402
from src.sargassum_kernels import SargassumParticle  # noqa: E402
import src.sargassum_kernels as sargassum_kernels  # noqa: E402

# Set release points based on the uo field
release_spacing = 2 #This is the spacing in the original grid (1/12 deg) at which we will select points for release.
ufile = xr.open_dataset('/Users/erik/Desktop/FromElena/copernicus_marine_data_cur.nc')
u_coarse = ufile.uo.isel(time=0, depth=0, longitude=slice(None, None, release_spacing), latitude=slice(None, None, release_spacing))
valid = u_coarse.notnull()
pts = valid.stack(points=("latitude", "longitude"))
release_lon = pts.longitude.where(pts, drop=True).values
release_lat = pts.latitude.where(pts, drop=True).values

fieldset = load_copernics_fieldset.create_fieldset()

#TODO these can be removed as fieldset constants
fieldset.add_constant('G', 9.81)  # Gravitational constant [m s-1]
# #Nitrogen half saturation constant
fieldset.add_constant('k_N', 0.001) #mmol/m3
#Overall maximal growth rate (Corbin & Oxenford)
fieldset.add_constant('MGR_SF3', 0.124)
fieldset.add_constant('MGR_SN1', 0.083)
fieldset.add_constant('MGR_SN8', 0.053)
#Set initial weight
fieldset.add_constant('initial_weight', 50) #grams

for type in ['Eulerian', 'Lagrangian']:
    filename = f"Simulation_Basin_{type}.zarr"


    pset = parcels.ParticleSet(
        fieldset=fieldset,
        pclass = SargassumParticle,
        lon = release_lon,
        lat = release_lat,
        z = np.zeros_like(release_lon),
        time = np.datetime64('2024-07-01T00:00:00'),
    )

    pfile = parcels.ParticleFile(
        filename,
        outputdt=np.timedelta64(2, 'h'),
        chunks = (len(release_lon), 50),
    )

    if type == 'Eulerian':
        kernels = [
            sargassum_kernels.sargassum_biological_growth_model,
        ]
    else:
        kernels = [
            parcels.kernels.AdvectionRK4,
            sargassum_kernels.di_Stokes_drift,
            sargassum_kernels.windage_drift,
            sargassum_kernels.sargassum_biological_growth_model,
            sargassum_kernels.stranding,
            sargassum_kernels.DeleteOutOfBounds,
        ]

    if not os.path.exists(filename):
        pset.execute(
            kernels,
            runtime=np.timedelta64(31, 'D'),
            dt=np.timedelta64(10, 'm'),
            output_file=pfile,
    )
