import os

import numpy as np
import xarray as xr

import parcels

import src.load_copernics_fieldset as load_copernics_fieldset  # noqa: E402
from src.sargassum_kernels import SargassumParticle  # noqa: E402
import src.sargassum_kernels as sargassum_kernels  # noqa: E402


for type in ['Eulerian', 'Lagrangian']:
    for month in range(1, 13):
        startmonth = f"2024-{month:02d}"
        filename = f"Simulations/Simulation_Basin_{startmonth}_{type}.parquet"

        if not os.path.exists(filename):
            print(f"Running simulation for {startmonth} ({type})...")
            fieldset = load_copernics_fieldset.create_fieldset(startmonth=startmonth)

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

            # Set release points based on the uo field
            release_spacing = 2 #This is the spacing in the original grid (1/12 deg) at which we will select points for release.
            u_coarse = fieldset.U.data.isel(time=0, depth=0, lon=slice(None, None, release_spacing), lat=slice(None, None, release_spacing))
            valid = u_coarse.notnull()
            pts = valid.stack(points=("lat", "lon"))
            release_lon = pts.lon.where(pts, drop=True).values
            release_lat = pts.lat.where(pts, drop=True).values

            pset = parcels.ParticleSet(
                fieldset=fieldset,
                pclass = SargassumParticle,
                lon = release_lon,
                lat = release_lat,
                z = np.zeros_like(release_lon),
                time = np.datetime64(f'{startmonth}-01T00:00:00'),
            )

            pfile = parcels.ParticleFile(
                filename,
                outputdt=np.timedelta64(2, 'h'),
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

            pset.execute(
                kernels,
                runtime=np.timedelta64(31, 'D'),
                dt=np.timedelta64(10, 'm'),
                output_file=pfile,
        )
