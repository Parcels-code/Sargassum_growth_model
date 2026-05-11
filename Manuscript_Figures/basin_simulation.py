import os

import numpy as np

import parcels

import src.load_copernics_fieldset as load_copernics_fieldset  # noqa: E402
from src.sargassum_kernels import SargassumParticle  # noqa: E402
import src.sargassum_kernels as sargassum_kernels  # noqa: E402

os.makedirs("Simulations", exist_ok=True)

for month in [7, 10, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12]:
    startmonth = f"2024-{month:02d}"
    filename = f"Simulations/Simulation_Basin_{startmonth}.parquet"

    if not os.path.exists(filename):
        print(f"Running simulation for {startmonth}...")
        fieldset = load_copernics_fieldset.create_fieldset(startmonth=startmonth)

        # Model parameters
        fieldset.z_upper = 0.0     # Upper depth extent of Sargassum raft [m]
        fieldset.z_lower = 1.0     # Lower depth extent of Sargassum raft [m]
        fieldset.wind_coeff = 0.01 # Windage coefficient [fraction]
        fieldset.mu_max = 0.095    # Maximum growth rate [doublings/day]
        fieldset.mort = 0.025      # Mortality rate [loss/day]
        fieldset.T_min = 20.0      # Minimum temperature [degC]
        fieldset.T_opt = 27.5      # Optimal temperature [degC]
        fieldset.T_max = 31.0      # Maximum temperature [degC]
        fieldset.S_opt = 36.0      # Optimal salinity [psu]
        fieldset.k_N = 0.001       # Nitrogen uptake half saturation [mmol/m3]

        # Set release points based on the uo field
        release_spacing = 6 #This is the spacing in the original grid (1/12 deg) at which we will select points for release.
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

        kernels = [
            sargassum_kernels.Stranding,
            sargassum_kernels.AdvectionRK2,
            sargassum_kernels.DepthIntegratedStokesDriftRK2,
            sargassum_kernels.WindageRK2,
            sargassum_kernels.SargassumBiologicalGrowthModel,
            sargassum_kernels.DeleteOutOfBounds,
        ]

        pset.execute(
            kernels=kernels,
            runtime=np.timedelta64(31, 'D'),
            dt=np.timedelta64(10, 'm'),
            output_file=pfile,
        )