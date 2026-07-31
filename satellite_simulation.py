import os
from datetime import datetime
import numpy as np

import parcels

import src.load_copernicus_fieldset as load_copernicus_fieldset  # noqa: E402
from src.saws_functions import release_points_from_SaWS_images  # noqa: E402
from src.sargassum_kernels import SargassumParticle  # noqa: E402
import src.sargassum_kernels as sargassum_kernels  # noqa: E402

startdate = datetime(2024, 7, 1)
enddate = datetime(2024, 8, 1)
coords = release_points_from_SaWS_images(startdate, stride=8)

fieldset = load_copernicus_fieldset.create_fieldset(startdate=np.datetime64(startdate), enddate=np.datetime64(enddate))

os.makedirs("Simulations", exist_ok=True)
for k_N, k_P in [(0.001, 0.0008), (0.000129, 0.0008), (0.01, 0.0008), (0.001, 0.8), (0.001, 0.008), (0.001, 0.0008)]:

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
    fieldset.k_N = k_N         # Nitrogen uptake half saturation [mmol/m3]
    fieldset.k_P = k_P         # Phosphorus uptake half saturation [mmol/m3]

    pset = parcels.ParticleSet(
        fieldset=fieldset,
        pclass=SargassumParticle,
        x=coords[:, 0],
        y=coords[:, 1],
        z=np.zeros_like(coords[:, 0]),
        t=np.datetime64('2024-07-01T00:00:00'),
    )

    # Do an initial sampling of the Biological Growth Model
    # to initialize the particle variables
    sargassum_kernels.SargassumBiologicalGrowthModel(pset, fieldset)

    # Now convert to windowed arrays for faster execution
    fieldset.to_windowed_arrays()
    fieldset.describe()

    pfile = parcels.ParticleFile(
        f"Simulations/Simulation_Satellite_kN_{k_N}_kP_{k_P}.parquet",
        outputdt=np.timedelta64(2, 'h'),
        mode="w",
    )
    fieldset.output_file = pfile  # for writing stranded particles

    kernels = [
        sargassum_kernels.Stranding,
        parcels.kernels.AdvectionRK2,
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
