import os

import numpy as np

import parcels

import src.load_copernics_fieldset as load_copernics_fieldset  # noqa: E402
from src.saws_functions import sarg_grid_from_sat  # noqa: E402
from src.sargassum_kernels import SargassumParticle  # noqa: E402
import src.sargassum_kernels as sargassum_kernels  # noqa: E402

start_images = [
    {
        'image_name': "../SaWSdata/C20241772024183.1KM.C_ATLANTIC.7DAY.L3D.FA_UNET_DENSITY.png",
        'bbox': [22.0, 0.0, -63.0, -38.0],
    },
    {
        'image_name': "../SaWSdata/C20241772024183.1KM.CE_ATLANTIC.7DAY.L3D.FA_UNET_DENSITY.png",
        'bbox': [22.0, 0.0, -38.0, -11.5],
    }
]
release_lon, release_lat, _ = sarg_grid_from_sat(start_images, coarse=True)

fieldset = load_copernics_fieldset.create_fieldset(startmonth="2024-07")

for k_N in [0.001, 0.000129, 0.01]:
    filename = f"Simulations/Simulation_Satellite_kN_{k_N}.parquet"

    # Model parameters
    fieldset.z_upper = 0       # Upper depth extent of Sargassum (meter)
    fieldset.z_lower = 1       # Lower depth extent of Sargassum (meter)
    fieldset.wind_coeff = 0.01 # Windage coefficient (fraction)
    fieldset.mu_max = 0.095    # Maximum_growth_rate (doublings/day)
    fieldset.mort = 0.025      # Mortality relative loss/day
    fieldset.T_min = 20        # Minimum temperature (degC)
    fieldset.T_opt = 27.5      # Optimal temperature (degC)
    fieldset.T_max = 31        # Maximum temperature (degC)
    fieldset.S_opt = 36        # Optimal salinity (psu)
    fieldset.k_N = k_N         # Nitrogen half saturation constant (mmol/m3)

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
    )

    kernels = [
        parcels.kernels.AdvectionRK2,
        sargassum_kernels.di_Stokes_drift,
        sargassum_kernels.windage_drift,
        sargassum_kernels.stranding,
        sargassum_kernels.sargassum_biological_growth_model,
        sargassum_kernels.DeleteOutOfBounds,
    ]

    if not os.path.exists(filename):
        pset.execute(
            kernels,
            runtime=np.timedelta64(31, 'D'),
            dt=np.timedelta64(10, 'm'),
            output_file=pfile,
    )
