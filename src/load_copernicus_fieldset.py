import copernicusmarine
import os
import parcels
import xarray as xr
import numpy as np

DIRNAME = "copernicus_marine_data"

DATASET_IDs_BY_GRID = [
    (
        "physics",
        [
            ("cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m", ["uo", "vo"]),
            ("cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m", ["so"]),
            ("cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m", ["thetao"]),
        ],
    ),
    (
        "bgc",
        [
            ("cmems_mod_glo_bgc-nut_anfc_0.25deg_P1D-m", ["no3", "po4"]),
        ],
    ),
    (
        "waves",
        [
            ("cmems_mod_glo_wav_anfc_0.083deg_PT3H-i", ["VSDX", "VSDY", "VTPK"]),
        ],
    ),
    (
        "wind",
        [
            ("cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H", ["northward_wind", "eastward_wind"]),
        ],
    ),
]


def _get_filename(dataset_id, startdate, enddate):
    startdoy = f"{startdate.astype('datetime64[D]').astype(object).timetuple().tm_yday:03d}"
    enddoy = f"{enddate.astype('datetime64[D]').astype(object).timetuple().tm_yday:03d}"
    return f"{DIRNAME}/copernicus_{dataset_id}_{startdoy}_{enddoy}.nc"


def create_fieldset(startdate, enddate):

    #%% Downloading the datasets from CopernicusMarine
    datasets = {}
    for name, grid_datasets in DATASET_IDs_BY_GRID:

        filename = _get_filename(name, startdate, enddate)
        if not os.path.exists(filename):

            datasets[name] = {}
            datasets_list = [
                copernicusmarine.open_dataset(
                    dataset_id=dataset_id,
                    variables=used_vars,
                    minimum_longitude=-100,
                    maximum_longitude=16,
                    minimum_latitude=-25,
                    maximum_latitude=45,
                    start_datetime=np.datetime_as_string(startdate, unit="s"),
                    end_datetime=np.datetime_as_string(enddate, unit="s"),
                    minimum_depth=0.5,
                    maximum_depth=0.5,
                )
                for dataset_id, used_vars in grid_datasets
            ]
            ds = xr.merge(datasets_list)

            for var in ds.data_vars:
                ds[var] = ds[var].fillna(0)

            datasets[name] = parcels.convert.copernicusmarine_to_sgrid(
                fields={name: da for name, da in ds.data_vars.items()}
            )

# ### NOTE CODE BELOW IS BECAUSE STREAMING FROM COPERNICUSMARINE IS VERY SLOW (2 HRS instead of 2 minutes for a simulation).
# ### CAN BE REMOVED WHEN STREAMING IS FAST AGAIN
#%% Saving the datasets to local files for later use
            os.makedirs(DIRNAME, exist_ok=True)
            datasets[name].to_netcdf(filename)

#%% Loading the datasets from the local files instead of streaming from CopernicusMarine
    fset = []
    for name, _ in DATASET_IDs_BY_GRID:
        filename = _get_filename(name, startdate, enddate)
        ds = xr.open_dataset(filename, engine="netcdf4", chunks="auto")
        if name == "physics":
            vector_fields = {"UV": ("uo", "vo")}
        elif name == "waves":
            vector_fields = {"UVStokes": ("VSDX", "VSDY")}
        elif name == "wind":
            vector_fields = {"UVWind": ("northward_wind", "eastward_wind")}
        else:
            vector_fields = {}
        fset.append(
            parcels.FieldSet.from_sgrid_conventions(
                ds,
                vector_fields=vector_fields
            )
        )
### REPLACE WITH CODE BELOW

        # if "uo" in ds.data_vars and "vo" in ds.data_vars:
        #     vector_fields = {"UV": ("uo", "vo")}
        # elif "northward_wind" in ds.data_vars and "eastward_wind" in ds.data_vars:
        #     vector_fields = {"UVWind": ("northward_wind", "eastward_wind")}
        # elif "VSDX" in ds.data_vars and "VSDY" in ds.data_vars:
        #     vector_fields = {"UVStokes": ("VSDX", "VSDY")}
        # else:
        #     vector_fields = {}
        # fset = []
        # fset.append(
        #     parcels.FieldSet.from_sgrid_conventions(
        #         datasets[name],
        #         vector_fields=vector_fields
        #     )
        # )
### END NOTE

#%% Combining the fieldsets into a single fieldset
    fieldset = fset[0]
    for f in fset[1:]:
        fieldset += f

    fieldset.to_windowed_arrays()
    fieldset.describe()

    return fieldset
