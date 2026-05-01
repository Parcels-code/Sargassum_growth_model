import copernicusmarine
import os
import parcels
import xarray as xr
import numpy as np

def create_fieldset(startmonth="2024-07"):
    # TODO make this on-the-fly access instead of downloading the file first and then loading it

    startdate = np.datetime64(f"{startmonth}-01T00:00:00")
    enddate = startdate + np.timedelta64(31, "D")
    start_datetime = np.datetime_as_string(startdate, unit="s")
    end_datetime = np.datetime_as_string(enddate, unit="s")
    dirname = "/home/evansebill/CopernicusMarine_data/copernicusmarine"

    DATASET_IDs = [
        "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
        "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
        "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
        "cmems_mod_glo_bgc-nut_anfc_0.25deg_P1D-m",
        "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i",
        "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H",
    ]
    variables = [["uo", "vo"], ["so"], ["thetao"], ["no3"], ["VSDX", "VSDY", "VTPK"], ["northward_wind", "eastward_wind"]]
    filenames = ["cur.nc", "so.nc", "thetao.nc", "no3.nc", "stokes.nc", "wind.nc"]
    for i in range(len(DATASET_IDs)):

        filename = f"{dirname}_{startmonth}_{filenames[i]}"
        if os.path.exists(filename):
                    print(f"File {filename} already exists, skipping...")
                    continue

        copernicusmarine.subset(
            dataset_id=DATASET_IDs[i],
            variables=variables[i],
            minimum_longitude=-75,
            maximum_longitude=-10,
            minimum_latitude=-4,
            maximum_latitude=24,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            minimum_depth=0.5,
            maximum_depth=0.5,
            output_filename=filename,
        )

    fields = []

    for i in range(len(filenames)):
        filename = f"{dirname}_{startmonth}_{filenames[i]}"

        print(f"Loading {filename}")
        ds_fields = xr.open_mfdataset(filename, combine="by_coords")

        # TODO check how we can get good performance without loading full dataset in memory
        ds_fields.load()  # load the dataset into memory

        # TODO clean up this part, maybe make it more generic and less hardcoded
        if "cur" in filename:
            flds = {"U": ds_fields["uo"], "V": ds_fields["vo"]}
        elif "so" in filename:
            flds = {"so": ds_fields["so"]}
        elif "thetao" in filename:
            flds = {"thetao": ds_fields["thetao"]}
        elif "no3" in filename:
            flds = {"no3": ds_fields["no3"]}
        elif "stokes" in filename:
            ds_fields = ds_fields.expand_dims(dim={"depth": [0]})
            flds = {"VSDX": ds_fields["VSDX"], "VSDY": ds_fields["VSDY"], "VTPK": ds_fields["VTPK"]}
        elif "wind" in filename:
            ds_fields = ds_fields.expand_dims(dim={"depth": [0]})
            flds = {"northward_wind": ds_fields["northward_wind"], "eastward_wind": ds_fields["eastward_wind"]}

        ds_fset = parcels.convert.copernicusmarine_to_sgrid(fields=flds)
        fset = parcels.FieldSet.from_sgrid_conventions(ds_fset, mesh="spherical")

        for fld in fset.fields.values():
            fields.append(fld)
    fieldset = parcels.FieldSet(fields)


    # making vector fields of wind and stokes
    fieldset.add_field(parcels.VectorField("UVStokes", fieldset.VSDX, fieldset.VSDY, vector_interp_method=parcels.interpolators.XLinear_Velocity))
    fieldset.add_field(parcels.VectorField("UVWind", fieldset.northward_wind, fieldset.eastward_wind, vector_interp_method=parcels.interpolators.XLinear_Velocity))

    # setting NaN values to 0 for vector fields
    for field in ["U", "V", "VSDX", "VSDY", "northward_wind", "eastward_wind"]:
        getattr(fieldset, field).data = getattr(fieldset, field).data.fillna(0)

    # TODO check if NaN setting also needed?
    fieldset.no3.data = fieldset.no3.data.fillna(0)
    fieldset.thetao.data = fieldset.thetao.data.fillna(0)
    fieldset.so.data = fieldset.so.data.fillna(0)
    fieldset.VTPK.data = fieldset.VTPK.data.fillna(0)

    #setting other interpolators to XLinearInvdistLandTracer,
    for field in [fieldset.so, fieldset.thetao, fieldset.no3, fieldset.VTPK]:
        field.interp_method = parcels.interpolators.XLinearInvdistLandTracer

    return fieldset
