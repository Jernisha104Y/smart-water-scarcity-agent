"""
Spatial Matching Foundation
===========================

Module to find nearest rainfall grid coordinates for CGWB groundwater observation points
using xarray's nearest-neighbor coordinate lookup on IMD NetCDF gridded data.

Assumptions:
1. Coordinate reference system (CRS) for CGWB observation coordinates and IMD NetCDF
   grid is standard WGS84 geographic lat/lon.
2. The IMD NetCDF spatial resolution is 0.25 deg x 0.25 deg spanning LATITUDE [6.5, 38.5]
   and LONGITUDE [66.5, 100.0].
3. The Tamil Nadu bounding box subset is sliced as LATITUDE=slice(8.0, 13.5) and
   LONGITUDE=slice(76.0, 80.5), which fully encloses all Tamil Nadu groundwater stations.
4. Nearest grid point selection is performed independently along the 1D LATITUDE and LONGITUDE
   coordinate axes using xarray's nearest-neighbor selection method="nearest".
"""

from pathlib import Path
from typing import Tuple
import pandas as pd
import xarray as xr

from src.config import PROCESSED_DIR, RAINFALL_DIR


def load_datasets() -> Tuple[pd.DataFrame, xr.Dataset, xr.Dataset]:
    """
    Load cleaned groundwater CSV and IMD rainfall NetCDF dataset,
    and create the Tamil Nadu bounding box subset.
    """
    gw_path = PROCESSED_DIR / "tamil_nadu_groundwater_clean.csv"
    rf_path = RAINFALL_DIR / "RF25_ind2024_rfp25.nc"

    if not gw_path.exists():
        raise FileNotFoundError(f"Cleaned groundwater dataset not found at: {gw_path}")
    if not rf_path.exists():
        raise FileNotFoundError(f"Rainfall NetCDF dataset not found at: {rf_path}")

    gw_df = pd.read_csv(gw_path)
    rainfall_ds = xr.open_dataset(rf_path)

    # Geographic subset covering Tamil Nadu extent
    tn_rainfall = rainfall_ds.sel(
        LATITUDE=slice(8.0, 13.5),
        LONGITUDE=slice(76.0, 80.5)
    )

    return gw_df, rainfall_ds, tn_rainfall


def find_nearest_rainfall_grid_point(
    lat: float,
    lon: float,
    tn_rainfall: xr.Dataset
) -> Tuple[float, float]:
    """
    Find the nearest rainfall grid point coordinates (latitude, longitude)
    for a given groundwater coordinate using xarray nearest-neighbor selection.
    """
    nearest_lat = float(
        tn_rainfall["LATITUDE"].sel(
            LATITUDE=lat,
            method="nearest"
        ).values
    )

    nearest_lon = float(
        tn_rainfall["LONGITUDE"].sel(
            LONGITUDE=lon,
            method="nearest"
        ).values
    )

    return nearest_lat, nearest_lon


def validate_first_observation() -> None:
    """
    Validation step: select first groundwater observation and map to nearest rainfall grid point.
    """
    gw_df, _, tn_rainfall = load_datasets()

    sample = gw_df.iloc[0]
    gw_lat = float(sample["Latitude"])
    gw_lon = float(sample["Longitude"])

    nearest_lat, nearest_lon = find_nearest_rainfall_grid_point(
        lat=gw_lat,
        lon=gw_lon,
        tn_rainfall=tn_rainfall
    )

    print("==================================================")
    print("SPATIAL MATCHING VALIDATION (First Observation)")
    print("==================================================")
    print("Groundwater location:")
    print(f"  State    : {sample.get('State', 'N/A')}")
    print(f"  District : {sample.get('District', 'N/A')}")
    print(f"  Block    : {sample.get('Block', 'N/A')}")
    print(f"  Village  : {sample.get('Village', 'N/A')}")
    print(f"  Latitude : {gw_lat}")
    print(f"  Longitude: {gw_lon}")
    print()
    print("Nearest rainfall grid point:")
    print(f"  Latitude : {nearest_lat}")
    print(f"  Longitude: {nearest_lon}")
    print("==================================================")


def validate_full_dataset_coverage() -> None:
    """
    Validation step: evaluate rainfall grid coverage for all groundwater observations.
    Does not modify the original groundwater CSV or fill missing values.
    """
    gw_df, _, tn_rainfall = load_datasets()

    total_gw_obs = len(gw_df)

    # Compute valid data counts per grid cell across TIME (366 days in 2024)
    valid_counts = tn_rainfall["RAINFALL"].notnull().sum(dim="TIME").to_pandas()

    # Vectorized nearest-coordinate mapping via xarray
    lats_da = xr.DataArray(gw_df["Latitude"].values, dims="points")
    lons_da = xr.DataArray(gw_df["Longitude"].values, dims="points")

    nearest_lats = tn_rainfall["LATITUDE"].sel(LATITUDE=lats_da, method="nearest").values
    nearest_lons = tn_rainfall["LONGITUDE"].sel(LONGITUDE=lons_da, method="nearest").values

    # Determine rainfall data availability for each observation
    status_list = []
    for lat, lon in zip(nearest_lats, nearest_lons):
        days_count = valid_counts.loc[lat, lon]
        if days_count == 366:
            status_list.append("complete")
        elif days_count == 0:
            status_list.append("unavailable")
        else:
            status_list.append("partial")

    mapped_df = pd.DataFrame({
        "Latitude": gw_df["Latitude"],
        "Longitude": gw_df["Longitude"],
        "nearest_lat": nearest_lats,
        "nearest_lon": nearest_lons,
        "rf_status": status_list
    })

    obs_complete = int((mapped_df["rf_status"] == "complete").sum())
    obs_unavailable = int((mapped_df["rf_status"] == "unavailable").sum())
    obs_partial = int((mapped_df["rf_status"] == "partial").sum())

    unique_cells = mapped_df.groupby(["nearest_lat", "nearest_lon", "rf_status"]).size().reset_index(name="obs_count")
    total_unique_cells = len(unique_cells)
    usable_unique_cells = len(unique_cells[unique_cells["rf_status"] == "complete"])
    unavailable_unique_cells = len(unique_cells[unique_cells["rf_status"] == "unavailable"])

    print("==================================================")
    print("FULL DATASET RAINFALL GRID COVERAGE VALIDATION")
    print("==================================================")
    print(f"Total groundwater observations: {total_gw_obs:,}")
    print(f"Observations mapped to cells with complete 2024 data: {obs_complete:,} ({obs_complete / total_gw_obs * 100:.2f}%)")
    print(f"Observations mapped to cells with no rainfall data   : {obs_unavailable:,} ({obs_unavailable / total_gw_obs * 100:.2f}%)")
    if obs_partial > 0:
        print(f"Observations mapped to cells with partial rainfall data: {obs_partial:,} ({obs_partial / total_gw_obs * 100:.2f}%)")
    print()
    print(f"Total unique rainfall grid cells used       : {total_unique_cells}")
    print(f"Unique usable (complete data) grid cells    : {usable_unique_cells}")
    print(f"Unique unavailable (no data) grid cells     : {unavailable_unique_cells}")
    print("==================================================")
    print("UNAVAILABLE RAINFALL GRID CELLS SELECTED BY OBSERVATIONS:")
    print("==================================================")
    print(f"{'Rainfall Latitude':<20}{'Rainfall Longitude':<20}{'Observations Mapped':<20}")
    print("-" * 60)

    unavail_cells_df = unique_cells[unique_cells["rf_status"] == "unavailable"].sort_values(
        by=["obs_count", "nearest_lat", "nearest_lon"], ascending=[False, True, True]
    )

    for _, row in unavail_cells_df.iterrows():
        print(f"{row['nearest_lat']:<20.2f}{row['nearest_lon']:<20.2f}{row['obs_count']:<20}")
    print("==================================================")


if __name__ == "__main__":
    validate_first_observation()
    print()
    validate_full_dataset_coverage()

