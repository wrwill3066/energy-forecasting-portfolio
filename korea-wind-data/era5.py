"""한반도 주변 ERA5 시간별 바람(10m·100m u/v)을 Google 공개 ARCO-ERA5에서 월 단위로 받는다.

    python era5.py <출력_폴더> [시작연도] [끝연도]

CDS API와 달리 대기열과 키가 없다. gcsfs 읽기가 가끔 멈추므로 하루씩 받아 재시도하고,
이미 받은 달은 건너뛴다. netCDF 라이브러리가 한글 경로를 못 여는 경우가 있어 임시 폴더에 쓴 뒤 옮긴다.
"""
import calendar
import shutil
import sys
import tempfile
import time
from pathlib import Path

import xarray as xr

STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
VARIABLES = ["100m_u_component_of_wind", "100m_v_component_of_wind",
             "10m_u_component_of_wind", "10m_v_component_of_wind"]
LAT, LON = slice(38.75, 32.5), slice(124, 131)


def open_store():
    ds = xr.open_zarr(STORE, chunks=None, storage_options={"token": "anon"}, consolidated=True)
    return ds[VARIABLES].sel(latitude=LAT, longitude=LON)


def load_day(ds, day, retries=5):
    for attempt in range(retries):
        try:
            return ds.sel(time=slice(f"{day}T00", f"{day}T23")).load()
        except Exception as e:
            print("retry", day, e, flush=True)
            time.sleep(20 * (attempt + 1))
    raise RuntimeError(f"{day} 다운로드 실패")


def download_month(ds, year, month, out):
    target = out / f"era5_wind_{year}{month:02d}.nc"
    if target.exists():
        return
    days = [f"{year}-{month:02d}-{d:02d}" for d in range(1, calendar.monthrange(year, month)[1] + 1)]
    data = xr.concat([load_day(ds, d) for d in days], "time")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_file = Path(tmp) / target.name
        data.to_netcdf(tmp_file, engine="netcdf4")
        shutil.move(tmp_file, target)
    print("done", year, month, flush=True)


def main(out, start=2015, end=2025):
    out.mkdir(parents=True, exist_ok=True)
    ds = open_store()
    for year in range(start, end + 1):
        for month in range(1, 13):
            download_month(ds, year, month, out)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(Path(args[0]), *(int(a) for a in args[1:3]))
