"""기상자료개방포털 ASOS 시간자료 묶음(zip 안에 zip)을 지점×시간 CSV 하나로 합친다.

    python asos.py <출력.csv.gz> <다운로드.zip> [다운로드.zip ...]

zip은 몇 겹으로 싸여 있어도 되고, 안쪽 CSV(cp949)를 모두 읽는다.
"""
import io
import sys
import zipfile

import pandas as pd

COLUMNS = {"지점": "stn", "일시": "datetime", "기온(°C)": "temp_c", "강수량(mm)": "precip_mm",
           "풍속(m/s)": "wind_ms", "풍향(16방위)": "wind_dir", "습도(%)": "rh_pct", "이슬점온도(°C)": "dewpoint_c",
           "해면기압(hPa)": "slp_hpa", "일조(hr)": "sunshine_hr", "일사(MJ/m2)": "solar_mj", "적설(cm)": "snow_cm",
           "전운량(10분위)": "cloud_total", "중하층운량(10분위)": "cloud_low", "시정(10m)": "vis_10m"}
STATIONS = {100: "대관령", 108: "서울", 112: "인천", 114: "원주", 119: "수원", 130: "울진", 133: "대전",
            138: "포항", 143: "대구", 156: "광주", 159: "부산", 165: "목포", 169: "흑산도", 184: "제주", 216: "태백"}


def read_nested(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for name in z.namelist():
            data = z.read(name)
            if name.lower().endswith(".zip"):
                yield from read_nested(data)
            elif name.lower().endswith(".csv"):
                yield pd.read_csv(io.BytesIO(data), encoding="cp949", low_memory=False)


def main(out, sources):
    frames = []
    for src in sources:
        with open(src, "rb") as f:
            frames.extend(read_nested(f.read()))
    d = pd.concat(frames, ignore_index=True)[list(COLUMNS)].rename(columns=COLUMNS)
    d["datetime"] = pd.to_datetime(d["datetime"])
    d = d.drop_duplicates(["stn", "datetime"]).sort_values(["stn", "datetime"])
    d.insert(1, "stn_name", d["stn"].map(STATIONS))
    d.to_csv(out, index=False, encoding="utf-8-sig", compression="gzip")

    coverage = d.assign(year=d["datetime"].dt.year).groupby("stn_name")[["wind_ms", "solar_mj", "cloud_total"]]
    print(d.shape, d["datetime"].min(), d["datetime"].max())
    print((coverage.apply(lambda g: g.notna().mean()) * 100).round(0))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
