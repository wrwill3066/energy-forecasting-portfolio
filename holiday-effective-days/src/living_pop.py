"""서울 생활인구(행정동, OA-14991) 월별 zip을 행정동×날짜 집계로 줄인다.

    python living_pop.py <zip_폴더> <출력_폴더>

zip마다 parquet 하나를 만들고, 이미 있으면 건너뛴다.
allday 하루 평균, lunch 11~13시, dinner 17~20시, work 9~17시 평균 (내국인 총생활인구)
"""
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

WINDOWS = {"lunch": (11, 13), "dinner": (17, 20), "work": (9, 17)}


def read_chunks(raw):
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, usecols=[0, 1, 2, 3],
                               chunksize=2_000_000, index_col=False)
        except UnicodeDecodeError:
            continue
    raise ValueError("인코딩을 알 수 없음")


def summarize(chunk):
    chunk.columns = ["date", "hr", "dong", "pop"]
    chunk = chunk.apply(pd.to_numeric, errors="coerce").dropna()
    chunk = chunk.drop_duplicates(["dong", "date", "hr"])
    agg = {"s_all": ("pop", "sum"), "n_all": ("pop", "count")}
    for name, (a, b) in WINDOWS.items():
        chunk[name] = chunk["pop"].where(chunk["hr"].between(a, b))
        agg[f"s_{name}"] = (name, "sum")
        agg[f"n_{name}"] = (name, "count")
    return chunk.groupby(["dong", "date"]).agg(**agg).reset_index()


def aggregate(zip_path):
    parts = []
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.filename.lower().endswith(".csv"):
                for chunk in read_chunks(z.read(info)):
                    parts.append(summarize(chunk))
    s = pd.concat(parts).groupby(["dong", "date"], as_index=False).sum()
    out = pd.DataFrame({"dong": s["dong"].astype(int), "date": s["date"].astype(int),
                        "allday": s["s_all"] / s["n_all"], "nh": s["n_all"]})
    for name in WINDOWS:
        out[name] = s[f"s_{name}"] / s[f"n_{name}"]
    return out[out["nh"] >= 22]


def main(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for zp in sorted(src.glob("LOCAL_PEOPLE_DONG_*.zip")):
        out = dst / zp.with_suffix(".parquet").name
        if out.exists():
            continue
        try:
            df = aggregate(zp)
        except zipfile.BadZipFile:
            print("손상된 파일:", zp.name)
            continue
        df.to_parquet(out)
        print(zp.name, "행정동", df["dong"].nunique(), "날짜", df["date"].nunique())


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
