"""행정동 상권 유형 분류.

1) 소상공인시장진흥공단 상가(상권)정보 스냅샷에서 서울 행정동별 업종 점포 수를 센다.
2) 2017~2019년 스냅샷 평균으로 유형을 고정한다 (분석 기간 이후 정보가 섞이지 않도록).

    python district_types.py counts <상가정보_zip_폴더> <출력.csv>
    python district_types.py types  <counts.csv> <출력.csv>

유형 규칙 (분위는 서울 행정동 안에서)
- 업무·점심형: 과학·기술 서비스 상위 20% 이면서 음식 상위 20%
- 야간여가형:  주점 상위 20% (업무·점심형 제외)
- 숙박·관광형: 숙박 상위 10%
- 주거·교육형: 음식 하위 40% 이면서 교육 상위 40%
- 그 밖
뒤 규칙이 앞 규칙을 덮어쓰며, 위 목록은 우선순위가 높은 순서다.
"""
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

MID_CATEGORIES = ["한식", "기타 간이", "주점", "비알코올", "일식", "중식", "서양식", "일반 숙박"]
BASE_YEARS = ("2017", "2018", "2019")


def zip_name(info):
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode("cp437").decode("cp949")
    except UnicodeError:
        return info.filename


def count_snapshot(path):
    with zipfile.ZipFile(path) as z:
        seoul = next(i for i in z.infolist() if "서울" in zip_name(i) and zip_name(i).lower().endswith(".csv"))
        raw = z.read(seoul)
    cols = ["상권업종대분류명", "상권업종중분류명", "행정동코드"]
    for enc in ("utf-8", "cp949"):
        try:
            d = pd.read_csv(io.BytesIO(raw), encoding=enc, usecols=cols, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    big = d.groupby(["행정동코드", "상권업종대분류명"]).size().unstack(fill_value=0)
    mid = (d[d["상권업종중분류명"].isin(MID_CATEGORIES)]
           .groupby(["행정동코드", "상권업종중분류명"]).size().unstack(fill_value=0).add_prefix("중_"))
    out = big.join(mid, how="left").fillna(0)
    out["전체"] = big.sum(axis=1)
    return out.reset_index().rename(columns={"행정동코드": "dong"})


def counts(src, out):
    rows = []
    for path in sorted(src.glob("sangga_*.zip")):
        snap = path.stem.split("_")[1]
        c = count_snapshot(path).assign(snap=snap)
        rows.append(c)
        print(snap, "행정동", len(c))
    pd.concat(rows, ignore_index=True).to_csv(out, index=False, encoding="utf-8-sig")


def classify(counts_csv):
    y = pd.read_csv(counts_csv, dtype={"snap": str})
    base = y[y["snap"].str[:4].isin(BASE_YEARS)].groupby("dong").mean(numeric_only=True)
    top = lambda col, p: base[col] >= base[col].quantile(p)

    office = top("과학·기술", 0.8) & top("음식", 0.8)
    t = pd.Series("그 밖", index=base.index)
    t[(base["음식"] <= base["음식"].quantile(0.4)) & top("교육", 0.6)] = "주거·교육형"
    t[top("숙박", 0.9)] = "숙박·관광형"
    t[top("중_주점", 0.8) & ~office] = "야간여가형"
    t[office] = "업무·점심형"

    base["type"] = t
    base["Q"] = pd.qcut(base["음식"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    return base


if __name__ == "__main__":
    mode, a, b = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    if mode == "counts":
        counts(a, b)
    else:
        types = classify(a)
        types.to_csv(b, encoding="utf-8-sig")
        print(types["type"].value_counts())
