"""전력거래소(KPX) 공공데이터로 풍력 시간별 발전량 패널을 만든다.

    python kpx_wind_hourly.py <원자료_폴더> <출력_폴더>

출력
- KR_wind_hourly_region_traded_2017_2025.csv   시도별 전력거래량 (MWh)
- KR_wind_hourly_region_metered_2020_2025.csv  시도별 5분 계량값을 시간 단위로 합친 값 (MWh)
- KR_wind_hourly_national_2012_2025.csv        전국 거래량·계량값

he는 그 시간이 끝나는 시각(1~24)이다.
"""
import sys
from pathlib import Path

import pandas as pd

TRADED_WIDE = [  # 지역별 태양광/풍력이 열로 나뉜 버전
    "DG15065269_한국전력거래소_지역별_시간별_태양광_발전량_20230228.csv",
    "DG15065269_한국전력거래소_지역별_시간별_태양광_및_풍력_발전량_20230531.csv",
    "DG15065269_한국전력거래소_지역별_시간별_태양광_및_풍력_발전량_20230831.csv",
    "DG15065269_한국전력거래소_지역별_시간별_태양광_및_풍력_발전량_20231130.csv",
]
TRADED_LONG = [  # 연료원이 행으로 들어간 버전
    "DG15065269_한국전력거래소_지역별_시간별_태양광_및_풍력_발전량_20230610.csv",
    "DG15065269_한국전력거래소_지역별_시간별_태양광_및_풍력_발전량_20241231.csv",
    "DG15065269_한국전력거래소_지역별_시간별_태양광_및_풍력_발전량_20251231.csv",
]
METERED = [
    ("DG15099452_한국전력거래소_지역별_5분단위_풍력_계량데이터_20241031.csv", "지역(광역시도)"),
    ("DG15099452_한국전력거래소_지역별_5분단위_풍력_계량데이터_20251231.csv", "지역"),
]
METERED_2019 = "DG15099452_한국전력거래소_5분_단위_풍력_계량데이터_20191231.csv"
NATIONAL_2012 = "DG15069372_한국전력거래소_시간별_신재생에너지_거래량_20220331.csv"

REGION = {"강원도": "강원", "경기도": "경기", "경상남도": "경남", "경상북도": "경북", "광주시": "광주",
          "대구시": "대구", "대전시": "대전", "부산시": "부산", "서울시": "서울", "세종시": "세종",
          "울산시": "울산", "인천시": "인천", "전라남도": "전남", "전라북도": "전북", "제주도": "제주",
          "충청남도": "충남", "충청북도": "충북", "제주": "제주", "육지": "육지"}


def read_csv(path):
    for enc in ("utf-8-sig", "cp949"):
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            df.columns = [c.strip() for c in df.columns]
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"인코딩을 알 수 없음: {path}")


def num(s):
    return pd.to_numeric(s.astype(str).str.strip().str.replace(",", ""), errors="coerce")


def traded_by_region(src):
    parts = []
    for name in TRADED_WIDE:
        d = read_csv(src / name)
        wind_col = next(c for c in d.columns if "풍력" in c)
        region_col = next(c for c in d.columns if c.startswith("지역"))
        parts.append(pd.DataFrame({"date": d.iloc[:, 0].astype(str).str.strip(),
                                   "he": num(d.iloc[:, 1]).astype(int),
                                   "region": d[region_col].str.strip().map(REGION),
                                   "MWh": num(d[wind_col])}))
    for name in TRADED_LONG:
        d = read_csv(src / name)
        d = d[d["연료원"].str.strip() == "풍력"]
        parts.append(pd.DataFrame({"date": d.iloc[:, 0].astype(str).str.strip(),
                                   "he": num(d.iloc[:, 1]).astype(int),
                                   "region": d["지역"].str.strip().map(REGION),
                                   "MWh": num(d.iloc[:, 4])}))
    out = pd.concat(parts)
    out["MWh"] = out["MWh"].fillna(0)
    return out.groupby(["date", "he", "region"], as_index=False)["MWh"].sum()


def metered_by_region(src):
    parts = []
    for name, region_col in METERED:
        d = read_csv(src / name)
        values = d[[c for c in d.columns if c.endswith("분")]].apply(num)
        parts.append(pd.DataFrame({
            "date": d["거래일"].astype(str).str.strip(),
            "he": num(d["시간"]).astype(int) + 1,
            "region": d[region_col].str.strip().map(REGION),
            "MWh": values.sum(axis=1, min_count=1) / 1e6,
            "n5min": values.notna().sum(axis=1),
            "max5min_MW": values.max(axis=1) * 12 / 1e6,
        }))
    return (pd.concat(parts)
            .groupby(["date", "he", "region"], as_index=False)
            .agg(MWh=("MWh", "sum"), n5min=("n5min", "sum"), max5min_MW=("max5min_MW", "max")))


def metered_national_2019(src):
    d = read_csv(src / METERED_2019)
    minutes = [c for c in d.columns if c.isdigit() or c.endswith("분")][-12:]
    return pd.DataFrame({"date": d["거래일"].astype(str).str.strip(),
                         "he": num(d["시간"]).astype(int) + 1,
                         "metered_2019": d[minutes].apply(num).sum(axis=1, min_count=1) / 1e6})


def national(src, traded, metered):
    d = read_csv(src / NATIONAL_2012)
    early = pd.DataFrame({"date": d["거래일"].astype(str).str.strip(),
                          "he": num(d["거래시간"]).astype(int),
                          "traded_file": num(d["풍력"])})
    traded_sum = traded.groupby(["date", "he"], as_index=False)["MWh"].sum().rename(columns={"MWh": "traded_sum"})
    metered_sum = metered.groupby(["date", "he"], as_index=False)["MWh"].sum().rename(columns={"MWh": "metered"})

    full = (early.merge(traded_sum, on=["date", "he"], how="outer")
                 .merge(metered_sum, on=["date", "he"], how="outer")
                 .merge(metered_national_2019(src), on=["date", "he"], how="left"))
    # 2017년부터는 지역별 거래량 합계를, 그 전은 전국 파일을 쓴다
    full["wind_MWh_traded"] = full["traded_sum"].where(full["date"] >= "2017-01-01", full["traded_file"])
    full["wind_MWh_metered"] = full["metered"].fillna(full["metered_2019"])
    return full[["date", "he", "wind_MWh_traded", "wind_MWh_metered"]].sort_values(["date", "he"])


def main(src, out):
    out.mkdir(parents=True, exist_ok=True)
    traded = traded_by_region(src)
    metered = metered_by_region(src)
    nat = national(src, traded, metered)

    traded.to_csv(out / "KR_wind_hourly_region_traded_2017_2025.csv", index=False, encoding="utf-8-sig")
    metered.to_csv(out / "KR_wind_hourly_region_metered_2020_2025.csv", index=False, encoding="utf-8-sig")
    nat.to_csv(out / "KR_wind_hourly_national_2012_2025.csv", index=False, encoding="utf-8-sig")

    summary = nat.groupby(nat["date"].str[:4])[["wind_MWh_traded", "wind_MWh_metered"]].sum() / 1000
    print(summary.round(0).rename(columns=lambda c: c.replace("MWh", "GWh")))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
