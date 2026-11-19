"""시간별 분석 패널을 만든다: 태양광·풍력 이용률, 전력수요, SMP (2015~2025).

    python panel.py <data_dir> <work_dir>

입력 파일 형식은 README의 '데이터' 절 참고.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NATIONAL = "전국"
SOLAR_REGIONS = ["서울_인천_경기", "원주_강원", "청주_충북", "대전_충남", "전주_전북",
                 "광주_전남", "대구_경북", "부산_경남", "제주"]
METER_SWITCH = "2024-06-01"
START, END = "2015-01-01", "2025-12-31"


def check_hourly(df, name, by=()):
    keys = [*by, "date", "he"]
    if not df["he"].between(1, 24).all():
        raise ValueError(f"{name}: he는 1~24여야 함")
    if df.duplicated(keys).any():
        raise ValueError(f"{name}: {keys} 중복")
    short = df.groupby([*by, "date"]).size().ne(24).sum()
    if short:
        print(f"[warn] {name}: 24시간이 아닌 날 {short}일")


def solar_by_region(data):
    gen = pd.read_csv(data / "solar_hourly.csv", parse_dates=["date"])
    cap = pd.read_csv(data / "solar_capacity_daily.csv", parse_dates=["date"])
    check_hourly(gen, "solar", by=["region"])

    out = {}
    for region, g in gen.groupby("region"):
        # 지역 BTM 설비는 2024년부터만 있어서 지역은 KPX+PPA, 전국만 BTM까지 더한다
        parts = ["KPX", "PPA", "BTM"] if region == NATIONAL else ["KPX", "PPA"]
        c = cap[cap["region"] == region].copy()
        g = g.assign(gen=g[parts].sum(axis=1, min_count=len(parts)))
        c["cap"] = c[[f"CAP_{p}" for p in parts]].sum(axis=1, min_count=len(parts))
        m = g.merge(c[["date", "cap"]], on="date", how="left", validate="many_to_one")
        m["cf"] = m["gen"] / m["cap"]
        out[region] = m.set_index(["date", "he"])

    nat = out[NATIONAL]
    c = cap[cap["region"] == NATIONAL].set_index("date")
    nat_kpx_ppa = nat[["KPX", "PPA"]].sum(axis=1, min_count=2)
    nat_cap = c[["CAP_KPX", "CAP_PPA"]].sum(axis=1, min_count=2)
    nat_cf_kpx_ppa = nat_kpx_ppa / nat_cap.reindex(nat.index.get_level_values(0)).to_numpy()
    return out, nat_cf_kpx_ppa


def wind_national(data):
    w = pd.read_csv(data / "wind_hourly_national.csv", parse_dates=["date"])
    check_hourly(w, "wind")
    # 2024-06부터 거래량에서 시장 밖 물량이 빠지므로 계량값으로 바꾼다.
    # 2019~2024 중첩 구간의 계량/거래 비율은 0.98~1.01이라 수준 단절은 1% 안쪽.
    w["gen"] = np.where(w["date"] >= METER_SWITCH, w["wind_MWh_metered"], w["wind_MWh_traded"])
    overlap = w[(w["date"] >= "2023-06-01") & (w["date"] < METER_SWITCH)]
    overlap = overlap.dropna(subset=["wind_MWh_traded", "wind_MWh_metered"])
    ratio = overlap["wind_MWh_metered"].sum() / overlap["wind_MWh_traded"].sum()
    print(f"전환 직전 12개월 계량/거래 비율: {ratio:.4f}")

    cap = pd.read_csv(data / "wind_capacity_daily.csv", parse_dates=["date"])
    w = w.merge(cap[["date", "cap_avg"]], on="date", how="left", validate="many_to_one")
    w["cf"] = w["gen"] / w["cap_avg"]
    return w.set_index(["date", "he"])[["gen", "cap_avg", "cf"]]


def wind_by_region(data):
    wr = pd.read_csv(data / "wind_hourly_region_metered.csv", parse_dates=["date"])
    check_hourly(wr, "wind region", by=["region"])
    wr["ym"] = wr["date"].dt.strftime("%Y-%m")
    cap = pd.read_csv(data / "wind_monthly_region_panel.csv", usecols=["ym", "region", "발전설비용량_MW"])
    wr = wr.merge(cap, on=["ym", "region"], how="left", validate="many_to_one")
    wr["cf"] = wr["MWh"] / wr["발전설비용량_MW"]
    print(f"지역 풍력 이용률 > 1 비율: {(wr['cf'] > 1).mean() * 100:.3f}%")
    return wr.pivot(index=["date", "he"], columns="region", values="cf")


def hourly_series(data, filename, column, name):
    df = pd.read_csv(data / filename, parse_dates=["date"])
    check_hourly(df, name)
    return df.set_index(["date", "he"])[[column]].astype(float)


def build(data, work):
    work.mkdir(parents=True, exist_ok=True)
    solar, nat_cf_kpx_ppa = solar_by_region(data)
    wind = wind_national(data)
    demand = hourly_series(data, "demand_hourly.csv", "demand_MW", "demand")
    smp = hourly_series(data, "smp_hourly.csv", "smp", "smp")

    s = solar[NATIONAL]
    nat = pd.DataFrame({"solar_cf": s["cf"], "solar_gen": s["gen"], "solar_cap": s["cap"]})
    nat = nat.join(wind.rename(columns={"gen": "wind_gen", "cap_avg": "wind_cap", "cf": "wind_cf"}), how="outer")
    nat = nat.join(demand, how="left").join(smp, how="left").loc[START:END]
    nat.to_parquet(work / "panel_national.parquet")
    s[["KPX", "PPA", "BTM"]].loc[START:END].to_parquet(work / "solar_parts.parquet")

    solar_reg = pd.DataFrame({r: solar[r]["cf"] for r in SOLAR_REGIONS})
    solar_reg["전국_KPX_PPA"] = nat_cf_kpx_ppa
    solar_reg.loc["2020-01-01":END].to_parquet(work / "panel_solar_region.parquet")
    wind_by_region(data).loc["2020-01-01":END].to_parquet(work / "panel_wind_region.parquet")

    yearly = nat.groupby(nat.index.get_level_values(0).year)[["solar_cf", "wind_cf"]].mean()
    print(yearly.round(4))


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]))
