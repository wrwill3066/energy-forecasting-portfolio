"""EPSIS(전력통계정보시스템) 조회 결과에서 연료원별 설비·거래량·SMP 표를 만든다.

EPSIS 조회 화면은 표 데이터를 페이지 안의 자바스크립트(gridData.push(...))로 내려준다.
화면별 응답 본문을 {"키": {"body": "<응답>"}} 형태의 JSON으로 저장해 두고 이 스크립트로 푼다.

    python epsis.py <원자료_폴더> <출력_폴더>

출력_폴더는 kpx_wind_hourly.py와 같은 곳을 쓴다 (시간별 풍력 파일을 읽어 월별 패널에 붙인다).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_IF = re.compile(r'if\s*\(\s*"([^"]*)"\s*==\s*"([^"]*)"\s*\)\s*\{(.*?)\}\s*else\s*\{(.*?)\}', re.S)
_TOKEN = re.compile(r'gridData\.push\(\s*\{(.*?)\}\s*\)|(\w+)\s*=\s*textFormmat\(\s*"([^"]*)"|(\w+)\s*=\s*"([^"]*)"\s*;', re.S)
_KV = re.compile(r'"?(\w+)"?\s*:\s*(textFormmat\(\s*"([^"]*)"[^)]*\)|"([^"]*)"|(\w+))')

FUEL = {"c1": "원자력", "c2": "유연탄", "c3": "무연탄", "c4": "유류", "c5": "LNG", "c6": "양수",
        "c91": "연료전지", "c92": "석탄가스화", "c93": "태양", "c94": "풍력", "c95": "수력", "c96": "해양",
        "c97": "바이오", "c98": "폐기물", "c7": "기타", "c8": "합계(전력시장)", "c9": "PPA계",
        "c10": "전력시장+PPA", "c23": "석탄계", "c912": "신재생계"}
REGION = {"강원도": "강원", "경기도": "경기", "경상남도": "경남", "경상북도": "경북",
          "전라남도": "전남", "전라북도": "전북", "제주도": "제주"}


def _resolve_if(s):
    """서버가 남긴 if ("a" == "b") {..} else {..} 분기를 문자열 비교로 미리 풀어 둔다."""
    while True:
        resolved = _IF.sub(lambda m: m.group(3) if m.group(1) == m.group(2) else m.group(4), s)
        if resolved == s:
            return s
        s = resolved


def parse(body, resolve=True):
    s = _resolve_if(body) if resolve else body
    env, rows = {}, []
    for m in _TOKEN.finditer(s):
        if m.group(1) is not None:
            row = {}
            for kv in _KV.finditer(m.group(1)):
                key = kv.group(1)
                if kv.group(3) is not None:
                    row[key] = kv.group(3)
                elif kv.group(4) is not None:
                    row[key] = kv.group(4)
                else:
                    row[key] = env.get(kv.group(5), kv.group(5))
            rows.append(row)
        elif m.group(2):
            env[m.group(2)] = m.group(3)
        else:
            env[m.group(4)] = m.group(5)
    return rows


def tidy(rows, value):
    df = pd.DataFrame(rows)
    df = df.melt(id_vars=["Period", "Region"], value_vars=[c for c in df.columns if c in FUEL],
                 var_name="code", value_name=value)
    df["fuel"] = df["code"].map(FUEL)
    df[value] = pd.to_numeric(df[value], errors="coerce")
    df["Region"] = df["Region"].replace({"소계": "전국", "합계": "전국"})
    return df.drop(columns="code")


def by_fuel(keys):
    frames = []
    for name, src, key in keys:
        frames.append(tidy(parse(src[key]["body"]), name).set_index(["Period", "Region", "fuel"]))
    return pd.concat(frames, axis=1).reset_index()


def monthly_by_fuel(capacity, market):
    df = by_fuel([
        ("발전설비용량_MW", capacity, "month_region"),
        ("시장참여설비_MW", market, "mkt_cap_month_region"),
        ("입찰량_GWh", market, "bid_month_region"),
        ("전력거래량_GWh", market, "trade_vol_month_region"),
        ("전력거래금액_억원", market, "trade_amt_month_region"),
    ])
    price = tidy(parse(market["unit_price_month_total"]["body"]), "정산단가_원per_kWh").assign(Region="전국")
    df = df.merge(price, on=["Period", "Region", "fuel"], how="left")
    df = df.rename(columns={"Period": "ym", "Region": "region"})
    df["ym"] = df["ym"].str.replace("/", "-")
    return df.sort_values(["ym", "region", "fuel"]), price


def annual_by_fuel(capacity, market):
    df = by_fuel([
        ("발전설비용량_MW", capacity, "year_region"),
        ("시장참여설비_MW", market, "mkt_cap_year_region"),
        ("입찰량_GWh", market, "bid_year_region"),
        ("전력거래량_GWh", market, "trade_vol_year_region"),
    ])
    for name, key in [("전력거래금액_억원", "trade_amt_year_total"), ("정산단가_원per_kWh", "unit_price_year_total")]:
        t = tidy(parse(market[key]["body"]), name).assign(Region="전국")
        df = df.merge(t, on=["Period", "Region", "fuel"], how="left")
    return df.rename(columns={"Period": "year", "Region": "region"}).sort_values(["year", "region", "fuel"])


def smp_monthly(market):
    smp = pd.DataFrame(parse(market["smp_month"]["body"]))
    smp = smp.rename(columns={"c1": "SMP_육지", "c2": "SMP_제주", "c3": "SMP_통합", "c4": "BLMP"})
    smp["ym"] = smp["Period"].str.replace("/", "-")
    return smp[["ym", "SMP_육지", "SMP_제주", "SMP_통합", "BLMP"]].sort_values("ym")


def official_cf(utilization):
    frames = []
    for fuel, key in [("풍력", "wind_month"), ("태양광", "solar_month")]:
        frames.append(pd.DataFrame(parse(utilization[key]["body"])).assign(source_fuel=fuel))
    cf = pd.concat(frames).rename(columns={"Period": "ym", "Region": "region_raw", "c1": "이용률_공식_pct"})
    cf["ym"] = cf["ym"].str.replace("/", "-")
    cf["region"] = cf["region_raw"].map(lambda s: REGION.get(s, s))
    cf["이용률_공식_pct"] = pd.to_numeric(cf["이용률_공식_pct"], errors="coerce")
    return cf[["ym", "region", "source_fuel", "이용률_공식_pct"]]


def wind_region_panel(monthly, price, cf, hourly_dir):
    """시도×월 풍력 패널: 설비, 거래량, 계량 발전량, 공식·계산 이용률."""
    w = monthly[monthly.fuel == "풍력"].drop(columns=["fuel", "정산단가_원per_kWh"])
    wind_price = price[price.fuel == "풍력"][["Period", "정산단가_원per_kWh"]].rename(columns={"Period": "ym"})
    wind_price["ym"] = wind_price["ym"].str.replace("/", "-")

    met = pd.read_csv(hourly_dir / "KR_wind_hourly_region_metered_2020_2025.csv")
    met["ym"] = met["date"].str[:7]
    met = pd.concat([met.groupby(["ym", "region"], as_index=False)["MWh"].sum(),
                     met.groupby("ym", as_index=False)["MWh"].sum().assign(region="전국")])
    met = met.rename(columns={"MWh": "계량발전량_MWh"})

    trd = pd.read_csv(hourly_dir / "KR_wind_hourly_region_traded_2017_2025.csv")
    trd["ym"] = trd["date"].str[:7]
    nat = pd.read_csv(hourly_dir / "KR_wind_hourly_national_2012_2025.csv")
    nat["ym"] = nat["date"].str[:7]
    trd = pd.concat([trd.groupby(["ym", "region"], as_index=False)["MWh"].sum(),
                     nat.groupby("ym", as_index=False)["wind_MWh_traded"].sum()
                        .rename(columns={"wind_MWh_traded": "MWh"}).assign(region="전국")])
    trd = trd.rename(columns={"MWh": "시간자료_거래량_MWh"})

    p = (w.merge(met, on=["ym", "region"], how="left")
          .merge(trd, on=["ym", "region"], how="left")
          .merge(cf[cf.source_fuel == "풍력"][["ym", "region", "이용률_공식_pct"]], on=["ym", "region"], how="left")
          .merge(wind_price, on="ym", how="left"))
    p.loc[p.region != "전국", "정산단가_원per_kWh"] = np.nan
    hours = pd.to_datetime(p["ym"] + "-01").dt.days_in_month * 24
    p["이용률_거래량기준_pct"] = 100 * p["전력거래량_GWh"] * 1000 / (p["시장참여설비_MW"] * hours)
    p["이용률_계량기준_pct"] = 100 * p["계량발전량_MWh"] / (p["시장참여설비_MW"] * hours)

    active = p[["발전설비용량_MW", "시장참여설비_MW", "전력거래량_GWh", "계량발전량_MWh"]].fillna(0).gt(0).any(axis=1)
    return p[active].sort_values(["region", "ym"])


def main(src, out):
    out.mkdir(parents=True, exist_ok=True)
    load = lambda name: json.loads((src / name).read_text(encoding="utf-8"))
    capacity = load("EPSIS_020100_연료원별설비.json")
    market = load("EPSIS_market_by_fuel.json")
    utilization = load("EPSIS_신재생이용률_지역별.json")

    monthly, price = monthly_by_fuel(capacity, market)
    cf = official_cf(utilization)
    panel = wind_region_panel(monthly, price, cf, out)

    monthly.to_csv(out / "KR_EPSIS_monthly_by_fuel_region_2002_2026.csv", index=False, encoding="utf-8-sig")
    annual_by_fuel(capacity, market).to_csv(out / "KR_EPSIS_annual_by_fuel_region_2001_2026.csv",
                                            index=False, encoding="utf-8-sig")
    smp_monthly(market).to_csv(out / "KR_SMP_monthly_2001_2026.csv", index=False, encoding="utf-8-sig")
    cf.to_csv(out / "KR_EPSIS_official_CF_wind_solar_region_monthly_2018_2025.csv", index=False, encoding="utf-8-sig")
    panel.to_csv(out / "KR_wind_monthly_region_panel.csv", index=False, encoding="utf-8-sig")
    print(panel[panel.region == "전국"].tail(12).round(1).to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
