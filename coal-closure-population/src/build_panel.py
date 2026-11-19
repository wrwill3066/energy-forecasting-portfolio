"""비수도권 시·군 월별 인구·이동 패널과 연도별 출생·사망 패널.

    python build_panel.py <원자료_폴더> <출력_폴더>

원자료_폴더 구성
- jumin_age_month/age5_sigungu_YYYY.csv   행정안전부 주민등록 인구통계 (5세별, 월말)
- kosis_migration_monthly.csv              KOSIS DT_1B26001 인구이동 (월별, long 형식)
- kosis_vital/DT_1B8000I_*.json.gz         KOSIS DT_1B8000I 출생·사망 (연도별, API 응답)
"""
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROVINCE = {"강원도": "강원", "강원특별자치도": "강원", "충청북도": "충북", "충청남도": "충남",
            "전라북도": "전북", "전북특별자치도": "전북", "전라남도": "전남", "경상북도": "경북",
            "경상남도": "경남", "제주특별자치도": "제주", "제주도": "제주", "전남광주통합특별시": "전남"}
KOSIS_PROVINCE = {"32": "강원", "33": "충북", "34": "충남", "35": "전북", "36": "전남",
                  "37": "경북", "38": "경남", "39": "제주"}
# 경계·소속이 바뀌어 연속 비교가 안 되는 시군
DROP = {"충북 청주", "충북 청원", "경남 창원", "경남 마산", "경남 진해", "경북 군위", "충남 연기"}
AGE_BANDS = {"p0014": (0, 14), "p1529": (15, 29), "p3044": (30, 44), "p4564": (45, 64), "p65": (65, 200)}
MIGRATION = {"총전입": "in_tot", "총전출": "out_tot", "순이동": "net", "시도간전입": "in_inter",
             "시도간전출": "out_inter", "시도내이동-시군구간 전입": "in_intra", "시도내이동-시군구간 전출": "out_intra"}


def unit_key(name):
    """'강원특별자치도 태백시' -> '강원 태백'. 자치구·출장소·구 단위는 None."""
    parts = re.sub(r"\s+", " ", str(name)).strip().split()
    if len(parts) != 2 or parts[0] not in PROVINCE or "출장소" in name or parts[1].endswith("구"):
        return None
    key = PROVINCE[parts[0]] + " " + re.sub(r"(시|군)$", "", parts[1])
    return None if key in DROP else key


def population(src):
    frames = []
    for f in sorted((src / "jumin_age_month").glob("age5_sigungu_*.csv")):
        d = pd.read_csv(f, encoding="cp949", dtype=str)
        d = d.melt(id_vars=d.columns[0], var_name="col", value_name="v")
        d.columns = ["row", "col", "v"]
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)

    d["name"] = d["row"].str.extract(r"^(.*?)\s*\(\d{10}\)\s*$")[0]
    c = d["col"].str.extract(r"^(\d{4})년(\d{2})월_(계|남|여)_(.+)$")
    d["ym"], d["sex"], d["age"] = c[0] + "-" + c[1], c[2], c[3]
    d = d[d["sex"] == "계"].dropna(subset=["ym", "name"])
    d["k"] = d["name"].map(unit_key)
    d = d[d["k"].notna()]
    d["v"] = pd.to_numeric(d["v"].str.replace(",", ""), errors="coerce")
    d["lo"] = pd.to_numeric(d["age"].str.extract(r"^(\d+)")[0], errors="coerce")

    total = d[d["age"] == "총인구수"].groupby(["k", "ym"])["v"].sum().rename("pop")
    banded = d[d["lo"].notna() & ~d["age"].isin(["총인구수", "연령구간인구수"])]
    parts = [total] + [banded[banded["lo"].between(a, b)].groupby(["k", "ym"])["v"].sum().rename(name)
                       for name, (a, b) in AGE_BANDS.items()]
    p = pd.concat(parts, axis=1).reset_index()
    gap = (p[list(AGE_BANDS)].sum(axis=1) - p["pop"]).abs().max()
    print(f"인구: 시군 {p['k'].nunique()}, {p['ym'].min()}~{p['ym'].max()}, 연령합-총인구 최대차 {gap}")
    return p


def migration(src):
    a = pd.read_csv(src / "kosis_migration_monthly.csv", dtype={"지역코드": str})
    a["k"] = a["지역명"].map(unit_key)
    a = a[a["k"].notna() & a["변수"].isin(MIGRATION)]
    a["var"] = a["변수"].map(MIGRATION)
    return (a.pivot_table(index=["k", "기간"], columns="var", values="값", aggfunc="sum")
             .reset_index().rename(columns={"기간": "ym"}))


def vital(src, pop):
    rows = []
    for f in sorted((src / "kosis_vital").glob("DT_1B8000I_*.json.gz")):
        rows += json.load(gzip.open(f))
    v = pd.DataFrame(rows)
    items = {"출생건수 (명)": "births", "사망건수 (명)": "deaths"}
    v = v[(v["C1"].str.len() == 5) & v["C1"].str[:2].isin(KOSIS_PROVINCE) & v["ITM_NM"].isin(items)].copy()
    v["nm"] = v["C1_NM"].str.replace(r"\(.*\)", "", regex=True).str.strip()
    v = v[~v["nm"].str.contains(" ")]
    v["k"] = v["C1"].str[:2].map(KOSIS_PROVINCE) + " " + v["nm"].str.replace(r"(시|군)$", "", regex=True)
    v = v[~v["k"].isin(DROP)]
    v["var"] = v["ITM_NM"].map(items)
    v["DT"] = pd.to_numeric(v["DT"], errors="coerce")
    out = (v.pivot_table(index=["k", "PRD_DE"], columns="var", values="DT", aggfunc="sum")
            .reset_index().rename(columns={"PRD_DE": "year"}))
    out["year"] = out["year"].astype(int)

    dec = pop[pop["ym"].str.endswith("-12")].copy()
    dec["year"] = dec["ym"].str[:4].astype(int) + 1
    out = out.merge(dec[["k", "year", "pop"]].rename(columns={"pop": "pop_prev_dec"}), on=["k", "year"], how="left")
    return out[out["k"].isin(set(pop["k"]))]


def main(src, out):
    out.mkdir(parents=True, exist_ok=True)
    pop = population(src)
    panel = pop.merge(migration(src), on=["k", "ym"], how="left")
    panel["prov"] = panel["k"].str.split().str[0]
    panel.to_csv(out / "panel_month.csv", index=False, encoding="utf-8-sig")
    vital(src, pop).to_csv(out / "panel_year_vital.csv", index=False, encoding="utf-8-sig")

    units = panel.groupby("k").agg(prov=("prov", "first"), n=("ym", "size"), first=("ym", "min"), last=("ym", "max"))
    print(units["prov"].value_counts().to_dict(), "합계", len(units))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
