"""평일 공휴일의 상권 유형별 유효일수(ED) 추정.

    python effective_days.py <living_pop_parquet_폴더> <types.csv> <출력_폴더>

식(1) 유형별:  ln P_it = Σ_c γ_c C_t^c + μ_i + δ_요일 + λ_연월 + ε,   ED_c = exp(γ_c)
식(4) 상호작용: ln P_it = Σ_c Σ_g θ_cg (C_t^c × G_i^g) + μ_i + τ_날짜 + ε   (기준: 그 밖)

P는 점심(11~13시) 내국인 생활인구, 표본은 평일. 표준오차는 행정동 + ISO 주 이원 군집.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

from kr_calendar import calendar, holidays, off_blocks

TYPES = ["업무·점심형", "숙박·관광형", "야간여가형", "주거·교육형", "그 밖"]
OFFICE = "업무·점심형"
EVENTS = ["Hol_single", "Hol_first", "Hol_mid", "Hol_last", "Nat", "NRm1_", "NRp1_", "Sand",
          "Elec", "Temp", "Labor", "NRm24", "NRp24"]
LABELS = {"Hol_single": "비명절 단독", "Hol_first": "연휴 첫날", "Hol_last": "연휴 마지막날",
          "Nat": "평일 명절", "NRm1_": "명절 전날", "NRp1_": "명절 다음날", "Sand": "징검다리 평일"}
VCOV = {"CRV1": "dong+week"}
SUBSTITUTE_DAYS = set(pd.to_datetime(["2017-01-30", "2017-10-06", "2018-09-26", "2020-01-27",
                                      "2022-09-12", "2023-01-24", "2024-02-12", "2025-10-08"]))


def load_panel(pop_dir, types_csv):
    types = pd.read_csv(types_csv, index_col=0)["type"]
    types.index = types.index.astype(int)

    pop = pd.concat(pd.read_parquet(f) for f in sorted(pop_dir.glob("*.parquet")))
    pop = pop.groupby(["dong", "date"], as_index=False)["lunch"].mean()
    pop["date"] = pd.to_datetime(pop["date"].astype(int).astype(str))
    pop = pop[pop["dong"].isin(types.index) & (pop["lunch"] > 0)]

    cal = calendar().reset_index()
    d = pop.merge(cal, on="date")
    d["type"] = d["dong"].map(types)
    d["W"] = (d["type"] == OFFICE).astype(int)
    d["grp"] = np.where(d["W"] == 1, "W", "O")
    d["ly"] = np.log(d["lunch"])
    d["wd"] = d["date"].dt.dayofweek.astype(str)
    d["ym"] = d["date"].dt.strftime("%Y%m")
    d["dfe"] = d["date"].dt.strftime("%Y%m%d")
    d["week"] = d["date"].dt.strftime("%G%V")
    d["NRm1_"], d["NRp1_"] = d["NRm1"], d["NRp1"]
    d["NRm24"] = d[["NRm2", "NRm3", "NRm4"]].sum(axis=1)
    d["NRp24"] = d[["NRp2", "NRp3", "NRp4"]].sum(axis=1)
    d["HolL"] = d[["Hol_single", "Hol_first", "Hol_mid", "Hol_last"]].sum(axis=1)
    return d[d["date"].dt.dayofweek < 5].copy()


def present(d, events):
    return [e for e in events if d[e].sum() > 0]


def fit_by_type(d, events=EVENTS):
    return pf.feols("ly ~ " + "+".join(present(d, events)) + " | dong + wd + ym", data=d, vcov=VCOV)


def fit_pooled(d, events=EVENTS):
    """업무·점심형과 그 밖을 한 모형에 넣고, 각 날짜유형에 업무형 상호작용을 붙인다."""
    d = d.copy()
    xs = []
    for c in present(d, events):
        d[f"{c}_W"] = d[c] * d["W"]
        xs += [c, f"{c}_W"]
    return pf.feols("ly ~ " + "+".join(xs) + " | dong + wd^grp + ym^grp", data=d, vcov=VCOV)


def psd(V):
    w, Q = np.linalg.eigh((V + V.T) / 2)
    return (Q * np.clip(w, 0, None)) @ Q.T


def weights(m, w):
    names = list(m.coef().index)
    g = np.zeros(len(names))
    for k, v in w.items():
        g[names.index(k)] = v
    return g


def combo(m, w):
    g = weights(m, w)
    return float(g @ m.coef().values), float(np.sqrt(g @ psd(m._vcov) @ g))


def ed_gap(m, a, b):
    """exp(a'β) - exp(b'β)와 델타법 표준오차."""
    ea, _ = combo(m, a)
    eb, _ = combo(m, b)
    g = weights(m, a) * np.exp(ea) - weights(m, b) * np.exp(eb)
    return np.exp(ea), np.exp(eb), np.exp(ea) - np.exp(eb), float(np.sqrt(g @ psd(m._vcov) @ g))


def office(c):
    return {c: 1, f"{c}_W": 1}


def table_by_type(d):
    rows = []
    for t in TYPES:
        coef = fit_by_type(d[d["type"] == t]).coef()
        rows += [{"유형": t, "날짜": LABELS[c], "ED": np.exp(coef[c])} for c in LABELS]
    table = pd.DataFrame(rows).pivot(index="날짜", columns="유형", values="ED")
    table = table.reindex(list(LABELS.values()))[TYPES].round(3)

    m = fit_pooled(d[d["type"].isin([OFFICE, "그 밖"])])
    table["업무−그 밖 차이(SE)"] = ["{2:.3f} ({3:.3f})".format(*ed_gap(m, office(c), {c: 1})) for c in LABELS]
    return table.reset_index()


def festival_labels():
    """명절 연휴 블록 안 위치와 앞뒤 1~4번째 평일."""
    hol = holidays()
    days = pd.date_range("2016-12-01", "2026-12-31")
    block = off_blocks(days, hol)
    festival = sorted({block[d] for d, k in hol.items() if k == "N" and d in block.index})
    label = {}
    for b in festival:
        span = days[block == b]
        weekdays = [d for d in span if d.weekday() < 5]
        core = [d for d in weekdays if hol.get(d) == "N" and d not in SUBSTITUTE_DAYS]
        for d in weekdays:
            if d not in core:
                label[d] = "F_sub"
            elif len(core) == 1:
                label[d] = "F_mid"
            else:
                label[d] = "F_first" if d == core[0] else ("F_last" if d == core[-1] else "F_mid")
        for sign, edge in [(-1, span[0]), (1, span[-1])]:
            k, d = 0, edge
            while k < 4:
                d += pd.Timedelta(days=sign)
                if d.weekday() >= 5:
                    continue
                k += 1
                if d not in hol:
                    label.setdefault(d, f"F{'m' if sign < 0 else 'p'}{k}")
    return pd.Series(label)


FESTIVAL = {"Fm4": "연휴 전 4번째 평일", "Fm3": "연휴 전 3번째 평일", "Fm2": "연휴 전 2번째 평일",
            "Fm1": "연휴 직전 평일", "F_first": "연휴 첫날", "F_mid": "연휴 가운데", "F_last": "연휴 마지막날",
            "F_sub": "대체공휴일·연휴 내 다른 공휴일", "Fp1": "연휴 직후 평일", "Fp2": "연휴 후 2번째 평일",
            "Fp3": "연휴 후 3번째 평일", "Fp4": "연휴 후 4번째 평일"}


def festival_curve(d):
    d = d[d["type"].isin([OFFICE, "그 밖"])].copy()
    lab = festival_labels()
    tag = d["date"].map(lab).fillna("")
    for c in FESTIVAL:
        d[c] = (tag == c).astype(int)
    inside = tag != ""
    others = ["Hol_single", "Hol_first", "Hol_mid", "Hol_last", "Elec", "Temp", "Labor", "Sand"]
    for c in others:
        d[f"{c}_o"] = d[c] * ~inside
    m = fit_pooled(d, list(FESTIVAL) + [f"{c}_o" for c in others])

    rows = []
    for c, name in FESTIVAL.items():
        bw, sw = combo(m, office(c))
        bo, so = combo(m, {c: 1})
        _, _, gap, se = ed_gap(m, office(c), {c: 1})
        rows.append({"날짜": name, "일수": int(d.loc[d[c] == 1, "date"].nunique()),
                     "업무 ED": np.exp(bw), "업무 lo95": np.exp(bw - 1.96 * sw), "업무 hi95": np.exp(bw + 1.96 * sw),
                     "그 밖 ED": np.exp(bo), "그 밖 lo95": np.exp(bo - 1.96 * so), "그 밖 hi95": np.exp(bo + 1.96 * so),
                     "차이": gap, "차이 SE": se})
    return pd.DataFrame(rows).round(3)


def holiday_kinds(d):
    """식(4): 공휴일 종류별 업무·점심형 θ (기준 그 밖, 날짜 FE)."""
    d = d.copy()
    groups = [t for t in TYPES if t != "그 밖"]
    events = ["HolL", "Elec", "Temp", "Labor", "Nat", "NRm1_", "NRp1_", "Sand", "NRm24", "NRp24"]
    xs = []
    for c in present(d, events):
        for i, g in enumerate(groups):
            d[f"{c}__g{i}"] = d[c] * (d["type"] == g)
            xs.append(f"{c}__g{i}")
    m = pf.feols("ly ~ " + "+".join(xs) + " | dong + dfe", data=d, vcov=VCOV)

    names = {"HolL": "법정·대체공휴일", "Elec": "선거일", "Temp": "임시공휴일", "Labor": "근로자의날", "Nat": "평일 명절"}
    rows = []
    for c, name in names.items():
        b, s = m.coef()[f"{c}__g0"], m.se()[f"{c}__g0"]
        rows.append({"종류": name, "일수": int(d.loc[d[c] == 1, "date"].nunique()),
                     "θ(로그포인트)": round(100 * b, 1), "SE": round(100 * s, 1), "%": round(100 * (np.exp(b) - 1), 1)})
    return pd.DataFrame(rows)


def main(pop_dir, types_csv, out):
    out.mkdir(parents=True, exist_ok=True)
    d = load_panel(pop_dir, types_csv)
    for name, table in [("ed_by_type", table_by_type(d)),
                        ("festival_curve", festival_curve(d)),
                        ("holiday_kinds", holiday_kinds(d))]:
        table.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
        print(f"\n[{name}]\n{table.to_string(index=False)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
