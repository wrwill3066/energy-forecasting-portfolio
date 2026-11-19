"""재생이 적은 날 누가 메우는가: 연료원별 시간 거래량으로 본 원전·석탄·LNG 운전 (그림 11).

연료원별 시간 거래량은 2021-01~2023-10, 2024년이 비어 있어 2017~2020과 2025를 비교한다.

    python fuel_mix.py <work_dir> <fuel_hourly.csv>
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style

FUELS = ["원자력", "석탄", "LNG", "양수", "기타"]
YEARS = [2017, 2018, 2019, 2020, 2025]
SEASONS = {3: "봄", 4: "봄", 5: "봄", 6: "여름", 7: "여름", 8: "여름", 9: "가을", 10: "가을", 11: "가을"}


def load(work, fuel_csv):
    f = pd.read_csv(fuel_csv, parse_dates=["date"]).set_index(["date", "he"])
    f = f[[c for c in f.columns if c.endswith("_MWh")]].dropna(how="all")
    f.columns = [c.removesuffix("_MWh") for c in f.columns]

    g = pd.DataFrame({
        "원자력": f["원자력"],
        "석탄": f[["유연탄", "무연탄"]].sum(axis=1, min_count=1),
        "LNG": f["LNG"],
        "양수": f["양수"],
    })
    g["기타"] = f.sum(axis=1) - g.sum(axis=1) - f["태양광"].fillna(0) - f["풍력"].fillna(0)
    g = g / 1000

    panel = pd.read_parquet(work / "panel_national.parquet")
    parts = pd.read_parquet(work / "solar_parts.parquet")
    g = g.join(panel[["demand_MW", "wind_gen"]]).join(parts)
    g["vre"] = (g.KPX + g.PPA + g.BTM + g.wind_gen) / 1000
    g["true_demand"] = (g.demand_MW + g.PPA + g.BTM) / 1000
    g = g.dropna(subset=["원자력", "vre"])
    dates = g.index.get_level_values(0)
    g["year"], g["month"] = dates.year, dates.month
    g["season"] = g["month"].map(SEASONS).fillna("겨울")
    return g


def operating_pattern(g):
    rows = {}
    for year in YEARS:
        x = g[g.year == year]
        day = x.groupby(level=0)
        spring = day.month.first().isin([3, 4, 5])
        r = {}
        for c in ["원자력", "석탄", "LNG"]:
            swing = day[c].max() - day[c].min()
            r[f"{c}_일중변동폭_GW"] = swing.mean()
            r[f"{c}_봄_일중변동폭_GW"] = swing[spring].mean()
            r[f"{c}_연최저/연최고_%"] = 100 * x[c].min() / x[c].max()
        r["원자력_p99대비90%미만_시간"] = int((x["원자력"] < 0.9 * x["원자력"].quantile(0.99)).sum())
        r["석탄_p99대비60%미만_시간"] = int((x["석탄"] < 0.6 * x["석탄"].quantile(0.99)).sum())
        r["양수_발전_연간_TWh"] = x["양수"].sum() / 1000
        r["재생비중_%"] = 100 * x.vre.sum() / x.true_demand.sum()
        rows[year] = r
    table = pd.DataFrame(rows).T.round(2)
    table.index.name = "연도"
    return table


def dark_vs_sunny(g):
    """계절마다 재생 하위 20% 날과 상위 20% 날의 연료원별 일평균 출력 차이."""
    rows = []
    for year in YEARS:
        for season in ["봄", "여름", "가을", "겨울"]:
            x = g[(g.year == year) & (g.season == season)]
            if x.empty:
                continue
            day = x.groupby(level=0)[FUELS + ["vre", "demand_MW"]].mean()
            day["demand_MW"] /= 1000
            dark = day[day.vre <= day.vre.quantile(0.2)].mean()
            sunny = day[day.vre >= day.vre.quantile(0.8)].mean()
            row = {"연도": year, "계절": season,
                   "재생_차이_GW": sunny.vre - dark.vre,
                   "시장수요_차이_GW": dark.demand_MW - sunny.demand_MW}
            row.update({f"{c}_증가_GW": dark[c] - sunny[c] for c in FUELS})
            rows.append(row)
    return pd.DataFrame(rows).round(2)


def ols_hac(y, X, lags):
    """OLS와 Newey-West(Bartlett) HAC 표준오차."""
    n, rank = len(y), np.linalg.matrix_rank(X)
    if rank < X.shape[1]:
        raise ValueError(f"설계행렬 rank 부족 ({rank} < {X.shape[1]})")
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = X * (y - X @ b)[:, None]
    S = u.T @ u
    for L in range(1, lags + 1):
        G = u[L:].T @ u[:-L]
        S += (1 - L / (lags + 1)) * (G + G.T)
    V = XtX_inv @ S @ XtX_inv * n / (n - rank)
    return b, np.sqrt(np.diag(V)), n


def fuel_response(g):
    """일별: 연료원 출력 = a·재생 + b·실제수요 + 연×월 FE + 요일 FE.

    재생이 1GW 줄 때 함께 움직이는 각 전원의 출력이다. 인과효과로 읽지 않는다.
    """
    d = g.groupby(level=0)[["원자력", "석탄", "LNG", "양수", "vre", "true_demand"]].mean()
    d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D"))
    d["ym"], d["dow"], d["year"] = d.index.strftime("%Y-%m"), d.index.dayofweek, d.index.year

    rows = []
    for period, sub in [("2017~2020", d[d.year <= 2020]), ("2025", d[d.year == 2025])]:
        for c in ["원자력", "석탄", "LNG", "양수"]:
            s = sub.dropna(subset=[c, "vre", "true_demand"])
            X = pd.concat([s[["vre", "true_demand"]],
                           pd.get_dummies(s["ym"], prefix="ym", dtype=float),
                           pd.get_dummies(s["dow"].astype(str), prefix="dow", drop_first=True, dtype=float)], axis=1)
            b, se7, n = ols_hac(s[c].to_numpy(), X.to_numpy(), 7)
            _, se14, _ = ols_hac(s[c].to_numpy(), X.to_numpy(), 14)
            rows.append({"기간": period, "전원": c, "재생1GW감소시_변화_GW": -b[0],
                         "se_HAC7": se7[0], "se_HAC14": se14[0], "일수": n})
    return pd.DataFrame(rows).round(3)


def plot_spring_dispatch(g, out, years=(2018, 2025)):
    fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    stack = ["원자력", "석탄", "LNG", "양수"]
    for i, year in enumerate(years):
        x = g[(g.year == year) & (g.season == "봄")]
        vre_day = x.groupby(level=0).vre.mean()
        groups = {"재생 많은 날(상위 20%)": vre_day[vre_day >= vre_day.quantile(0.8)].index,
                  "재생 적은 날(하위 20%)": vre_day[vre_day <= vre_day.quantile(0.2)].index}
        profiles = {}
        for j, (label, days) in enumerate(groups.items()):
            p = x[x.index.get_level_values(0).isin(days)].groupby(level=1)[stack + ["vre"]].mean()
            profiles[j] = p
            a = ax[i, j]
            a.stackplot(p.index, [p[c] for c in stack], colors=[style.FUEL_COLORS[c] for c in stack],
                        labels=stack, alpha=0.85)
            a.plot(p.index, p[stack].sum(axis=1) + p.vre, "k--", lw=1, label="+재생(전체)")
            a.set_title(f"{year}년 봄, {label}")
            a.set_ylim(0, 75)
        a = ax[i, 2]
        for c in ["원자력", "석탄", "LNG"]:
            a.plot(profiles[1].index, profiles[1][c] - profiles[0][c], color=style.FUEL_COLORS[c], lw=2, label=c)
        a.axhline(0, color="k", lw=0.8)
        a.set_title(f"{year}년 봄: 흐린 날 - 맑은 날 (GW)")
        a.legend(fontsize=9)
    ax[0, 0].legend(loc="lower left", fontsize=8)
    for a in ax[1]:
        a.set_xlabel("시각")
    ax[0, 0].set_ylabel("GW")
    ax[1, 0].set_ylabel("GW")
    fig.suptitle("봄철 맑은 날과 흐린 날, 원전·석탄·LNG의 운전 (2018 vs 2025)", fontsize=12)
    style.save(fig, out / "11_fuel_dispatch_spring.png")


def main(work, fuel_csv):
    style.setup()
    out = work / "out"
    out.mkdir(parents=True, exist_ok=True)
    g = load(work, fuel_csv)

    operating_pattern(g).to_csv(out / "tab_fuel_pattern_by_year.csv", encoding="utf-8-sig")
    dark_vs_sunny(g).to_csv(out / "tab_fuel_dark_vs_sunny.csv", index=False, encoding="utf-8-sig")
    response = fuel_response(g)
    response.to_csv(out / "tab_fuel_response_regression.csv", index=False, encoding="utf-8-sig")
    plot_spring_dispatch(g, out)
    print(response.to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
