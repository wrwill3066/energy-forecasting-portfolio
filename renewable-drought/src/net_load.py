"""조정 전원(화력·원전 등)이 실제로 받는 순부하의 연도별 변화 (그림 9~10).

순부하 = 시장 수요 - KPX 태양광 - 풍력.
시장 수요에는 PPA·BTM 태양광이 이미 빠져 있으므로, 실제 수요는 시장 수요 + PPA + BTM으로 근사한다.

    python net_load.py <work_dir>
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style


def load(work):
    p = pd.read_parquet(work / "panel_national.parquet")
    s = pd.read_parquet(work / "solar_parts.parquet")
    d = p.join(s, how="left").dropna(subset=["demand_MW"])
    d["vre"] = d.KPX + d.PPA + d.BTM + d.wind_gen
    d["true_demand"] = d.demand_MW + d.PPA + d.BTM
    d["net_load"] = d.demand_MW - d.KPX - d.wind_gen
    dates = d.index.get_level_values(0)
    d["year"], d["month"] = dates.year, dates.month
    return d


def yearly_indicators(d):
    rows = {}
    for year, g in d.groupby("year"):
        day = g.groupby(level=0)
        nl = g.net_load / 1000
        vre_day = day.vre.mean() / 1000
        worst_week = vre_day.mean() - vre_day.rolling(7).mean().min()
        winter = g[g.month.isin([12, 1, 2])]
        peak_hour = winter.groupby(level=0).net_load.idxmax().map(lambda t: t[1])
        rows[year] = {
            "재생비중_실제수요대비_%": 100 * g.vre.sum() / g.true_demand.sum(),
            "순부하_최저_GW": nl.min(),
            "순부하_최고_GW": nl.max(),
            "최저/최고_%": 100 * nl.min() / nl.max(),
            "봄_순부하_최저_GW": g[g.month.isin([3, 4, 5])].net_load.min() / 1000,
            "순부하40GW미만_시간": int((nl < 40).sum()),
            "순부하45GW미만_시간": int((nl < 45).sum()),
            "일중변동폭_평균_GW": ((day.net_load.max() - day.net_load.min()) / 1000).mean(),
            "실제수요_일중변동폭_GW": ((day.true_demand.max() - day.true_demand.min()) / 1000).mean(),
            "3시간상승_p99_GW": (g.net_load.diff(3) / 1000).quantile(0.99),
            "재생일평균_표준편차_GW": vre_day.std(),
            "최악7일_재생부족_GW": worst_week,
            "최악7일_부족_수요대비_%": 100 * worst_week / (g.true_demand.mean() / 1000),
            "겨울_순부하피크_17시이후_%": 100 * (peak_hour >= 17).mean(),
        }
    table = pd.DataFrame(rows).T.round(2)
    table.index.name = "연도"
    return table


def spring_profiles(d, year):
    """봄(3~5월) 재생 하위 20% 날과 상위 20% 날의 시간대별 평균 순부하 (GW)."""
    g = d[(d.year == year) & d.month.isin([3, 4, 5])]
    vre_day = g.groupby(level=0).vre.mean()
    dates = g.index.get_level_values(0)
    dark = g[dates.isin(vre_day[vre_day <= vre_day.quantile(0.2)].index)]
    sunny = g[dates.isin(vre_day[vre_day >= vre_day.quantile(0.8)].index)]
    return dark.groupby(level=1).net_load.mean() / 1000, sunny.groupby(level=1).net_load.mean() / 1000


def plot_spring(d, out, years=(2016, 2020, 2025)):
    gap = {}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, year in zip(axes, years):
        dark, sunny = spring_profiles(d, year)
        ax.plot(dark.index, dark.values, color="#555", lw=2, label="재생 적은 날(하위 20%)")
        ax.plot(sunny.index, sunny.values, color="#e8a33d", lw=2, label="재생 많은 날(상위 20%)")
        ax.fill_between(dark.index, sunny.values, dark.values, color="#e8a33d", alpha=0.15)
        ax.set_title(f"{year}년 봄(3~5월)")
        ax.set_xlabel("시각")
        gap[year] = {"정오_차이_GW": float(dark.loc[13] - sunny.loc[13]),
                     "하루평균_차이_GW": float((dark - sunny).mean()),
                     "맑은날_정오_GW": float(sunny.loc[13]),
                     "맑은날_저녁피크_GW": float(sunny.loc[17:21].max())}
    axes[0].set_ylabel("조정 전원이 받는 순부하 (GW)")
    axes[0].legend(fontsize=9)
    fig.suptitle("같은 봄이라도 날씨에 따라 화력·원전이 맡는 부하가 크게 갈린다", fontsize=12)
    style.save(fig, out / "09_spring_net_load.png")
    return gap


def plot_yearly(t, out):
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
    ax[0, 0].plot(t.index, t["최악7일_재생부족_GW"], "o-", color="#c0392b")
    ax[0, 0].set_title("최악의 7일 동안 빠진 재생 출력 (연평균 대비, GW)")
    ax[0, 1].plot(t.index, t["순부하_최저_GW"], "o-", label="순부하 최저")
    ax[0, 1].plot(t.index, t["순부하_최고_GW"], "s-", label="순부하 최고")
    ax[0, 1].set_title("조정 전원이 받는 부하의 범위 (GW)")
    ax[0, 1].legend()
    ax[1, 0].plot(t.index, t["일중변동폭_평균_GW"], "o-", label="순부하")
    ax[1, 0].plot(t.index, t["실제수요_일중변동폭_GW"], "s--", label="실제 수요")
    ax[1, 0].set_title("하루 안의 최대-최저 폭, 평균 (GW)")
    ax[1, 0].legend()
    ax[1, 1].bar(t.index, t["순부하45GW미만_시간"], color="#2e86c1")
    ax[1, 1].set_title("순부하 45GW 미만 시간 수 (연간)")
    style.save(fig, out / "10_net_load_by_year.png")


def residual_by_wind_share(work, out):
    """재생이 연간 수요의 s만큼을 공급한다고 둘 때, 풍력 비중별 순부하의 꼬리."""
    nat = pd.read_parquet(work / "panel_national.parquet")
    day = nat.groupby(level=0)[["solar_cf", "wind_cf", "demand_MW"]].mean()
    demand = day.demand_MW / day.demand_MW.groupby(day.index.year).transform("mean")

    def supply(share):
        g = (1 - share) * day.solar_cf + share * day.wind_cf
        return g / g.mean()

    rows = []
    for s in [0.3, 0.5]:
        for share in np.round(np.arange(0, 1.0001, 0.05), 2):
            r = demand - s * supply(share)
            rows.append({"s": s, "wind_share": share, "std_daily": r.std(), "p99_daily": r.quantile(0.99),
                         "max7": r.rolling(7).mean().max(), "max14": r.rolling(14).mean().max(),
                         "max30": r.rolling(30).mean().max()})
    table = pd.DataFrame(rows)
    table.to_csv(out / "tab_residual_by_wind_share.csv", index=False, encoding="utf-8-sig")
    best = table.loc[table.groupby("s").max14.idxmin(), ["s", "wind_share", "max14"]]
    return best.to_dict(orient="records")


def main(work):
    style.setup()
    out = work / "out"
    out.mkdir(parents=True, exist_ok=True)
    d = load(work)

    table = yearly_indicators(d)
    table.to_csv(out / "tab_evidence_by_year.csv", encoding="utf-8-sig")
    plot_yearly(table, out)
    gap = plot_spring(d, out)
    best = residual_by_wind_share(work, out)

    with open(out / "results_net_load.json", "w", encoding="utf-8") as f:
        json.dump({"by_year": table.reset_index().to_dict("records"), "spring_dark_vs_sunny": gap,
                   "best_wind_share_max14": best}, f, ensure_ascii=False, indent=1, default=float)
    print(table.T.to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
