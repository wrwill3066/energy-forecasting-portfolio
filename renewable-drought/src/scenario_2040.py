"""12차 전력수급기본계획(안)의 2040년 설비에 2015~2025년 실측 기상 11년을 입힌 시나리오 (그림 12~13).

조정 전원 필요량 R = 수요 - 태양광 - 풍력 - 원전(일정 출력). R < 0이면 잉여(출력제어나 저장 충전).

    python scenario_2040.py <work_dir>
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style

# 12차 전기본 토론회 자료·보도 기준 (2026.8~9)
PEAK_2040 = 158.4   # GW, 기준 수요 재전망
FLAT_ADD = 26.6     # GW, 반도체·데이터센터 등 24시간 평탄 부하 증가분
NUCLEAR = 30.0      # GW, 원전 평균 가용 출력 (설비 약 35GW × 이용률 85%)
OFFSHORE_CF = 0.35  # 해상풍력 평균 이용률 가정 (시간 패턴은 육상풍력 사용)

# (태양광, 육상풍력, 해상풍력) GW. 태양광이 None이면 S1과 연간 재생 발전량이 같도록 맞춘다.
SCENARIOS = {
    "S1 12차 계획": (155.3 + 16.3, 15.5, 45.0),
    "S2 풍력 지연": (155.3 + 16.3, 10.0, 10.0),
    "S3 풍력 지연분을 태양광으로": (None, 10.0, 10.0),
    "S4 태양광 축소·풍력 확대": (None, 25.0, 70.0),
}
COLORS = ["#2e86c1", "#c0392b", "#e8a33d", "#27ae60"]


def load(work):
    p = pd.read_parquet(work / "panel_national.parquet").join(pd.read_parquet(work / "solar_parts.parquet"))
    true_demand = p.demand_MW + p.PPA + p.BTM
    year = p.index.get_level_values(0).year
    shape = true_demand / true_demand.groupby(year).transform("mean")
    scale = (PEAK_2040 - FLAT_ADD) / shape.groupby(year).max().mean()
    demand = shape * scale + FLAT_ADD

    solar_cf = p.solar_cf.clip(0, 1).fillna(0)
    wind_cf = p.wind_cf.clip(0, 1).fillna(0)
    offshore_cf = (wind_cf * OFFSHORE_CF / wind_cf.mean()).clip(upper=1)
    return demand, solar_cf, wind_cf, offshore_cf


def evaluate(R, demand, vre, solar, wind):
    daily = R.groupby(level=0).mean()
    year = R.index.get_level_values(0).year
    top = R[R >= R.quantile(0.99)]
    top_month = top.index.get_level_values(0).month
    yearly_max = R.groupby(year).max()
    return {
        "태양광_GW": solar,
        "풍력_GW": wind,
        "재생비중_%": 100 * vre.sum() / demand.sum(),
        "조정전원_필요_최대_GW(시간)": R.max(),
        "조정전원_필요_최대_GW(일평균)": daily.max(),
        "최악7일_평균_GW": daily.rolling(7).mean().max(),
        "최악14일_평균_GW": daily.rolling(14).mean().max(),
        "최악14일_에너지_TWh": daily.rolling(14).mean().max() * 24 * 14 / 1000,
        "연평균_필요_GW": R.clip(lower=0).mean(),
        "잉여시간_연평균": (R < 0).groupby(year).sum().mean(),
        "잉여_TWh_연평균": (-R.clip(upper=0)).groupby(year).sum().mean() / 1000,
        "3시간상승_p99_GW": R.diff(3).quantile(0.99),
        "상위1%시간_겨울(12~2월)_%": 100 * top_month.isin([12, 1, 2]).mean(),
        "상위1%시간_여름(7~9월)_%": 100 * top_month.isin([7, 8, 9]).mean(),
        "상위1%시간_17~21시_%": 100 * pd.Index(top.index.get_level_values(1)).isin(range(17, 22)).mean(),
        "기상연도간_최대필요_범위_GW": f"{yearly_max.min():.1f}~{yearly_max.max():.1f}",
    }


def run(work):
    demand, solar_cf, wind_cf, offshore_cf = load(work)

    def generation(solar, onshore, offshore):
        return solar * solar_cf + onshore * wind_cf + offshore * offshore_cf

    base_energy = generation(*SCENARIOS["S1 12차 계획"]).sum()
    results, series = {}, {}
    for name, (solar, onshore, offshore) in SCENARIOS.items():
        if solar is None:
            wind_energy = (onshore * wind_cf + offshore * offshore_cf).sum()
            solar = (base_energy - wind_energy) / solar_cf.sum()
        vre = generation(solar, onshore, offshore)
        R = demand - vre - NUCLEAR
        results[name] = evaluate(R, demand, vre, solar, onshore + offshore)
        series[name] = R
    return pd.DataFrame(results).T, series, demand


def plot_duration(series, out):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for (name, R), color in zip(series.items(), COLORS):
        daily = R.groupby(level=0).mean()
        ax[0].plot(np.linspace(0, 100, len(daily)), np.sort(daily.values)[::-1], color=color, label=name)
        by_month = daily.groupby(daily.index.month)
        ax[1].plot(range(1, 13), by_month.mean().values, "-o", color=color, label=name)
        ax[1].plot(range(1, 13), by_month.quantile(0.98).values, ":", color=color)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_title("일평균 조정 전원 필요량 지속곡선 (11개 기상연도, GW)")
    ax[0].set_xlabel("% of days")
    ax[0].legend(fontsize=9)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_title("월별 일평균 필요량 (실선 평균, 점선 상위 2%)")
    ax[1].set_xticks(range(1, 13))
    style.save(fig, out / "12_scenario_2040.png")


def plot_worst_window(R, demand, out, days=14):
    daily = R.groupby(level=0).mean()
    end = daily.rolling(days).mean().idxmax()
    start = end - pd.Timedelta(days=days - 1)
    r = R.loc[start:end]
    d = demand.loc[start:end].to_numpy()
    vre = (demand - NUCLEAR - R).loc[start:end].to_numpy()
    used = np.minimum(NUCLEAR + vre, d)
    x = np.arange(len(r))

    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.plot(x, d, "k", lw=1, label="수요")
    ax.fill_between(x, 0, NUCLEAR, color="#7d3c98", alpha=0.6, label="원전")
    ax.fill_between(x, NUCLEAR, used, color="#e8a33d", alpha=0.7, label="태양광+풍력(쓰인 만큼)")
    ax.fill_between(x, used, NUCLEAR + vre, color="#e8a33d", alpha=0.25, hatch="//", label="잉여 재생")
    ax.fill_between(x, used, d, where=d > used, color="#5d4037", alpha=0.5, label="조정 전원 필요(LNG·석탄·양수·ESS 등)")
    ticks = r.index.get_level_values(0)[::24]
    ax.set_xticks(x[::24], [t.strftime("%m-%d") for t in ticks])
    ax.set_title(f"S1 12차 계획: 최악의 {days}일 ({start:%Y-%m-%d}~{end:%m-%d} 기상), GW")
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5)
    ax.set_ylim(0, 175)
    style.save(fig, out / "13_scenario_worst14.png")
    return f"{start:%Y-%m-%d}~{end:%Y-%m-%d}"


def main(work):
    style.setup()
    out = work / "out"
    out.mkdir(parents=True, exist_ok=True)
    table, series, demand = run(work)
    table.to_csv(out / "tab_scenario_2040.csv", encoding="utf-8-sig")
    plot_duration(series, out)
    window = plot_worst_window(series["S1 12차 계획"], demand, out)

    assumptions = {"PEAK_2040": PEAK_2040, "FLAT_ADD": FLAT_ADD, "NUCLEAR": NUCLEAR,
                   "OFFSHORE_CF": OFFSHORE_CF, "SCENARIOS": SCENARIOS, "worst14_S1": window}
    with open(out / "results_scenario_2040.json", "w", encoding="utf-8") as f:
        json.dump({"assumptions": assumptions, "results": table.reset_index().to_dict("records")},
                  f, ensure_ascii=False, indent=1, default=str)
    print(table.to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
