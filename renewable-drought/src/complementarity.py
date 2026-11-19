"""태양광·풍력 상호보완성, 재생 가뭄, 공간 평활 (그림 1~8).

    python complementarity.py <work_dir> <wind_monthly_region_panel.csv>
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style
from drought import daily_mean, drought_events, max_drought

SEASON = {12: "겨울", 1: "겨울", 2: "겨울", 3: "봄", 4: "봄", 5: "봄",
          6: "여름", 7: "여름", 8: "여름", 9: "가을", 10: "가을", 11: "가을"}
MONTHS = np.arange(1, 13)

# 2040년 목표 설비 (태양광 155GW, 풍력 60.5GW)
SOLAR_2040, WIND_2040 = 155.0, 60.5

WIND_REGIONS = {"강원": (37.65, 128.70), "경북": (36.45, 129.25), "전남": (35.15, 126.40),
                "제주": (33.45, 126.55), "경남": (35.45, 128.05), "전북": (35.60, 126.55)}
SOLAR_REGIONS = {"서울_인천_경기": (37.40, 127.00), "원주_강원": (37.60, 128.20), "청주_충북": (36.80, 127.70),
                 "대전_충남": (36.50, 126.80), "전주_전북": (35.70, 127.00), "광주_전남": (34.90, 126.80),
                 "대구_경북": (36.20, 128.70), "부산_경남": (35.30, 128.40), "제주": (33.40, 126.50)}


def load(work):
    nat = pd.read_parquet(work / "panel_national.parquet")
    d = nat.index.get_level_values(0)
    nat["year"], nat["month"] = d.year, d.month
    nat["hr"] = nat.index.get_level_values(1)
    nat["season"] = nat["month"].map(SEASON)
    return nat


def blend(nat, wind_share):
    return (1 - wind_share) * nat["solar_cf"] + wind_share * nat["wind_cf"]


def mixes(nat):
    last = nat.loc["2025-12-31"].iloc[-1]
    return {
        "태양광만": 0.0,
        "2025년 설비 구성": float(last.wind_cap / (last.wind_cap + last.solar_cap)),
        "2040년 목표 구성": WIND_2040 / (SOLAR_2040 + WIND_2040),
        "풍력만": 1.0,
    }


def monthly_profile(nat, out, R):
    mon = nat.groupby(["year", "month"])[["solar_cf", "wind_cf"]].mean().reset_index()
    stats = mon.groupby("month")[["solar_cf", "wind_cf"]].agg(["mean", "min", "max"])
    R["monthly_cf"] = (stats.xs("mean", axis=1, level=1) * 100).round(1).to_dict()
    R["corr_monthly_level"] = round(mon.solar_cf.corr(mon.wind_cf), 3)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for col, color, label in [("solar_cf", style.SOLAR, "태양광"), ("wind_cf", style.WIND, "풍력")]:
        ax.fill_between(MONTHS, stats[(col, "min")] * 100, stats[(col, "max")] * 100, color=color, alpha=0.15, lw=0)
        ax.plot(MONTHS, stats[(col, "mean")] * 100, color=color, lw=2.2, marker="o", ms=4, label=label)
    ax.set_xticks(MONTHS, [f"{m}월" for m in MONTHS])
    ax.set_ylabel("월평균 이용률 (%)")
    ax.set_title("태양광·풍력 월별 이용률 (2015–2025, 띠는 연도별 최소–최대)", loc="left", fontsize=10)
    ax.legend(frameon=False, ncol=2, loc="upper center")
    style.save(fig, out / "01_monthly_cf.png")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)
    for ax, col, cmap, label in [(axes[0], "solar_cf", "YlOrBr", "태양광"), (axes[1], "wind_cf", "Blues", "풍력")]:
        grid = nat.groupby(["month", "hr"])[col].mean().unstack("month") * 100
        im = ax.imshow(grid.values, aspect="auto", origin="lower", cmap=cmap, extent=[0.5, 12.5, 0.5, 24.5])
        ax.set_xticks(MONTHS, [str(m) for m in MONTHS])
        ax.set_xlabel("월")
        ax.grid(False)
        ax.set_title(f"{label} 이용률 (%)", fontsize=9.5)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    axes[0].set_ylabel("시각 (1–24)")
    axes[0].set_yticks([1, 6, 12, 18, 24])
    fig.suptitle("월×시각 평균 이용률 (2015–2025)", x=0.01, ha="left", fontsize=10)
    style.save(fig, out / "02_month_hour_heatmap.png")


def weather_correlation(nat, out, R):
    """계절 평균을 빼고 남은 날씨 변동끼리의 상관. 음수면 서로 메워 준다."""
    day = nat.groupby(level=0)[["solar_cf", "wind_cf"]].mean()
    anom = day - day.groupby([day.index.year, day.index.month]).transform("mean")
    by_day = anom.groupby(day.index.month).apply(lambda g: g.solar_cf.corr(g.wind_cf))

    hourly = nat[["solar_cf", "wind_cf"]] - nat.groupby(["year", "month", "hr"])[["solar_cf", "wind_cf"]].transform("mean")
    daytime = nat["hr"].between(9, 17)
    by_hour = hourly[daytime].groupby(nat.loc[daytime, "month"]).apply(lambda g: g.solar_cf.corr(g.wind_cf))

    R["corr_daily_anom_all"] = round(anom.solar_cf.corr(anom.wind_cf), 3)
    R["corr_daily_anom_by_month"] = by_day.round(3).to_dict()
    R["corr_hourly_anom_daytime_by_month"] = by_hour.round(3).to_dict()

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    w = 0.38
    ax.bar(MONTHS - w / 2, by_day.values, w, color=style.MIX, label="일 단위 (월평균 제거)")
    ax.bar(MONTHS + w / 2, by_hour.values, w, color="#B794F4", label="9–17시 시간 단위 (월×시각 평균 제거)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(MONTHS, [f"{m}월" for m in MONTHS])
    ax.set_ylabel("상관계수")
    ax.set_title("태양광–풍력 날씨 변동의 상관 (음수일수록 서로 메움)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    style.save(fig, out / "03_anomaly_correlation.png")


def portfolio(nat, mix, out, R):
    rows = []
    for share in np.round(np.arange(0, 1.0001, 0.05), 2):
        p = daily_mean(blend(nat, share))
        z = p / p.mean()
        mo = p.groupby(p.index.to_period("M")).mean()
        rows.append({
            "wind_share": share,
            "cv_daily": z.std(),
            "cv_monthly": (mo / mo.mean()).std(),
            "max_drought_50": max_drought(p, 0.5),
            "max_drought_75": max_drought(p, 0.75),
            "days_below_50pct": (z.dropna() < 0.5).mean() * 100,
        })
    pf = pd.DataFrame(rows)
    pf.to_csv(out / "tab_portfolio_by_wind_share.csv", index=False, encoding="utf-8-sig")
    R["portfolio"] = pf.round(3).to_dict(orient="records")

    panels = [("cv_monthly", "월평균 출력 변동계수"),
              ("days_below_50pct", "일평균이 장기평균의 50% 미만인 날 (%)"),
              ("max_drought_50", "최장 가뭄 (일, 50% 임계값)")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for ax, (col, title) in zip(axes, panels):
        ax.plot(pf.wind_share * 100, pf[col], color=style.MIX, lw=2)
        ax.axvline(mix["2025년 설비 구성"] * 100, color=style.GREY, ls="--", lw=1)
        ax.axvline(mix["2040년 목표 구성"] * 100, color=style.MIX_ALT, ls="--", lw=1)
        i = pf[col].idxmin()
        ax.plot(pf.wind_share[i] * 100, pf[col][i], "o", color=style.MIX)
        ax.set_xlabel("풍력 설비 비중 (%)")
        ax.set_title(title, fontsize=9)
    top = axes[0].get_ylim()[1] * 0.95
    axes[0].text(mix["2025년 설비 구성"] * 100 + 1, top, "2025", color=style.GREY, fontsize=8)
    axes[0].text(mix["2040년 목표 구성"] * 100 + 1, top, "2040 목표", color=style.MIX_ALT, fontsize=8)
    fig.suptitle("풍력 설비 비중에 따른 태양광+풍력 합계 출력의 안정성 (2015–2025, 점은 최솟값)",
                 x=0.01, ha="left", fontsize=10)
    style.save(fig, out / "04_portfolio.png")


def droughts(nat, mix, out, R):
    taus = np.round(np.arange(0.1, 1.0001, 0.05), 2)
    table = pd.DataFrame({k: [max_drought(daily_mean(blend(nat, s)), t) for t in taus] for k, s in mix.items()},
                         index=pd.Index(taus, name="임계값(장기평균 대비)"))
    table.to_csv(out / "tab_max_drought_multithreshold.csv", encoding="utf-8-sig")
    R["max_drought"] = table.loc[[0.3, 0.5, 0.6, 0.75, 0.9]].to_dict()

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for (name, col), color in zip(table.items(), [style.SOLAR, style.GREY, style.MIX_ALT, style.WIND]):
        ax.plot(taus * 100, col, lw=2, color=color, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("임계값: 장기평균 대비 (%)")
    ax.set_ylabel("최장 가뭄 기간 (일, 로그)")
    ax.set_title("다중 임계값 최장 재생 가뭄 (2015–2025 일평균)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    style.save(fig, out / "05_drought_threshold.png")

    rows = []
    for name, share in mix.items():
        p = daily_mean(blend(nat, share))
        events = drought_events(p, 0.5, min_len=2)
        years = p.index.year.nunique()
        lengths = np.array([n for _, n in events]) if events else np.array([0])
        seasons = pd.Series([SEASON[d.month] for d, _ in events]).value_counts()
        rows.append({"구성": name,
                     "연평균_횟수(2일+)": round(len(events) / years, 1),
                     "연평균_횟수(5일+)": round((lengths >= 5).sum() / years, 1),
                     "최장_연속일": int(lengths.max()),
                     **{f"계절_{s}": int(seasons.get(s, 0)) for s in ["봄", "여름", "가을", "겨울"]}})
    events = pd.DataFrame(rows)
    events.to_csv(out / "tab_drought_events_tau50.csv", index=False, encoding="utf-8-sig")
    R["events_tau50"] = events.to_dict(orient="records")


def drought_2040(nat, mix, out, R, tau=0.5):
    p = daily_mean(blend(nat, mix["2040년 목표 구성"]))
    z = p / p.mean()
    dry = (z < tau)[z.notna()]

    peak = nat.groupby(level=0)["demand_MW"].max()
    top = peak >= peak.groupby(peak.index.year).transform(lambda s: s.quantile(0.9))
    top = top.reindex(dry.index, fill_value=False)
    winter = dry.index.month.isin([12, 1, 2])
    summer = dry.index.month.isin([6, 7, 8])
    R["dry40_share_all"] = round(dry.mean() * 100, 1)
    R["dry40_share_on_top10_demand"] = round(dry[top].mean() * 100, 1)
    R["dry40_share_on_top10_demand_winter"] = round(dry[top & winter].mean() * 100, 1)
    R["dry40_share_on_top10_demand_summer"] = round(dry[top & summer].mean() * 100, 1)
    R["dry40_by_month_pct"] = (dry.groupby(dry.index.month).mean() * 100).round(1).to_dict()

    day = nat.groupby(level=0)[["smp", "demand_MW"]].mean()
    ym = [day.index.year, day.index.month]
    smp_dev = day.smp - day.groupby(ym).smp.transform("mean")
    dem_dev = (day.demand_MW / day.groupby(ym).demand_MW.transform("mean") - 1) * 100
    dry_days, wet_days = dry.index[dry.to_numpy()], dry.index[~dry.to_numpy()]
    R["smp_dev_dry40"] = round(smp_dev.loc[dry_days].mean(), 2)
    R["smp_dev_wet40"] = round(smp_dev.loc[wet_days].mean(), 2)
    R["dem_dev_dry40_pct"] = round(dem_dev.loc[dry_days].mean(), 2)

    years = sorted(set(z.index.year))
    grid = np.full((len(years), 366), np.nan)
    for i, y in enumerate(years):
        s = z[z.index.year == y]
        grid[i, s.index.dayofyear - 1] = s.values
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    im = ax.imshow(grid, aspect="auto", cmap="RdBu", vmin=0, vmax=2, interpolation="nearest")
    ax.set_yticks(range(len(years)), years)
    ax.grid(False)
    month_start = pd.date_range("2021-01-01", periods=12, freq="MS").dayofyear - 1
    ax.set_xticks(month_start, [f"{m}월" for m in MONTHS])
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("장기평균 대비 (1=평균)")
    ax.set_title("2040년 목표 구성을 과거 날씨에 얹은 일별 출력 (빨강 = 가뭄)", loc="left", fontsize=10)
    style.save(fig, out / "06_calendar_2040.png")


def km(a, b):
    lat1, lon1, lat2, lon2 = map(np.radians, [a[0], a[1], b[0], b[1]])
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))


def anomalies(df):
    d = df.index.get_level_values(0)
    return df - df.groupby([d.year, d.month, df.index.get_level_values(1)]).transform("mean")


def spatial_smoothing(work, region_panel, out, R):
    wind = pd.read_parquet(work / "panel_wind_region.parquet")[list(WIND_REGIONS)].clip(upper=1)
    solar = pd.read_parquet(work / "panel_solar_region.parquet")
    hr = solar.index.get_level_values(1)
    solar_day = solar.loc[(hr >= 9) & (hr <= 17), list(SOLAR_REGIONS)]

    pairs = []
    for kind, df, coords in [("풍력", wind, WIND_REGIONS), ("태양광", solar_day, SOLAR_REGIONS)]:
        a = anomalies(df).dropna()
        hourly, daily = a.corr(), a.groupby(level=0).mean().corr()
        hourly.round(3).to_csv(out / f"tab_corr_{kind}.csv", encoding="utf-8-sig")
        names = list(coords)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pairs.append({"전원": kind, "a": names[i], "b": names[j],
                              "거리_km": km(coords[names[i]], coords[names[j]]),
                              "상관_시간": hourly.iloc[i, j], "상관_일": daily.iloc[i, j]})
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(out / "tab_pair_corr_distance.csv", index=False, encoding="utf-8-sig")

    # rho = exp(-d / l). 로그 적합은 rho > 0인 쌍만 쓸 수 있어서 모든 쌍을 쓰는 NLS도 같이 본다.
    length, length_nls, excluded = {}, {}, {}
    grid = np.linspace(20, 3000, 2981)
    for kind in ["풍력", "태양광"]:
        q = pairs[pairs.전원 == kind]
        pos = q[q.상관_시간 > 0]
        length[kind] = round(float(-(pos.거리_km ** 2).sum() / (pos.거리_km * np.log(pos.상관_시간)).sum()))
        sse = [((q.상관_시간 - np.exp(-q.거리_km / g)) ** 2).sum() for g in grid]
        length_nls[kind] = round(float(grid[int(np.argmin(sse))]))
        excluded[kind] = int((q.상관_시간 <= 0).sum())
    R["corr_length_km"] = length
    R["corr_length_km_nls_allpairs"] = length_nls
    R["corr_length_excluded_pairs"] = excluded

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    dd = np.linspace(0, 600, 100)
    for kind, color in [("풍력", style.WIND), ("태양광", style.SOLAR)]:
        q = pairs[pairs.전원 == kind]
        ax.scatter(q.거리_km, q.상관_시간, color=color, s=22, alpha=0.85, label=f"{kind} (지역 쌍)")
        ax.plot(dd, np.exp(-dd / length[kind]), color=color, lw=1.2, ls="--")
    ax.plot(dd, np.exp(-dd / 430), color="k", lw=1, ls=":", label="미국 동부 해상풍력 (Kempton 외, 약 430km)")
    ax.set_xlabel("지역 중심 간 거리 (km)")
    ax.set_ylabel("시간별 이상치 상관계수")
    ax.set_ylim(min(0, pairs.상관_시간.min() - 0.05), 1.02)
    ax.set_title("거리에 따른 지역 간 상관 (2020–2025, 태양광은 9–17시)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    style.save(fig, out / "07_corr_distance.png")

    wind = wind.dropna()
    cap = pd.read_csv(region_panel, usecols=["ym", "region", "발전설비용량_MW"])
    cap = cap[(cap.ym == "2025-12") & cap.region.isin(WIND_REGIONS)].set_index("region")["발전설비용량_MW"]
    weights = cap / cap.sum()
    total = (wind * weights).sum(axis=1)
    sig_region = anomalies(wind).std()
    sig_total = anomalies(total.to_frame("total"))["total"].std()
    R["wind_weights_2025"] = weights.round(3).to_dict()
    R["neff_wind"] = round(float(((sig_region * weights).sum() / sig_total) ** 2), 2)
    R["var_ratio_wind"] = round(float(sig_total ** 2 / (sig_region ** 2 * weights).sum()), 3)

    def summary(label, series):
        d = daily_mean(series)
        return {"단위": label,
                "최장가뭄_50": max_drought(d, 0.5, 366),
                "최장가뭄_75": max_drought(d, 0.75, 366),
                "일평균_변동계수": round(float((d / d.mean()).std()), 3)}

    rows = [summary(r, wind[r]) for r in WIND_REGIONS]
    rows.append(summary("6개 지역 합계", total))
    for r in ["서울_인천_경기", "광주_전남", "대구_경북", "제주"]:
        rows.append(summary("태양광 " + r, solar[r]))
    rows.append(summary("태양광 전국 (KPX+PPA)", solar["전국_KPX_PPA"]))
    table = pd.DataFrame(rows)
    table.to_csv(out / "tab_smoothing_region_vs_national.csv", index=False, encoding="utf-8-sig")
    R["smoothing"] = table.to_dict(orient="records")

    q = table.iloc[:7]
    colors = [style.WIND] * 6 + ["#1A365D"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for ax, col, title in [(axes[0], "일평균_변동계수", "일평균 출력 변동계수 (낮을수록 안정)"),
                           (axes[1], "최장가뭄_50", "최장 가뭄 (일, 50% 임계값)")]:
        ax.bar(q.단위, q[col], color=colors)
        for i, v in enumerate(q[col]):
            ax.text(i, v * 1.01, f"{v:.2f}" if col.startswith("일") else f"{int(v)}", ha="center", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("풍력: 지역 단위와 6개 지역 합계 비교 (2020–2025). 단기 변동은 줄지만 긴 가뭄은 거의 그대로",
                 x=0.01, ha="left", fontsize=10)
    style.save(fig, out / "08_spatial_smoothing.png")


def main(work, region_panel):
    style.setup()
    out = work / "out"
    out.mkdir(parents=True, exist_ok=True)
    nat = load(work)
    mix = mixes(nat)
    R = {"mix_wind_share": {k: round(v, 3) for k, v in mix.items()}}

    monthly_profile(nat, out, R)
    weather_correlation(nat, out, R)
    portfolio(nat, mix, out, R)
    droughts(nat, mix, out, R)
    drought_2040(nat, mix, out, R)
    spatial_smoothing(work, region_panel, out, R)

    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
