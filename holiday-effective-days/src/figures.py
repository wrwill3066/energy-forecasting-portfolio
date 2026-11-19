"""그림: 공휴일이 낀 주의 요일별 ED, 행정동별 ED 지도.

    python figures.py weeks <living_pop_폴더> <상권영역.dbf> <직장인구.csv> <출력_폴더>
    python figures.py map   <living_pop_폴더> <types.csv> <행정동경계.geojson> <출력_폴더>

직장인구는 서울시 상권분석서비스(직장인구-상권, OA-15569), 상권영역은 OA-15560,
행정동 경계는 vuski/admdongkor (2023-07-01 기준)를 썼다.
"""
import json
import struct
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf
from matplotlib import font_manager
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from kr_calendar import calendar, holidays, is_normal_weekday

ORANGE, GREY, BLACK = "#C8742B", "#9AA3AE", "#1A1A1A"
WEEKDAYS = ["월", "화", "수", "목", "금"]


def set_font():
    for path in [r"C:\Windows\Fonts\malgun.ttf", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def read_dbf(path, encoding="utf-8"):
    with open(path, "rb") as f:
        header = f.read(32)
        n = struct.unpack("<I", header[4:8])[0]
        header_len, record_len = struct.unpack("<HH", header[8:12])
        fields = []
        while (desc := f.read(32))[0] != 0x0D:
            fields.append((desc[:11].split(b"\0")[0].decode("ascii"), desc[16]))
        f.seek(header_len)
        rows = []
        for _ in range(n):
            rec, pos, row = f.read(record_len), 1, []
            for _, length in fields:
                row.append(rec[pos:pos + length].decode(encoding, "ignore").strip())
                pos += length
            rows.append(row)
    return pd.DataFrame(rows, columns=[name for name, _ in fields])


def worker_exposure(area_dbf, workers_csv):
    """행정동별 직장인구 (2022~2023 분기 평균)."""
    area = read_dbf(area_dbf)
    to_dong = area.set_index(area["TRDAR_CD"].astype(int))["ADSTRD_CD"].astype(int)
    w = pd.read_csv(workers_csv, encoding="cp949", usecols=[0, 3, 5])
    w.columns = ["q", "area", "workers"]
    w = w[w["q"].between(20221, 20234)]
    w["dong"] = w["area"].map(to_dong)
    return w.groupby(["dong", "q"])["workers"].sum().groupby("dong").mean()


def load_lunch(pop_dir, pattern="*.parquet"):
    pop = pd.concat(pd.read_parquet(f) for f in sorted(pop_dir.glob(pattern)))
    pop = pop.groupby(["dong", "date"], as_index=False)["lunch"].mean()
    pop["dong"] = pop["dong"].astype(int)
    pop["dt"] = pd.to_datetime(pop["date"].astype(int).astype(str))
    return pop


def week_examples(pop_dir, workers, out):
    """ED = 그날 점심 생활인구 합 / 같은 요일 전후 4주 중 정상 평일 평균."""
    hol, cal = holidays(), calendar()
    top = set(workers.index[workers.rank(pct=True) >= 0.8])
    pop = load_lunch(pop_dir, "*202[3-5]*.parquet")
    pop = pop[pop["dong"].isin(workers.index)]
    days_seen = pop.groupby("dong")["dt"].nunique()
    pop = pop[pop["dong"].isin(days_seen[days_seen == days_seen.max()].index)]
    pop["g"] = np.where(pop["dong"].isin(top), "top", "rest")
    total = pop.groupby(["g", "dt"])["lunch"].sum().unstack(0)
    normal = is_normal_weekday(cal, hol).reindex(total.index, fill_value=False)

    def ed(day):
        ref = [day + pd.Timedelta(weeks=k) for k in (-4, -3, -2, -1, 1, 2, 3, 4)]
        ref = [r for r in ref if r in total.index and normal[r]]
        return total.loc[day] / total.loc[ref].mean()

    weeks = [("2024-05-13", "수요일 공휴일(부처님오신날)"), ("2025-06-02", "화요일 공휴일(대통령 선거일)"),
             ("2024-09-30", "화·목 공휴일, 수요일 징검다리"), ("2024-09-16", "월~수 추석 연휴")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    x = np.arange(5)
    for ax, (monday, title) in zip(axes, weeks):
        days = pd.date_range(monday, periods=5)
        vals = [ed(d) for d in days]
        t, r = [v["top"] for v in vals], [v["rest"] for v in vals]
        for i, d in enumerate(days):
            if d in hol:
                ax.axvspan(i - 0.45, i + 0.45, color=GREY, alpha=0.15, lw=0)
            ax.text(i - 0.19, t[i] + 0.015, f"{t[i]:.2f}", ha="center", fontsize=7.5, color=ORANGE)
            ax.text(i + 0.19, r[i] + 0.015, f"{r[i]:.2f}", ha="center", fontsize=7.5, color=BLACK)
        ax.bar(x - 0.19, t, 0.38, color=ORANGE)
        ax.bar(x + 0.19, r, 0.38, color=GREY)
        ax.axhline(1, color=BLACK, ls=":", lw=0.9)
        ax.set_xticks(x, [f"{WEEKDAYS[i]}\n{d.month}/{d.day}" for i, d in enumerate(days)], fontsize=8.5)
        ax.set_title(f"{days[0].year}년 {title}", fontsize=9, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("유효일수(같은 요일 정상 평일=1)")
    axes[0].set_ylim(0, 1.25)
    tr = axes[0].get_yaxis_transform()
    axes[0].text(0.02, 1.2, "직장인구 상위 20% 동", color=ORANGE, fontsize=8.5, transform=tr)
    axes[0].text(0.02, 1.12, "그 밖 동", color=GREY, fontsize=8.5, transform=tr)
    fig.text(0.01, 0.005, "주: 점심(11~13시) 내국인 생활인구 동 합계. 회색 음영은 공휴일.", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "week_examples.png", dpi=300)
    plt.close(fig)


def dong_ed(pop_dir, types):
    """정상 평일로 동×연월, 동×요일×연도 FE를 추정하고 실제값 / 예측값을 ED로 본다."""
    hol, cal = holidays(), calendar()
    pop = load_lunch(pop_dir)
    pop = pop[pop["dong"].isin(types.index) & (pop["lunch"] > 0) & (pop["dt"].dt.dayofweek < 5)].copy()
    pop["dm"] = pop["dong"].astype(str) + pop["dt"].dt.strftime("%Y%m")
    pop["ddy"] = pop["dong"].astype(str) + pop["dt"].dt.dayofweek.astype(str) + pop["dt"].dt.year.astype(str)
    pop["ly"] = np.log(pop["lunch"])
    pop["normal"] = pop["dt"].map(is_normal_weekday(cal, hol))

    m = pf.feols("ly ~ 1 | dm + ddy", data=pop[pop["normal"]].copy())
    pop["ed"] = np.exp(pop["ly"] - m.predict(newdata=pop))

    out = pd.DataFrame(index=sorted(pop["dong"].unique()))
    for key, day in [("ED1", "2024-10-09"), ("ED2", "2025-06-03"), ("ED3", "2025-01-29")]:
        out[key] = pop[pop["dt"] == pd.Timestamp(day)].set_index("dong")["ed"]
    single = set(cal.index[cal["Hol_single"] == 1])
    out["ED4"] = pop[pop["dt"].isin(single)].groupby("dong")["ed"].mean()
    out["type"] = out.index.map(types)
    return out


def seoul_polygons(geojson):
    with open(geojson, encoding="utf-8") as f:
        features = json.load(f)["features"]
    polys = []
    for feat in features:
        code = str(feat["properties"].get("adm_cd2", ""))
        if not code.startswith("11"):
            continue
        geom = feat["geometry"]
        rings = geom["coordinates"][:1] if geom["type"] == "Polygon" else [p[0] for p in geom["coordinates"]]
        polys += [(int(code[:8]), np.array(r)) for r in rings]
    return polys


def ed_map(pop_dir, types, geojson, out):
    ed = dong_ed(pop_dir, types)
    ed.round(3).to_csv(out / "dong_ed.csv", encoding="utf-8-sig")
    polys = seoul_polygons(geojson)
    cmap = LinearSegmentedColormap.from_list("ed", ["#7A3E12", "#C8742B", "#F3E3D3", "#F2F2F2", "#C9D3DE", "#5C7593"])
    norm = TwoSlopeNorm(vmin=0.4, vcenter=1.0, vmax=1.3)
    panels = [("ED1", "2024년 10월 9일(수) 한글날"), ("ED2", "2025년 6월 3일(화) 대통령 선거일"),
              ("ED3", "2025년 1월 29일(수) 설날"), ("ED4", "비명절 단독 공휴일 평균(2017~2026)")]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9.6))
    for ax, (key, title) in zip(axes.ravel(), panels):
        colors = [cmap(norm(v)) if pd.notna(v) else "#FFFFFF" for v in (ed[key].get(c, np.nan) for c, _ in polys)]
        ax.add_collection(PolyCollection([r for _, r in polys], facecolors=colors, edgecolors="#FFFFFF", linewidths=0.25))
        ax.autoscale()
        ax.set_aspect(1 / np.cos(np.radians(37.55)))
        ax.axis("off")
        v = ed[key]
        ax.set_title(f"{title}\n중앙값 {v.median():.2f}, 업무·점심형 평균 {v[ed['type'] == '업무·점심형'].mean():.2f}, "
                     f"그 밖 평균 {v[ed['type'] == '그 밖'].mean():.2f}", fontsize=9.5, loc="left")
    cax = fig.add_axes([0.25, 0.06, 0.5, 0.015])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_ticks([0.4, 0.6, 0.8, 1.0, 1.15, 1.3])
    cax.set_title("유효일수(정상 평일 예측값=1)", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "dong_ed_map.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    set_font()
    mode, args = sys.argv[1], [Path(a) for a in sys.argv[2:]]
    args[-1].mkdir(parents=True, exist_ok=True)
    if mode == "weeks":
        week_examples(args[0], worker_exposure(args[1], args[2]), args[3])
    else:
        types = pd.read_csv(args[1], index_col=0)["type"]
        types.index = types.index.astype(int)
        ed_map(args[0], types, args[2], args[3])
