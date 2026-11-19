"""LP 결과 그림. 음영은 예측오차 SE 기준 90% 구간.

    python figures.py <lp_all.csv> <출력_폴더>
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

POOLED = "통합(태백2018·태백2024·삼척2025)"
EVENT_STYLE = {
    "태백2018": dict(color="#4d4d4d", ls="--", lw=0.8, marker="o", ms=2.2, mfc="white", markevery=4),
    "태백2024": dict(color="#4d4d4d", ls="-.", lw=0.8, marker="^", ms=2.4, mfc="white", markevery=4),
    "삼척2025": dict(color="#4d4d4d", ls=":", lw=0.9, marker="s", ms=2.2, mfc="white", markevery=4),
}
CM = 1 / 2.54
WIDTH = 15.5 * CM


def setup():
    for path in [r"C:\Windows\Fonts\malgun.ttf", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 9, "legend.fontsize": 8, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "axes.linewidth": 0.6, "savefig.dpi": 300})


def with_origin(s, col):
    """h = -1 (기준점, 0)을 끼워 넣는다."""
    h = np.r_[s["h"].to_numpy(), -1]
    v = np.r_[s[col].to_numpy(), 0]
    order = np.argsort(h)
    return h[order], v[order]


def panel(ax, res, outcome, ylabel, xlabel):
    q = res[res["결과변수"] == outcome]
    pooled = q[(q["묶음"] == POOLED) & (q["사건수"] == 3)].sort_values("h")
    h, b = with_origin(pooled, "계수")
    _, lo = with_origin(pooled, "하한90_예측오차")
    _, hi = with_origin(pooled, "상한90_예측오차")
    ax.fill_between(h, lo, hi, color="#d0d0d0", lw=0, zorder=0)
    ax.plot(h, b, color="black", lw=1.4, label="통합(3개 사건)")
    for event, style in EVENT_STYLE.items():
        hh, bb = with_origin(q[q["묶음"] == event].sort_values("h"), "계수")
        ax.plot(hh, bb, label=event, **style)
    ax.axhline(0, color="black", lw=0.4)
    ax.axvline(-0.5, color="#9e9e9e", lw=0.5, ls=(0, (1, 2)))
    ax.set_xlim(-24, 24)
    ax.set_xticks([-24, -12, 0, 12, 24])
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel)


def legend(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles + [plt.Rectangle((0, 0), 1, 1, color="#d0d0d0")], labels + ["통합 90% 구간"],
               ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.0), handlelength=2.4)


def main(lp_csv, out):
    setup()
    out.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(lp_csv)
    res = res[res["사양"] == "주사양"]

    outcomes = [("로그 총인구", "총인구"), ("로그 15~29세", "15~29세"), ("로그 30~44세", "30~44세"),
                ("로그 45~64세", "45~64세"), ("로그 65세 이상", "65세 이상"), ("로그 0~14세", "0~14세")]
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 10.5 * CM))
    for i, (ax, (outcome, name)) in enumerate(zip(axes.flat, outcomes)):
        panel(ax, res, outcome, "변화(%)" if i % 3 == 0 else "", f"({'가나다라마바'[i]}) {name}")
    legend(fig, axes.flat[0])
    fig.tight_layout()
    fig.savefig(out / "lp_population_age.png", bbox_inches="tight")
    plt.close(fig)

    outcomes = [("전입률", "전입"), ("전출률", "전출"), ("순이동률", "순이동")]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 5.8 * CM))
    for i, (ax, (outcome, name)) in enumerate(zip(axes, outcomes)):
        panel(ax, res, outcome, "인구 1천 명당 누적(명)" if i == 0 else "", f"({'가나다'[i]}) {name}")
    legend(fig, axes[0])
    fig.tight_layout()
    fig.savefig(out / "lp_migration.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
