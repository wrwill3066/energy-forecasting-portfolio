import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

SOLAR = "#E3A21A"
WIND = "#2B6CB0"
MIX = "#6B46C1"
MIX_ALT = "#C53030"
GREY = "#718096"

FUEL_COLORS = {"원자력": "#7d3c98", "석탄": "#5d4037", "LNG": "#2e86c1", "양수": "#17a589"}

_FONT_CANDIDATES = [
    os.environ.get("KFONT", ""),
    r"C:\Windows\Fonts\malgun.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def setup():
    family = "DejaVu Sans"
    for path in _FONT_CANDIDATES:
        if path and os.path.exists(path):
            font_manager.fontManager.addfont(path)
            family = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams.update({
        "font.family": family,
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 9,
    })


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
