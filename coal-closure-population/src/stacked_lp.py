"""폐광 전후 인구 변화: 사건별 묶음(stacked) 패널 국소투영.

    y(i, t+h) - y(i, t-1) = β_h · 폐광(i, t) + 시군 FE + 월 FE + ε,   h = -24..24

- 묶음마다 처치 시군 1곳과 비교 시군을 둔다. 처치 시점이 다른 시군끼리는 서로 비교군이 되지 않는다.
- 표본 기점은 사건 기점 t0와, 구간 [t-1, t+h]가 모두 t0 이전인 폐광 전 기점(최근 L개월)이다.
  시군 FE가 처치 시군의 폐광 전 평균 h개월 변화(고유 추세)를, 월 FE가 비교군 공통 변화를 걷어 낸다.
- 처치 시군이 한 곳이라 군집 SE는 사건 달의 고유 변동을 담지 못한다. 그래서 처치 시군의 폐광 전
  잔차로 잰 예측오차 SE를 주 구간으로 쓰고, 겹치는 구간의 자기상관은 Newey-West 장기분산으로 반영한다.

    python stacked_lp.py <panel_month.csv> <closures.csv> <출력_폴더>

closures.csv: 지역명, 값(1989년 이후 연도별 석탄광 폐광 수) — 공공데이터포털 석탄광 폐광 현황
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EVENTS = [
    {"id": "태백2018", "k": "강원 태백", "t0": "2018-08"},
    {"id": "태백2024", "k": "강원 태백", "t0": "2024-07"},
    {"id": "삼척2025", "k": "강원 삼척", "t0": "2025-07"},
    {"id": "화순2023", "k": "전남 화순", "t0": "2023-07"},
]
GROUPS = {
    "통합(태백2018·태백2024·삼척2025)": ["태백2018", "태백2024", "삼척2025"],
    "태백2018": ["태백2018"], "태백2024": ["태백2024"], "삼척2025": ["삼척2025"],
    "화순2023(보조)": ["화순2023"],
}
AGES = {"p0014": "0~14세", "p1529": "15~29세", "p3044": "30~44세", "p4564": "45~64세", "p65": "65세 이상"}
# KOSIS 총전입·총전출에는 같은 시군 안 이동이 섞여 있어, 시군 경계를 넘는 이동만 쓴다
FLOWS = {"전입률": ("in_inter", "in_intra"), "전출률": ("out_inter", "out_intra"), "순이동률": None,
         "시도간 전입률": ("in_inter",), "시도내 전입률": ("in_intra",),
         "시도간 전출률": ("out_inter",), "시도내 전출률": ("out_intra",)}
PROVINCE = {"강원특별자치도": "강원", "충청북도": "충북", "충청남도": "충남", "전북특별자치도": "전북",
            "전라남도": "전남", "경상북도": "경북", "경상남도": "경남"}
MDE_FACTOR = 2.486  # 양측 10%, 검출력 80%


class Panel:
    def __init__(self, panel_csv):
        p = pd.read_csv(panel_csv)
        self.units = sorted(p["k"].unique())
        self.months = sorted(p["ym"].unique())
        self.index = {m: i for i, m in enumerate(self.months)}

        def wide(col):
            return p.pivot(index="k", columns="ym", values=col).reindex(index=self.units, columns=self.months)

        self.pop = wide("pop")
        self.logs = {"로그 총인구": np.log(self.pop)}
        self.logs.update({f"로그 {name}": np.log(wide(col)) for col, name in AGES.items()})
        self.cum = {}
        for name, cols in FLOWS.items():
            if cols is None:
                flow = wide("in_inter") + wide("in_intra") - wide("out_inter") - wide("out_intra")
            else:
                flow = sum(wide(c) for c in cols)
            self.cum[name] = flow.cumsum(axis=1, skipna=False)
        self.outcomes = list(self.logs) + list(FLOWS)

    def change(self, outcome, rows, tau, h):
        """y(t+h) - y(t-1). 이동은 t-1 인구 1천 명당 누적."""
        if outcome in self.logs:
            a = self.logs[outcome].values[rows]
            return a[:, tau + h] - a[:, tau - 1]
        c, p = self.cum[outcome].values[rows], self.pop.values[rows]
        return (c[:, tau + h] - c[:, tau - 1]) / p[:, tau - 1] * 1000


def long_run_variance(e, lag):
    e = e - e.mean()
    s = g0 = (e * e).mean()
    for l in range(1, min(lag, len(e) - 1) + 1):
        s += 2 * (1 - l / (lag + 1)) * (e[l:] * e[:-l]).mean()
    return max(s, g0)


def two_way_demean(m):
    return m - m.mean(1, keepdims=True) - m.mean(0, keepdims=True) + m.mean()


def origins(t0, h, lookback, same_month):
    lo = max(t0 - lookback, 1 - min(h, 0))
    hi = t0 - 1 - max(h, 0)
    tau = np.arange(lo, hi + 1)
    if same_month:
        tau = tau[(t0 - tau) % 12 == 0]
        if len(tau) < 4:
            return None
    elif len(tau) < 12:
        return None
    return np.append(tau, t0)


def prediction_se(pre, tau_pre, t0, lag, trend):
    """처치 시군 - 비교군 평균의 폐광 전 경로로 t0 값을 예측할 때의 오차 SE."""
    if not trend:
        s2 = pre.var(ddof=1)
        lrv = long_run_variance(pre, lag)
        return pre.mean(), np.sqrt(s2 + 2 * lrv / len(pre))
    s = (tau_pre - tau_pre.mean()).astype(float)
    slope = (s * (pre - pre.mean())).sum() / (s ** 2).sum()
    fit = pre.mean() + slope * (t0 - tau_pre.mean())
    resid = pre - pre.mean() - slope * s
    s2 = (resid ** 2).sum() / (len(resid) - 2)
    leverage = 1 / len(pre) + (t0 - tau_pre.mean()) ** 2 / (s ** 2).sum()
    return fit, np.sqrt(s2 + long_run_variance(resid, lag) * (1 / len(pre) + leverage))


def event_block(panel, event, outcome, h, controls, lookback=60, trend=False, same_month=False):
    t0 = panel.index[event["t0"]]
    if t0 + h >= len(panel.months) or t0 + min(h, 0) - 1 < 0:
        return None
    tau = origins(t0, h, lookback, same_month)
    if tau is None:
        return None

    names = controls + [event["k"]]
    Y = panel.change(outcome, [panel.units.index(u) for u in names], tau, h)
    ok = np.isfinite(Y).all(axis=1)
    if not ok[-1]:
        return None
    Y, names = Y[ok], [n for n, o in zip(names, ok) if o]

    D = np.zeros_like(Y)
    D[-1, -1] = 1.0
    Yt, Dt = two_way_demean(Y), two_way_demean(D)
    if trend:
        X = np.column_stack([np.ones(len(tau)), tau - tau.mean()])
        proj = X @ np.linalg.pinv(X)
        Yt, Dt = Yt - Yt @ proj.T, Dt - Dt @ proj.T

    gap = Y[-1] - Y[:-1].mean(0)
    window = abs(h) + 1 if h >= 0 else abs(h) - 1
    lag = max(1, int(np.ceil(window / 12))) if same_month else max(1, window)
    fit, se_pred = prediction_se(gap[:-1], tau[:-1], t0, lag, trend)
    return {"y": Yt.ravel(), "d": Dt.ravel(), "cluster": np.repeat(names, len(tau)),
            "simple": gap[-1] - fit, "se_pred": se_pred}


def pooled_ols(blocks):
    y = np.concatenate([b["y"] for b in blocks])
    x = np.concatenate([b["d"] for b in blocks])
    cluster = np.concatenate([b["cluster"] for b in blocks])
    sxx = (x * x).sum()
    beta = (x * y).sum() / sxx
    score = pd.Series(x * (y - beta * x)).groupby(cluster).sum().values
    G = len(score)
    return beta, np.sqrt(G / (G - 1) * (score ** 2).sum() / sxx ** 2)


def comparison_groups(panel, closures_csv):
    d = pd.read_csv(closures_csv)
    d = d[d["지역명"].str.split().str[0].isin(PROVINCE)]
    d["k"] = d["지역명"].map(lambda s: PROVINCE[s.split()[0]] + " " + s.split()[1][:-1] if len(s.split()) == 2 else None)
    closures = d.groupby("k")["값"].sum()
    coal = set(closures[closures >= 3].index)
    treated = {e["k"] for e in EVENTS}
    base = [u for u in panel.units if u not in coal and u not in treated]
    return {
        "주사양": dict(controls=base),
        "시군별 선형추세": dict(controls=base, trend=True),
        "비교군 강원·전남": dict(controls=[u for u in base if u.split()[0] in ("강원", "전남")]),
        "비교군 폐광이력 전무": dict(controls=[u for u in base if u not in set(closures[closures >= 1].index)]),
        "사전기간 36개월": dict(controls=base, lookback=36),
        "같은 달 기점(계절성)": dict(controls=base, lookback=120, same_month=True),
    }


def estimate(panel, specs):
    rows = []
    for spec, kw in specs.items():
        for outcome in panel.outcomes:
            for h in (h for h in range(-24, 25) if h != -1):
                blocks = {e["id"]: event_block(panel, e, outcome, h, **kw) for e in EVENTS}
                for group, ids in GROUPS.items():
                    got = [blocks[i] for i in ids if blocks[i] is not None]
                    if not got:
                        continue
                    beta, se_cl = pooled_ols(got)
                    sv = np.array([b["se_pred"] for b in got])
                    n = len(got)
                    rows.append({
                        "사양": spec, "결과변수": outcome, "묶음": group, "h": h, "사건수": n,
                        "계수": beta, "군집SE": se_cl,
                        "단순차이계수": np.mean([b["simple"] for b in got]),
                        "예측오차SE": np.sqrt((sv ** 2).sum()) / n,
                        "예측오차SE_상관05": np.sqrt((sv ** 2).sum() + 0.5 * (sv.sum() ** 2 - (sv ** 2).sum())) / n,
                    })
            print(spec, outcome, flush=True)
    return pd.DataFrame(rows)


def finalize(res):
    for level, z in [("90", 1.645), ("95", 1.96)]:
        res[f"하한{level}_예측오차"] = res["계수"] - z * res["예측오차SE"]
        res[f"상한{level}_예측오차"] = res["계수"] + z * res["예측오차SE"]
    res["MDE80_예측오차"] = MDE_FACTOR * res["예측오차SE"]
    log = res["결과변수"].str.startswith("로그")
    scaled = [c for c in res.columns if c not in ("사양", "결과변수", "묶음", "h", "사건수")]
    res.loc[log, scaled] = res.loc[log, scaled] * 100
    res["단위"] = np.where(log, "%(로그×100)", "인구 1천 명당 누적")
    return res


def main(panel_csv, closures_csv, out):
    out.mkdir(parents=True, exist_ok=True)
    panel = Panel(panel_csv)
    res = finalize(estimate(panel, comparison_groups(panel, closures_csv)))
    res.to_csv(out / "lp_all.csv", index=False, encoding="utf-8-sig")

    main_spec = res[res["사양"] == "주사양"]
    print("회귀 계수와 단순 차이의 최대 차:", (main_spec["계수"] - main_spec["단순차이계수"]).abs().max())
    key = main_spec[main_spec["h"].isin([11, 23])]
    print(key[["결과변수", "묶음", "h", "사건수", "계수", "예측오차SE"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
