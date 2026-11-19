"""재생 가뭄(renewable drought) 지표.

Kittel & Schill (2024)의 MBT(mean-below-threshold)와 CBT(consecutive-below-threshold) 정의를 따른다.
임계값 tau는 장기평균 대비 비율이다.
"""
import numpy as np
import pandas as pd


def daily_mean(series):
    """24시간이 모두 있는 날만 평균을 내고, 나머지는 결측으로 둔 채 달력 전체로 다시 맞춘다."""
    s = series.replace([np.inf, -np.inf], np.nan)
    g = s.groupby(level=0)
    d = g.mean().where(g.count() == 24)
    return d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D"))


def max_drought(daily, tau, max_len=None):
    """L일 이동평균이 tau 미만인 가장 긴 L (MBT). 결측일이 낀 구간은 제외한다.

    길이 L+1 구간의 평균이 tau 미만이면 그 안에 길이 L 구간도 반드시 tau 미만이므로,
    L을 늘려가다 처음 실패하는 지점에서 멈추면 된다.
    """
    v = daily.to_numpy(dtype=float)
    v = v / np.nanmean(v)
    ok = ~np.isnan(v)
    csum = np.concatenate([[0], np.cumsum(np.where(ok, v, 0))])
    cmiss = np.concatenate([[0], np.cumsum(~ok)])
    max_len = len(v) if max_len is None else max_len

    best = 0
    for L in range(1, max_len + 1):
        mean = (csum[L:] - csum[:-L]) / L
        complete = (cmiss[L:] - cmiss[:-L]) == 0
        if not (complete & (mean < tau)).any():
            break
        best = L
    return best


def drought_events(daily, tau, min_len=1):
    """일평균이 tau 미만인 날이 이어지는 구간 목록 (CBT). [(시작일, 길이), ...]"""
    below = (daily / daily.mean() < tau).to_numpy()
    events, i, n = [], 0, len(below)
    while i < n:
        if not below[i]:
            i += 1
            continue
        j = i
        while j < n and below[j]:
            j += 1
        if j - i >= min_len:
            events.append((daily.index[i], j - i))
        i = j
    return events
