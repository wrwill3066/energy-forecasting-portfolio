"""2017~2026년 한국 공휴일과 일별 날짜유형 더미.

공휴일 종류: L 법정, S 대체, E 선거, T 임시, N 명절(설·추석), W 근로자의날
명절 연휴에 붙은 대체공휴일은 명절(N)로 본다.
"""
import numpy as np
import pandas as pd

HOLIDAYS = {
    "2017-01-01": "L", "2017-01-27": "N", "2017-01-28": "N", "2017-01-29": "N", "2017-01-30": "S",
    "2017-03-01": "L", "2017-05-01": "W", "2017-05-03": "L", "2017-05-05": "L", "2017-05-09": "E",
    "2017-06-06": "L", "2017-08-15": "L", "2017-10-02": "T", "2017-10-03": "L", "2017-10-04": "N",
    "2017-10-05": "N", "2017-10-06": "S", "2017-10-09": "L", "2017-12-25": "L",
    "2018-01-01": "L", "2018-02-15": "N", "2018-02-16": "N", "2018-02-17": "N", "2018-03-01": "L",
    "2018-05-01": "W", "2018-05-05": "L", "2018-05-07": "S", "2018-05-22": "L", "2018-06-06": "L",
    "2018-06-13": "E", "2018-08-15": "L", "2018-09-23": "N", "2018-09-24": "N", "2018-09-25": "N",
    "2018-09-26": "S", "2018-10-03": "L", "2018-10-09": "L", "2018-12-25": "L",
    "2019-01-01": "L", "2019-02-04": "N", "2019-02-05": "N", "2019-02-06": "N", "2019-03-01": "L",
    "2019-05-01": "W", "2019-05-05": "L", "2019-05-06": "S", "2019-05-12": "L", "2019-06-06": "L",
    "2019-08-15": "L", "2019-09-12": "N", "2019-09-13": "N", "2019-09-14": "N", "2019-10-03": "L",
    "2019-10-09": "L", "2019-12-25": "L",
    "2020-01-01": "L", "2020-01-24": "N", "2020-01-25": "N", "2020-01-26": "N", "2020-01-27": "S",
    "2020-03-01": "L", "2020-04-15": "E", "2020-04-30": "L", "2020-05-01": "W", "2020-05-05": "L",
    "2020-06-06": "L", "2020-08-15": "L", "2020-08-17": "T", "2020-09-30": "N", "2020-10-01": "N",
    "2020-10-02": "N", "2020-10-03": "L", "2020-10-09": "L", "2020-12-25": "L",
    "2021-01-01": "L", "2021-02-11": "N", "2021-02-12": "N", "2021-03-01": "L", "2021-05-05": "L",
    "2021-05-19": "L", "2021-08-16": "S", "2021-09-20": "N", "2021-09-21": "N", "2021-09-22": "N",
    "2021-10-04": "S", "2021-10-11": "S",
    "2022-01-31": "N", "2022-02-01": "N", "2022-02-02": "N", "2022-03-01": "L", "2022-03-09": "E",
    "2022-05-05": "L", "2022-06-01": "E", "2022-06-06": "L", "2022-08-15": "L", "2022-09-09": "N",
    "2022-09-12": "S", "2022-10-03": "L", "2022-10-10": "S",
    "2023-01-23": "N", "2023-01-24": "S", "2023-03-01": "L", "2023-05-01": "W", "2023-05-05": "L",
    "2023-05-29": "S", "2023-06-06": "L", "2023-08-15": "L", "2023-09-28": "N", "2023-09-29": "N",
    "2023-10-02": "T", "2023-10-03": "L", "2023-10-09": "L", "2023-12-25": "L",
    "2024-01-01": "L", "2024-02-09": "N", "2024-02-12": "S", "2024-03-01": "L", "2024-04-10": "E",
    "2024-05-01": "W", "2024-05-06": "S", "2024-05-15": "L", "2024-06-06": "L", "2024-08-15": "L",
    "2024-09-16": "N", "2024-09-17": "N", "2024-09-18": "N", "2024-10-01": "T", "2024-10-03": "L",
    "2024-10-09": "L", "2024-12-25": "L",
    "2025-01-01": "L", "2025-01-27": "T", "2025-01-28": "N", "2025-01-29": "N", "2025-01-30": "N",
    "2025-03-03": "S", "2025-05-01": "W", "2025-05-05": "L", "2025-05-06": "S", "2025-06-03": "E",
    "2025-06-06": "L", "2025-08-15": "L", "2025-10-03": "L", "2025-10-06": "N", "2025-10-07": "N",
    "2025-10-08": "S", "2025-10-09": "L", "2025-12-25": "L",
    "2026-01-01": "L", "2026-02-16": "N", "2026-02-17": "N", "2026-02-18": "N", "2026-03-02": "S",
    "2026-05-01": "W", "2026-05-05": "L", "2026-05-25": "S", "2026-06-03": "E",
}

FESTIVAL_SUBSTITUTE = {"2017-01-30", "2017-10-03", "2017-10-06", "2018-09-26", "2020-01-27",
                       "2022-09-12", "2023-01-24", "2024-02-12", "2025-10-08"}

DAY_TYPES = ["Nat", "NatWE", "Hol_single", "Hol_first", "Hol_mid", "Hol_last", "Elec", "Temp", "Labor",
             "Sand", "NRm4", "NRm3", "NRm2", "NRm1", "NRp1", "NRp2", "NRp3", "NRp4", "Sat", "Sun", "WEhol"]


def holidays():
    return {pd.Timestamp(d): ("N" if d in FESTIVAL_SUBSTITUTE else k) for d, k in HOLIDAYS.items()}


def off_blocks(days, hol):
    """주말·공휴일이 이어지는 구간마다 번호를 붙인다 (평일은 0)."""
    off = pd.Series([(d.weekday() >= 5) or (d in hol) for d in days], index=days)
    return (off != off.shift()).cumsum().where(off, 0)


def calendar(start="2016-12-01", end="2026-12-31"):
    """일별 날짜유형 더미.

    Nat        평일 명절
    NatWE      명절 연휴 안의 주말
    Hol_*      명절이 아닌 평일 공휴일. 연휴 안 위치에 따라 single/first/mid/last
    Elec, Temp, Labor  선거일, 임시공휴일, 근로자의날
    Sand       징검다리 평일 (공휴일 사이에 낀 월·금)
    NRm/NRp k  명절 연휴 k번째 전/후 평일
    Sat, Sun, WEhol  명절과 무관한 토·일, 주말과 겹친 공휴일
    """
    hol = holidays()
    days = pd.date_range(start, end)
    block = off_blocks(days, hol)
    size = block.groupby(block).transform("size").where(block > 0, 0)
    pos = block.groupby(block).cumcount() + 1

    kind = pd.Series([hol.get(d, "") for d in days], index=days)
    weekday = pd.Series(days.weekday < 5, index=days)
    festival_blocks = {block[d] for d, k in hol.items() if k == "N" and d in block.index}
    in_festival = block.isin(festival_blocks) & (block > 0)

    rel = pd.Series(np.nan, index=days)
    for b in festival_blocks:
        span = block.index[block == b]
        for k in range(1, 5):
            before, after = span[0] - pd.Timedelta(days=k), span[-1] + pd.Timedelta(days=k)
            if before in rel.index:
                rel[before] = -k
            if after in rel.index:
                rel[after] = k

    cal = pd.DataFrame(index=days)
    cal["Nat"] = ((kind == "N") & weekday).astype(int)
    cal["NatWE"] = (in_festival & ~weekday).astype(int)

    regular = weekday & kind.isin(["L", "S"])
    position = np.select([size == 1, pos == 1, pos == size], ["single", "first", "last"], "mid")
    for p in ["single", "first", "mid", "last"]:
        cal[f"Hol_{p}"] = (regular & (position == p)).astype(int)

    cal["Elec"] = ((kind == "E") & weekday).astype(int)
    cal["Temp"] = ((kind == "T") & weekday).astype(int)
    cal["Labor"] = ((kind == "W") & weekday).astype(int)

    next_off = pd.Series([(d + pd.Timedelta(days=1)) in hol for d in days], index=days)
    prev_off = pd.Series([(d - pd.Timedelta(days=1)) in hol for d in days], index=days)
    sandwich = ((days.weekday == 0) & next_off) | ((days.weekday == 4) & prev_off)
    cal["Sand"] = (weekday & (kind == "") & sandwich).astype(int)

    for k in [-4, -3, -2, -1, 1, 2, 3, 4]:
        cal[f"NR{'m' if k < 0 else 'p'}{abs(k)}"] = ((rel == k) & (kind == "") & weekday).astype(int)

    plain = ~in_festival & (kind == "")
    cal["Sat"] = ((days.weekday == 5) & plain).astype(int)
    cal["Sun"] = ((days.weekday == 6) & plain).astype(int)
    cal["WEhol"] = (~weekday & (kind != "") & ~in_festival).astype(int)
    cal.index.name = "date"
    return cal


def is_normal_weekday(cal, hol):
    """공휴일, 명절 전후, 징검다리 어디에도 해당하지 않는 평일."""
    special = cal.drop(columns=["Sat", "Sun"]).sum(axis=1) > 0
    return pd.Series([(d.weekday() < 5) and (d not in hol) and not special[d] for d in cal.index], index=cal.index)
