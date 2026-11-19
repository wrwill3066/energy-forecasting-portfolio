"""기상청 API허브 해양기상부이 시간자료를 받아 바람·파고 열만 CSV로 합친다.

    KMA_API_KEY=... python kma_buoy.py download <원자료_폴더>
    python kma_buoy.py tidy <원자료_폴더> <출력.csv.gz>

인증키는 apihub.kma.go.kr에서 발급받아 환경변수로 넘긴다.
"""
import calendar
import gzip
import os
import sys
import time
import urllib.request
from pathlib import Path

URL = "https://apihub.kma.go.kr/api/typ01/url/kma_buoy2.php"
COLUMNS = ["TM", "STN", "WD1", "WS1", "WS1_GST", "WD2", "WS2", "WS2_GST", "PA", "HM", "TA", "TW",
           "WH_MAX", "WH_SIG", "WH_AVE", "WP", "WO", "AQC", "MQC"]
KEEP = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 17, 18]
MISSING = {"-99", "-99.0"}


def download(raw_dir, start=2015, end=2025):
    key = os.environ["KMA_API_KEY"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    for year in range(start, end + 1):
        for month in range(1, 13):
            path = raw_dir / f"kma_buoy_{year}{month:02d}.txt"
            if path.exists() and path.stat().st_size > 1000:
                continue
            last = calendar.monthrange(year, month)[1]
            url = (f"{URL}?tm1={year}{month:02d}010000&tm2={year}{month:02d}{last}2300"
                   f"&stn=0&help=1&authKey={key}")
            for attempt in range(5):
                try:
                    path.write_bytes(urllib.request.urlopen(url, timeout=300).read())
                    print("done", year, month, flush=True)
                    break
                except OSError as e:
                    print("retry", year, month, e, flush=True)
                    time.sleep(30)
            time.sleep(2)


def tidy(raw_dir, out):
    n, stations = 0, {}
    with gzip.open(out, "wt", encoding="utf-8", newline="") as g:
        g.write(",".join(COLUMNS[i] for i in KEEP) + "\n")
        for path in sorted(raw_dir.glob("kma_buoy_*.txt")):
            for line in path.open(encoding="utf-8", errors="replace"):
                if line.startswith("#") or "," not in line:
                    continue
                parts = [x.strip() for x in line.rstrip().rstrip("=").rstrip(",").split(",")]
                if len(parts) < len(COLUMNS) or not parts[0].endswith("00"):
                    continue
                values = ["" if parts[i] in MISSING else parts[i] for i in KEEP]
                g.write(",".join(values) + "\n")
                n += 1
                stations[parts[1]] = stations.get(parts[1], 0) + 1
    print("rows", n, "stations", len(stations))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "download":
        download(Path(sys.argv[2]))
    else:
        tidy(Path(sys.argv[2]), Path(sys.argv[3]))
