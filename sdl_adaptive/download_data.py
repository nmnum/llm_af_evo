"""
download_data.py — Download ADA datasets from GitHub.

Usage
-----
python download_data.py --data_dir data/

Downloads
---------
data/coatings_2022.csv                        — ADA 2022 coatings campaign (162 KB)
data/campaign 2020-12-18_17-38-40.csv         — ADA 2021 Pareto campaign 1
data/campaign 2020-12-23_17-06-50.csv         — ADA 2021 Pareto campaign 2
data/campaign 2021-01-04_08-37-39.csv         — ADA 2021 Pareto campaign 3
data/campaign 2021-01-12_16-26-56.csv         — ADA 2021 Pareto campaign 4

Source: https://github.com/berlinguette/ada
Note: Files are served via Git LFS — uses media.githubusercontent.com endpoint.
"""

import argparse
import pathlib
import urllib.parse

import requests

# ── URLs ──────────────────────────────────────────────────────────────────────
COATINGS_URL = (
    "https://media.githubusercontent.com/media/berlinguette/ada/master/"
    "2022_09%20A%20self-driving%20laboratory%20optimizes%20a%20scalable%20"
    "materials%20manufacturing%20process/optimization%20campaign%20data/"
    "compiled_optimization_data.csv"
)

PARETO_BASE = (
    "https://media.githubusercontent.com/media/berlinguette/ada/master/"
    "2021_01%20Self-driving%20laboratories%20can%20advance%20the%20Pareto%20"
    "front%20for%20thin-film%20materials/processed_data/"
)

PARETO_FILES = {
    "campaign 2020-12-18_17-38-40.csv": "campaign%202020-12-18_17-38-40.csv",
    "campaign 2020-12-23_17-06-50.csv": "campaign%202020-12-23_17-06-50.csv",
    "campaign 2021-01-04_08-37-39.csv": "campaign%202021-01-04_08-37-39.csv",
    "campaign 2021-01-12_16-26-56.csv": "campaign%202021-01-12_16-26-56.csv",
}


def download(url: str, dest: pathlib.Path, min_size: int = 500) -> bool:
    if dest.exists() and dest.stat().st_size > min_size:
        print(f"  already exists: {dest.name} ({dest.stat().st_size // 1024} KB)")
        return True
    try:
        r = requests.get(url, timeout=60)
        if r.status_code == 200 and len(r.content) > min_size:
            dest.write_bytes(r.content)
            print(f"  downloaded: {dest.name} ({len(r.content) // 1024} KB)")
            return True
        else:
            print(f"  FAILED: {dest.name} — HTTP {r.status_code}, {len(r.content)} bytes")
            return False
    except Exception as e:
        print(f"  ERROR: {dest.name} — {e}")
        return False


def main(data_dir: str):
    data_dir = pathlib.Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading ADA 2022 coatings dataset...")
    ok = download(COATINGS_URL, data_dir / "coatings_2022.csv")
    if not ok:
        print("  !! coatings_2022.csv download failed. Check your internet connection.")

    print("\nDownloading ADA 2021 Pareto front campaigns...")
    for local_name, url_name in PARETO_FILES.items():
        download(PARETO_BASE + url_name, data_dir / local_name)

    print("\nVerifying downloads...")
    all_ok = True
    for fname in ["coatings_2022.csv"] + list(PARETO_FILES.keys()):
        p = data_dir / fname
        if p.exists() and p.stat().st_size > 500:
            print(f"  OK: {fname} ({p.stat().st_size // 1024} KB)")
        else:
            print(f"  MISSING or empty: {fname}")
            all_ok = False

    if all_ok:
        print("\nAll datasets downloaded successfully.")
    else:
        print("\nSome downloads failed. See messages above.")
    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ADA datasets")
    parser.add_argument("--data_dir", default="data", help="Directory to save data files")
    args = parser.parse_args()
    main(args.data_dir)
