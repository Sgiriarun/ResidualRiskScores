"""Download NHANES hip accelerometry and preserve seven separate wear days."""

from __future__ import annotations

import argparse
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "source"
RAW = DATA / "raw"
TABLES = HERE / "tables"
for directory in (DATA, RAW, TABLES):
    directory.mkdir(parents=True, exist_ok=True)

ZIP_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.ZIP"
CYCLES = {"C": (2003, "paxraw_c"), "D": (2005, "paxraw_d")}
NONWEAR_MIN = 60
VALID_DAY_MIN = 600
MIN_VALID_DAYS = 4
MVPA_CUT = 2020
DAYS = 7
MINUTES = 1440

def download_archive(year: int, name: str) -> Path:
    destination = RAW / f"{name}.ZIP"
    if destination.exists() and zipfile.is_zipfile(destination):
        print(f"Using cached {destination}", flush=True)
        return destination
    partial = destination.with_suffix(".ZIP.part")
    url = ZIP_URL.format(year=year, name=name)
    for attempt in range(1, 7):
        existing = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "Mozilla/5.0"}
        if existing:
            headers["Range"] = f"bytes={existing}-"
            print(f"Resuming {name}.ZIP from {existing / (1 << 20):.1f} MB (attempt {attempt}/6)", flush=True)
        else:
            print(f"Downloading {name}.ZIP (attempt {attempt}/6)", flush=True)
        try:
            response = requests.get(url, stream=True, headers=headers, timeout=(30, 900))
            response.raise_for_status()
            append = existing > 0 and response.status_code == 206
            with partial.open("ab" if append else "wb") as handle:
                for block in response.iter_content(1 << 20):
                    if block:
                        handle.write(block)
            break
        except requests.RequestException as error:
            if attempt == 6:
                raise
            print(f"  transfer interrupted: {error}; retrying", flush=True)
            time.sleep(min(5 * attempt, 20))
    if not zipfile.is_zipfile(partial):
        raise ValueError(f"Completed transfer is not a valid zip: {partial}")
    os.replace(partial, destination)
    if not zipfile.is_zipfile(destination):
        raise ValueError(f"Downloaded archive is not a valid zip: {destination}")
    return destination


def extract_xpt(archive: Path, cycle: str) -> Path:
    output = Path("/tmp") / f"average_day_paxraw_{cycle}.xpt"
    if output.exists():
        return output
    with zipfile.ZipFile(archive) as zipped:
        member = next(name for name in zipped.namelist() if name.lower().endswith(".xpt"))
        with zipped.open(member) as source, output.open("wb") as target:
            while block := source.read(1 << 20):
                target.write(block)
    return output


def nonwear_mask(intensity: np.ndarray) -> np.ndarray:
    zero = intensity == 0
    edges = np.diff(np.concatenate(([0], zero.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    result = np.zeros(len(intensity), dtype=bool)
    for start, end in zip(starts, ends):
        if end - start >= NONWEAR_MIN:
            result[start:end] = True
    return result


def process_person(group: pd.DataFrame):
    group = group[(group.PAXSTAT == 1) & (group.PAXCAL == 1)]
    counts = np.zeros((DAYS, MINUTES), dtype=np.float32)
    wear_mask = np.zeros((DAYS, MINUTES), dtype=bool)
    valid_days = np.zeros(DAYS, dtype=bool)
    wear_minutes = np.zeros(DAYS, dtype=np.int16)
    daily_mvpa = np.full(DAYS, np.nan, dtype=np.float32)
    for day_value, day in group.groupby("PAXDAY"):
        day_number = int(day_value)
        if not 1 <= day_number <= DAYS:
            continue
        day = day.sort_values("PAXN")
        intensity = day.PAXINTEN.to_numpy(np.float64)
        minute = ((day.PAXN.to_numpy(np.int64) - 1) % MINUTES).astype(int)
        wear = ~nonwear_mask(intensity)
        if int(wear.sum()) < VALID_DAY_MIN:
            continue
        index = day_number - 1
        sums = np.zeros(MINUTES, dtype=np.float64)
        observations = np.zeros(MINUTES, dtype=np.int16)
        np.add.at(sums, minute[wear], intensity[wear])
        np.add.at(observations, minute[wear], 1)
        observed = observations > 0
        counts[index, observed] = (sums[observed] / observations[observed]).astype(np.float32)
        wear_mask[index, observed] = True
        valid_days[index] = True
        wear_minutes[index] = int(wear.sum())
        daily_mvpa[index] = float(((intensity >= MVPA_CUT) & wear).sum())
    if valid_days.sum() < MIN_VALID_DAYS:
        return None
    return counts, wear_mask, valid_days, wear_minutes, daily_mvpa


def build_cycle(cycle: str, keep_extracted: bool = False) -> None:
    output = DATA / f"weekly_{cycle}.npz"
    if output.exists():
        print(f"Weekly cache already exists: {output}")
        validate_cycle(cycle)
        return
    year, name = CYCLES[cycle]
    archive = download_archive(year, name)
    xpt = extract_xpt(archive, cycle)
    print(f"Streaming all participants from {xpt.name}", flush=True)
    reader = pd.read_sas(xpt, format="xport", chunksize=1_000_000, iterator=True)
    records: dict[int, tuple] = {}
    buffer: list[pd.DataFrame] = []
    current = None
    seen = 0

    def finish(seqn, pieces):
        nonlocal seen
        if seqn is None:
            return
        seen += 1
        seqn = int(seqn)
        result = process_person(pd.concat(pieces, ignore_index=True))
        if result is not None:
            records[seqn] = result

    for chunk in reader:
        columns = ["SEQN", "PAXDAY", "PAXN", "PAXSTAT", "PAXCAL", "PAXINTEN"]
        chunk = chunk[columns]
        for seqn, group in chunk.groupby("SEQN", sort=False):
            if current is None:
                current = seqn
            if seqn != current:
                finish(current, buffer)
                buffer, current = [], seqn
            buffer.append(group)
    finish(current, buffer)

    order = np.asarray(sorted(records), dtype=np.int64)
    if len(order) == 0:
        raise ValueError(f"Cycle {cycle}: no participants passed the wear criteria")
    unpacked = [records[int(seqn)] for seqn in order]
    np.savez_compressed(
        output,
        seqn=order,
        cycle=np.full(len(order), cycle),
        counts=np.stack([item[0] for item in unpacked]),
        wear_mask=np.stack([item[1] for item in unpacked]),
        valid_day_mask=np.stack([item[2] for item in unpacked]),
        wear_minutes=np.stack([item[3] for item in unpacked]),
        daily_mvpa=np.stack([item[4] for item in unpacked]),
    )
    print(f"Cycle {cycle}: processed {seen:,} raw participants; saved {len(order):,} -> {output}")
    if not keep_extracted:
        xpt.unlink(missing_ok=True)
    validate_cycle(cycle)


def validate_cycle(cycle: str) -> dict[str, float]:
    with np.load(DATA / f"weekly_{cycle}.npz") as weekly:
        seqn = weekly["seqn"].astype(int)
        counts = weekly["counts"]
        mask = weekly["wear_mask"]
        valid = weekly["valid_day_mask"]
        daily_mvpa = weekly["daily_mvpa"]
        wear_minutes = weekly["wear_minutes"]
    if len(np.unique(seqn)) != len(seqn):
        raise AssertionError(f"Cycle {cycle}: duplicate participant identifiers")
    if counts.shape != mask.shape or counts.shape[1:] != (DAYS, MINUTES):
        raise AssertionError(f"Cycle {cycle}: unexpected weekly array shape")
    if (valid.sum(axis=1) < MIN_VALID_DAYS).any():
        raise AssertionError(f"Cycle {cycle}: participant below valid-day minimum")
    if np.any(counts[~mask] != 0):
        raise AssertionError(f"Cycle {cycle}: activity found outside the wear mask")
    valid_wear = wear_minutes[valid]
    if (valid_wear < VALID_DAY_MIN).any():
        raise AssertionError(f"Cycle {cycle}: valid day below wear-time minimum")
    metrics = {
        "cycle": cycle,
        "participants": len(seqn),
        "mean_valid_days": float(valid.sum(axis=1).mean()),
        "minimum_valid_days": int(valid.sum(axis=1).min()),
        "mean_daily_wear_minutes": float(valid_wear.mean()),
        "mean_daily_mvpa_minutes": float(np.nanmean(daily_mvpa)),
    }
    existing = TABLES / "weekly_data_validation.csv"
    previous = pd.read_csv(existing) if existing.exists() else pd.DataFrame()
    previous = previous[previous.cycle != cycle] if len(previous) else previous
    pd.concat([previous, pd.DataFrame([metrics])], ignore_index=True).to_csv(existing, index=False)
    print(pd.DataFrame([metrics]).round(4).to_string(index=False))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", choices=("C", "D", "both"), default="both")
    parser.add_argument("--keep-extracted", action="store_true")
    args = parser.parse_args()
    cycles = ("C", "D") if args.cycle == "both" else (args.cycle,)
    for cycle in cycles:
        build_cycle(cycle, keep_extracted=args.keep_extracted)


if __name__ == "__main__":
    main()
