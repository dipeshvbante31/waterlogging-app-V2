#!/usr/bin/env python
"""Build the training table from real, public data. Run on your own computer (needs internet).

Default source  (--source nyc311):
    hourly weather  : Open-Meteo Historical Weather API (ERA5-based reanalysis), one point per NYC borough
    observed outcome: NYC 311 street-flooding complaints, counted per borough per hour

Your own events (--source custom): supply a CSV of dated incident reports and a CSV of locations.
    events CSV : columns  timestamp, unit_id
    units CSV  : columns  unit_id, latitude, longitude
  Use this to train for an area such as Mumbai IF you legitimately hold dated waterlogging records
  (for example obtained from the municipal corporation). Nothing is invented: no records, no dataset.

Examples
    python scripts/prepare_dataset.py --list-descriptors
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --source custom --events-csv my_events.csv --units-csv my_units.csv \\
        --events-tz Asia/Kolkata --region-name "My City" --events-source "Municipal records, RTI 2026/123" \\
        --events-licence "Permission granted by ..."

Outputs (git-ignored): data/processed/<name>.csv and <name>.manifest.json. Provider responses are cached in data/raw.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
import pandas as pd  # noqa: E402

from app.ml.dataset import WINDOW_HOURS, WEATHER_COLUMNS, assemble_dataset, local_to_utc_hours  # noqa: E402
from app.ml import sources as src  # noqa: E402

NYC_TZ = "America/New_York"


def log(message: str) -> None:
    print(message, flush=True)


def counts_series(local_times: pd.Series, n: pd.Series, tz: str) -> pd.Series:
    utc = local_to_utc_hours(local_times, tz)
    keep = utc.notna()
    return pd.Series(pd.to_numeric(n[keep]).to_numpy(), index=pd.DatetimeIndex(utc[keep]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["nyc311", "custom"], default="nyc311")
    p.add_argument("--start", type=date.fromisoformat, default=date(2016, 1, 1))
    p.add_argument("--end", type=date.fromisoformat, default=date(2026, 9, 30),
                   help="last day to include; keep it at least ~10 days in the past (reanalysis and 311 lag)")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--name", default=None, help="output file stem (default depends on --source)")
    p.add_argument("--app-token", default=os.getenv("SOCRATA_APP_TOKEN"), help="optional NYC Open Data app token (raises rate limits)")
    p.add_argument("--complaint-type", default="Sewer", help="311 complaint_type to count (verify with --list-descriptors)")
    p.add_argument("--descriptor-like", default="Street Flooding%", help="SoQL LIKE pattern for the 311 descriptor")
    p.add_argument("--list-descriptors", action="store_true", help="print 311 descriptors containing 'flood' and exit")
    p.add_argument("--events-csv", type=Path)
    p.add_argument("--units-csv", type=Path)
    p.add_argument("--events-tz", default="UTC", help="time zone of the events' timestamps (IANA name)")
    p.add_argument("--region-name")
    p.add_argument("--events-source", help="where the custom events came from (recorded in the manifest)")
    p.add_argument("--events-licence", help="terms under which you may use them (recorded in the manifest)")
    p.add_argument("--max-distance-km", type=float, default=25.0, help="radius around each unit where the model may be used")
    return p.parse_args()


def build_nyc(args, client):
    log(f"Downloading 311 '{args.complaint_type}' / '{args.descriptor_like}' counts {args.start}..{args.end}")
    raw = src.fetch_nyc311_counts(client, args.start, args.end, args.complaint_type, args.descriptor_like,
                                  args.raw_dir / "nyc311", log, args.app_token)
    raw = raw[raw["borough"].isin(src.NYC_BOROUGH_POINTS)]
    if raw.empty:
        sys.exit("No 311 records matched. Run with --list-descriptors and adjust --complaint-type / --descriptor-like.")
    units = []
    for borough, (lat, lon) in src.NYC_BOROUGH_POINTS.items():
        weather = src.fetch_weather(client, lat, lon, args.start, args.end, args.raw_dir / "weather", log)
        c = raw[raw["borough"] == borough]
        units.append({"unit_id": borough.title().replace(" ", "_"), "latitude": lat, "longitude": lon,
                      "weather": weather, "counts": counts_series(c["local_time"], c["n"], NYC_TZ)})
    sources = [
        {"name": "Open-Meteo Historical Weather API (hourly ERA5-based reanalysis)",
         "url": "https://open-meteo.com/en/docs/historical-weather-api",
         "licence": "Data CC BY 4.0; the free API is for non-commercial use only (check open-meteo.com/en/terms)",
         "variables": list(WEATHER_COLUMNS)},
        {"name": f"NYC Open Data, 311 Service Requests (datasets {src.NYC_311_2010_2019} for 2010-2019 and "
                 f"{src.NYC_311_2020_PRESENT} for 2020 onward)",
         "url": "https://data.cityofnewyork.us/",
         "licence": "NYC Open Data Terms of Use (check data.cityofnewyork.us for current terms)",
         "filter": {"complaint_type": args.complaint_type, "descriptor_like": args.descriptor_like}},
    ]
    region = {"name": "New York City, USA (five boroughs)", "max_distance_km": args.max_distance_km}
    return units, sources, region, "nyc_street_flooding_reports"


def build_custom(args, client):
    for flag in ("events_csv", "units_csv", "region_name", "events_source", "events_licence"):
        if not getattr(args, flag):
            sys.exit(f"--source custom requires --{flag.replace('_', '-')}")
    events = pd.read_csv(args.events_csv)
    units_df = pd.read_csv(args.units_csv)
    for frame, cols, label in ((events, ["timestamp", "unit_id"], "events"), (units_df, ["unit_id", "latitude", "longitude"], "units")):
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            sys.exit(f"The {label} CSV lacks columns: {missing}")
    events["timestamp"] = pd.to_datetime(events["timestamp"], errors="coerce")
    bad = int(events["timestamp"].isna().sum())
    if bad:
        log(f"Dropping {bad} event rows with unparseable timestamps")
        events = events.dropna(subset=["timestamp"])
    units = []
    for row in units_df.itertuples(index=False):
        ev = events[events["unit_id"].astype(str) == str(row.unit_id)]
        local = ev["timestamp"].dt.floor("h")
        counts = counts_series(local, pd.Series(1, index=ev.index), args.events_tz)
        weather = src.fetch_weather(client, float(row.latitude), float(row.longitude), args.start, args.end,
                                    args.raw_dir / "weather", log)
        units.append({"unit_id": str(row.unit_id), "latitude": float(row.latitude), "longitude": float(row.longitude),
                      "weather": weather, "counts": counts})
    sources = [
        {"name": "Open-Meteo Historical Weather API (hourly ERA5-based reanalysis)",
         "url": "https://open-meteo.com/en/docs/historical-weather-api",
         "licence": "Data CC BY 4.0; the free API is for non-commercial use only (check open-meteo.com/en/terms)",
         "variables": list(WEATHER_COLUMNS)},
        {"name": f"User-supplied incident records ({args.events_csv.name})", "url": None,
         "licence": args.events_licence, "provenance": args.events_source},
    ]
    return units, sources, {"name": args.region_name, "max_distance_km": args.max_distance_km}, "custom_incident_reports"


def main() -> int:
    args = parse_args()
    if args.end >= date.today():
        sys.exit("--end must be in the past.")
    headers = {"User-Agent": "WaterloggingWatch-data-prep/2.0 (educational project)"}
    with httpx.Client(timeout=httpx.Timeout(90.0, connect=10.0), headers=headers, follow_redirects=True) as client:
        if args.list_descriptors:
            for row in src.list_flood_descriptors(client, args.start, args.end, log, args.app_token):
                print(row)
            return 0
        units, sources, region, default_name = (build_nyc if args.source == "nyc311" else build_custom)(args, client)

    log("Assembling features and labels (no future information in the features)...")
    table, reports = assemble_dataset(units)
    total_events = int(table["window_event_count"].sum())
    if total_events == 0:
        sys.exit("The assembled dataset contains no events; a supervised model cannot be trained from it.")

    name = args.name or default_name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"{name}.csv"
    table.to_csv(csv_path, index=False)
    region["points"] = [{"unit_id": u["unit_id"], "latitude": u["latitude"], "longitude": u["longitude"]} for u in units]
    manifest = {
        "dataset_name": name, "created_utc": datetime.now(timezone.utc).isoformat(),
        "time_range": {"start": args.start.isoformat(), "end": args.end.isoformat()},
        "region": region, "sources": sources,
        "feature_window": "rain over the last 1/3/24/72 hours plus temperature, humidity and wind at the sample hour",
        "label_window_hours": WINDOW_HOURS,
        "assembly_reports": reports, "rows": int(len(table)), "total_events_in_rows": total_events,
        "weather_resolution_note": "Reanalysis weather is gridded (about 10 km); boroughs/units share grid cells and are not independent.",
    }
    manifest_path = args.out_dir / f"{name}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"Wrote {csv_path} ({len(table):,} rows, {total_events:,} events) and {manifest_path}")
    log(f"Next: python scripts/train_model.py --dataset {csv_path} --manifest {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
