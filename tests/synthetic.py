"""SYNTHETIC fixtures used ONLY by the automated tests.

Nothing generated here is real weather or real incident data. It exists to prove
that the pipeline code runs end-to-end and rejects bad input. It must never be
used to produce a deployed model, and no metric computed from it may be reported
as model performance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml.dataset import assemble_dataset


def synthetic_units(n_units: int = 3, start: str = "2020-01-01", end: str = "2023-12-31", seed: int = 0):
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, end, freq="h")
    units = []
    for u in range(n_units):
        # bursty rain: rare storms of a few hours
        storm = rng.random(len(index)) < 0.01
        rain = np.zeros(len(index))
        for i in np.where(storm)[0]:
            length = rng.integers(2, 8)
            rain[i:i + length] += rng.gamma(2.0, 3.0, size=min(length, len(index) - i))
        weather = pd.DataFrame({
            "precipitation": rain,
            "temperature_2m": 12 + 10 * np.sin(np.arange(len(index)) / 24 / 58) + rng.normal(0, 2, len(index)),
            "relative_humidity_2m": np.clip(70 + rng.normal(0, 10, len(index)), 5, 100),
            "wind_speed_10m": np.abs(rng.normal(12, 5, len(index))),
        }, index=index)
        # events follow heavy recent rain (+ noise), created in the hours after the rain
        recent = pd.Series(rain, index=index).rolling(3, min_periods=1).sum().shift(1).fillna(0)
        lam = 0.02 + 0.6 * np.clip(recent.to_numpy() - 3, 0, None)
        counts = pd.Series(rng.poisson(lam), index=index)
        units.append({"unit_id": f"u{u}", "latitude": 40.7 + 0.01 * u, "longitude": -74.0 + 0.01 * u,
                      "weather": weather, "counts": counts})
    return units


def synthetic_dataset(**kw) -> pd.DataFrame:
    df, _ = assemble_dataset(synthetic_units(**kw))
    return df
