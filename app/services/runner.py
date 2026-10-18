"""Glue used by both the live and the manual endpoints: predict (off the event loop) and record history."""
from __future__ import annotations

from typing import Any, Optional

from starlette.concurrency import run_in_threadpool

from app.core.features import WeatherInputs


async def predict_and_record(state: Any, *, inputs: WeatherInputs, source: str, latitude: Optional[float] = None,
                             longitude: Optional[float] = None, area_confirmed: Optional[bool] = None,
                             data_timestamp: Optional[str] = None, location_name: Optional[str] = None) -> dict[str, Any]:
    """``state`` is ``app.state``. Raises ``NoPredictionPossible`` if the data cannot support any estimate."""
    result = await run_in_threadpool(state.predictor.predict, inputs, latitude=latitude, longitude=longitude,
                                     area_confirmed=area_confirmed, data_timestamp=data_timestamp)
    if state.settings.save_history:
        await run_in_threadpool(state.db.add_prediction, source=source, result=result, inputs=inputs.to_dict(),
                                location_name=location_name, latitude=latitude, longitude=longitude)
    return result
