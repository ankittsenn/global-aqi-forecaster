"""Small HTTP API around the forecaster.

    uvicorn api.main:app --reload        # then open http://127.0.0.1:8000/docs
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Query     # noqa: E402

from airq import aqi_scale                            # noqa: E402
from airq.service import Service                      # noqa: E402

app = FastAPI(title="Global AQI Forecaster API", version="1.0.0",
              description="Seven-day PM2.5 forecasts with uncertainty and a 48-hour episode alert. "
                          "Data: CAMS model output and ERA5 via Open-Meteo (CC BY 4.0). "
                          "Model-based grid data, not station measurements.")
_service = None


def service() -> Service:
    global _service
    if _service is None:
        _service = Service()
    return _service


@app.get("/health")
def health():
    s = service()
    return {"status": "ok", "cities": len(s.cat), "trained_until": s.bundle.meta.get("trained_until")}


@app.get("/cities")
def cities():
    return service().cities()


@app.get("/forecast/{city}")
def forecast(city: str, live: bool = Query(True, description="false forecasts from the stored dataset")):
    try:
        return service().forecast(city, live)
    except KeyError:
        raise HTTPException(404, f"Unknown city '{city}'. See /cities.")


@app.get("/alerts")
def alerts():
    return service().alerts()


@app.get("/advice")
def advice(aqi: float = Query(..., ge=0, le=500), group: str = "General public"):
    try:
        return service().advice(aqi, group)
    except ValueError as err:
        raise HTTPException(422, str(err))


@app.get("/groups")
def groups():
    return aqi_scale.GROUPS
