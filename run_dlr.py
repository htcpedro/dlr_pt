import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Callable, Optional, Dict, Any
import time

import numpy as np
import pandas as pd
import requests

from cigre601 import thermal_rating_steady_state

# Constants / configuration
SPANS_FILE = Path("./data/spans.csv")
FORECAST_DIR = Path("./forecasts")
RATINGS_TEMPS = {"nor": 50, "lte": 75, "ste": 85}
WEATHER_SAMPLE_STEP = 10  # sampled every 10 points
HTTP_RETRIES = 3
HTTP_BACKOFF = 1.0  # seconds between retries
REQUEST_TIMEOUT = 10  # seconds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class HeatMaterial:
    name: str
    mass_per_unit_length: float
    specific_heat_20deg: float
    beta: float = 0.0


@dataclass
class ConductorConstants:
    stranded: bool
    high_rs: bool
    diameter: float
    cross_section: Optional[float]
    absorptivity: float
    emissivity: float
    materials_heat: List[HeatMaterial]
    resistance: Callable[[float], float]  # function of conductor_temperature -> resistance


@dataclass
class WeatherRecord:
    lat: float
    lon: float
    elevation: float
    time: pd.DatetimeIndex  # hourly time series
    temperature: np.ndarray  # shape (n_hours,)
    wind_speed: np.ndarray  # shape (n_hours,)
    wind_direction: np.ndarray  # shape (n_hours,)
    incidence: Optional[np.ndarray] = None  # shape (n_hours,)


@dataclass
class Span:
    lat: float
    lon: float
    azimuth: float
    lola_start: Sequence[float]  # [lon, lat] or similar - keep original shape
    lola_end: Sequence[float]
    conductor: ConductorConstants


@dataclass
class RatingsSpan:
    span: Span
    weather_record: WeatherRecord
    ratings: Dict[str, np.ndarray] = field(default_factory=dict)


def linear_resistance_factory(temperature_points: Sequence[float], resistance_points: Sequence[float]):
    """Return a function resist(T) that linearly interpolates between given points."""
    t0, t1 = temperature_points
    r0, r1 = resistance_points
    slope = (r1 - r0) / (t1 - t0)

    def _resist(T: float) -> float:
        return r0 + slope * (T - t0)

    return _resist


def read_spans_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Spans file not found: {file_path!s}")
    df = pd.read_csv(file_path)
    required_cols = {"mid_la", "mid_lo", "azimuth", "start_lo", "start_la", "end_lo", "end_la"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in spans CSV: {missing}")
    return df


def get_weather_data_for_points(points: Sequence[tuple[float, float]], sample_step: int = 1) -> List[Dict[str, Any]]:
    """Query open-meteo for each (lat, lon). Uses simple retry logic and reuses a session.
    Returns list of parsed JSON responses (one per unique queried point).
    """
    session = requests.Session()
    outs: List[Dict[str, Any]] = []
    for idx, (lat, lon) in enumerate(points[::sample_step]):
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&hourly=temperature_80m,wind_speed_80m,wind_direction_80m&timezone=UTC"
        )
        for attempt in range(1, HTTP_RETRIES + 1):
            try:
                resp = session.get(url, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                # basic validation
                if "hourly" not in data or not data["hourly"].get("time"):
                    raise ValueError(f"Unexpected weather response for {lat},{lon}")
                outs.append(data)
                logger.debug("Fetched weather for %s,%s", lat, lon)
                break
            except Exception as exc:
                logger.warning("Failed to fetch weather for %s,%s (attempt %d/%d): %s", lat, lon, attempt, HTTP_RETRIES, exc)
                if attempt == HTTP_RETRIES:
                    raise
                time.sleep(HTTP_BACKOFF * attempt)
    return outs


def run() -> None:
    now_utc = pd.Timestamp.utcnow().floor("h")
    logger.info("Starting DLr run at %s", now_utc.isoformat())

    # conductor: ACSR Drake (kept same numbers as before, but fixed material order)
    conductor_constants = ConductorConstants(
        stranded=True,
        high_rs=True,
        diameter=0.0281,
        cross_section=None,
        absorptivity=0.6,
        emissivity=0.6,
        materials_heat=[
            HeatMaterial(name="aluminum", mass_per_unit_length=1.116, specific_heat_20deg=900, beta=0.0),
            HeatMaterial(name="steel", mass_per_unit_length=0.5126, specific_heat_20deg=500.4, beta=0.0),
        ],
        resistance=linear_resistance_factory(temperature_points=[25.0, 75.0], resistance_points=[7.27e-05, 8.637e-05]),
    )

    # read spans
    df_spans = read_spans_data(SPANS_FILE)

    # build Span objects
    spans: List[Span] = []
    for i, row in df_spans.iterrows():
        spans.append(
            Span(
                lat=float(row["mid_la"]),
                lon=float(row["mid_lo"]),
                azimuth=float(row["azimuth"]),
                lola_start=[row["start_lo"], row["start_la"]],
                lola_end=[row["end_lo"], row["end_la"]],
                conductor=conductor_constants,
            )
        )
    logger.info("Loaded %d spans", len(spans))

    # sample weather points (original code queried every 10 points)
    # ensure we query same ordering as the original code: sample every WEATHER_SAMPLE_STEP
    points = [(float(r["mid_la"]), float(r["mid_lo"])) for _, r in df_spans.iterrows()]

    # fetch weather for sampled points
    logger.info("Fetching weather for %d sampled points (step=%d)", len(points) // WEATHER_SAMPLE_STEP + 1, WEATHER_SAMPLE_STEP)
    weather_jsons = get_weather_data_for_points(points, sample_step=WEATHER_SAMPLE_STEP)

    # Convert weather responses to arrays (assumes all responses share same hours list)
    # create arrays: locations x hours
    # build arrays safely (handle small count and ensure shapes match)
    if len(weather_jsons) == 0:
        raise RuntimeError("No weather data fetched")

    # extract coordinate pairs for each weather result
    weather_coords = np.array([[w["latitude"], w["longitude"]] for w in weather_jsons], dtype=float)

    # times (pandas UTC timestamps)
    time_index = pd.DatetimeIndex([pd.Timestamp(t, tz="UTC") for t in weather_jsons[0]["hourly"]["time"]])
    # convert arrays
    temperature_80m = np.vstack([w["hourly"]["temperature_80m"] for w in weather_jsons])
    wind_speed_80m = np.vstack([w["hourly"]["wind_speed_80m"] for w in weather_jsons])
    wind_direction_80m = np.vstack([w["hourly"]["wind_direction_80m"] for w in weather_jsons])

    # wind speed conversion from km/h to m/s
    wind_speed_80m = wind_speed_80m * 1000.0 / 3600.0

    # restrict to times >= now_utc
    keep_mask = time_index >= now_utc
    if not keep_mask.any():
        logger.warning("No forecast times at or after now (%s). Using full forecast range.", now_utc)
        keep_mask = np.ones(len(time_index), dtype=bool)

    time_index = time_index[keep_mask]
    temperature_80m = temperature_80m[:, keep_mask]
    wind_speed_80m = wind_speed_80m[:, keep_mask]
    wind_direction_80m = wind_direction_80m[:, keep_mask]

    # Now associate weather points to spans and compute ratings
    ratings_spans: List[RatingsSpan] = []
    # Precompute weather coords as columns (lat, lon)
    weather_lats = weather_coords[:, 0]
    weather_lons = weather_coords[:, 1]

    for span in spans:
        # compute euclidean distance in lat/lon space (approx — same as original)
        dlat = span.lat - weather_lats
        dlon = span.lon - weather_lons
        dist = np.sqrt(dlat ** 2 + dlon ** 2)
        idx = int(np.argmin(dist))

        wr = WeatherRecord(
            lat=float(span.lat),
            lon=float(span.lon),
            elevation=80.0,
            time=time_index,
            temperature=temperature_80m[idx, :].astype(float),
            wind_speed=wind_speed_80m[idx, :].astype(float),
            wind_direction=wind_direction_80m[idx, :].astype(float),
        )

        wr.incidence = np.mod(wr.wind_direction - span.azimuth, 360.0)

        n_hours = wr.temperature.shape[0]
        # preallocate rating arrays
        nor = np.empty(n_hours, dtype=float)
        lte = np.empty(n_hours, dtype=float)
        ste = np.empty(n_hours, dtype=float)

        # compute ratings per hour (retain original sequential calls)
        for i in range(n_hours):
            T = float(wr.temperature[i])
            ws = float(wr.wind_speed[i])
            inc = float(wr.incidence[i])
            # Note: thermal_rating_steady_state returns a sequence; keep [0] as original code did
            nor[i] = float(thermal_rating_steady_state(T, ws, inc, 1000, span.conductor, RATINGS_TEMPS["nor"])[0])
            lte[i] = float(thermal_rating_steady_state(T, ws, inc, 1000, span.conductor, RATINGS_TEMPS["lte"])[0])
            ste[i] = float(thermal_rating_steady_state(T, ws, inc, 1000, span.conductor, RATINGS_TEMPS["ste"])[0])

        rs = RatingsSpan(span=span, weather_record=wr, ratings={"nor": nor, "lte": lte, "ste": ste})
        ratings_spans.append(rs)

    # Convert lists to arrays (spans x hours)
    def stack_rating(key: str) -> np.ndarray:
        return np.vstack([rs.ratings[key] for rs in ratings_spans]) if ratings_spans else np.empty((0, 0))

    nor_arr = stack_rating("nor")
    lte_arr = stack_rating("lte")
    ste_arr = stack_rating("ste")

    wind_speed_arr = np.vstack([rs.weather_record.wind_speed for rs in ratings_spans]) if ratings_spans else np.empty((0, 0))
    wind_direction_arr = np.vstack([rs.weather_record.wind_direction for rs in ratings_spans]) if ratings_spans else np.empty((0, 0))
    temperature_arr = np.vstack([rs.weather_record.temperature for rs in ratings_spans]) if ratings_spans else np.empty((0, 0))

    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    npz_file = FORECAST_DIR / f"{now_utc.strftime('%Y_%m_%d_%H')}_allspans.npz"
    np.savez(
        npz_file,
        time=time_index,
        nor=nor_arr,
        lte=lte_arr,
        ste=ste_arr,
        temperature=temperature_arr,
        wind_speed=wind_speed_arr,
        wind_direction=wind_direction_arr,
    )
    logger.info("Saved full spans NPZ to %s", npz_file)

    # summarize per-hour minimum rating across spans and index-of-min (mle_idx)
    rating_types = list(RATINGS_TEMPS.keys())
    ratings_line = pd.DataFrame({"time": time_index})

    for rt in rating_types:
        r = np.vstack([rs.ratings[rt] for rs in ratings_spans])  # shape: spans x hours
        # mle index (span index having minimum rating) and minimum rating (value)
        mle_idx = np.nanargmin(r, axis=0)
        mle_rating = np.nanmin(r, axis=0)

        ratings_line[rt] = mle_rating
        ratings_line[f"{rt}_mle"] = mle_idx

    csv_file = FORECAST_DIR / f"{now_utc.strftime('%Y_%m_%d_%H')}.csv"
    ratings_line.to_csv(csv_file, index=False, float_format="%.1f")
    logger.info("Saved line summary CSV to %s", csv_file)


if __name__ == "__main__":
    run()
