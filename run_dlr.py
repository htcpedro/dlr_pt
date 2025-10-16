import pandas as pd
import numpy as np
import requests
from cigre601 import thermal_rating_steady_state, thermal_rating_unsteady_state
from collections import namedtuple
from functools import partial

SPANS_FILE = './data/spans.csv'

RATINGS_TEMPS = {'nor': 50, 'lte': 75, 'ste': 85}

ConductorConstants = namedtuple(
    "ConductorConstants",
    [
        "stranded",
        "high_rs",
        "diameter",
        "cross_section",
        "absortivity",
        "emmisivity",
        "materials_heat",
        "resistance",
    ],
)

HeatMaterial = namedtuple(
    "HeatMaterial", ["name", "mass_per_unit_length", "specific_heat_20deg", "beta"]
)


def linear_resistance(conductor_temperature, temperature, resistence):
    # helper linear interpolation function
    per_1 = (resistence[1] - resistence[0]) / (temperature[1] - temperature[0])
    resistance = resistence[0] + (conductor_temperature - temperature[0]) * per_1
    return resistance


class WeatherRecord:
    def __init__(self, lat, lon, elevation, time, temperature, wind_speed, wind_direction):
        self.lat = lat
        self.lon = lon
        self.elevation = elevation
        self.time = time
        self.temperature = temperature
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.incidence = None


class Span:
    def __init__(self, lat, lon, azimuth, lola_start, lola_end, conductor):
        self.lat = lat
        self.lon = lon
        self.lola_start = lola_start
        self.lola_end = lola_end
        self.azimuth = azimuth
        self.conductor = conductor


class RatingsSpan:
    def __init__(self, span: Span, weather_record: WeatherRecord):
        self.span = span
        self.weather_record = weather_record
        self.ratings = dict()


def read_spans_data(file_name):
    return pd.read_csv(file_name).to_dict(orient='list')


def get_weather_data(spans):
    weather_data = []
    for lon, lat in zip(spans['mid_lo'][::10], spans['mid_la'][::10]):
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&"
               f"hourly=temperature_80m,wind_speed_80m,wind_direction_80m")
        weather_data.append(requests.get(url).json())

    return weather_data


def run_dlr():
    now_utc = pd.Timestamp.utcnow().floor('h')

    # defines conductor: ACSR Drake
    conductor_constants = ConductorConstants(
        stranded=True, high_rs=True,
        diameter=0.0281,
        cross_section=None,
        absortivity=0.6,
        emmisivity=0.6,
        materials_heat=[HeatMaterial(name='aluminum', mass_per_unit_length=1.116, beta=0.0, specific_heat_20deg=900),
                        HeatMaterial(name='steel', mass_per_unit_length=0.5126, beta=0.0, specific_heat_20deg=500.4)],
        resistance=partial(linear_resistance, temperature=[25, 75], resistence=[7.27e-05, 8.637e-05]),
    )

    ##
    spans_dict = read_spans_data(SPANS_FILE)

    spans = []
    for i, _ in enumerate(spans_dict['mid_lo']):
        spans.append(Span(spans_dict['mid_la'][i], spans_dict['mid_lo'][i], spans_dict['azimuth'][i],
                          [spans_dict['start_lo'][i], spans_dict['start_la'][i]],
                          [spans_dict['end_lo'][i], spans_dict['end_la'][i]], conductor_constants))

    ##
    weather_data = get_weather_data(spans_dict)

    weather_data_lalo = np.array([[x['latitude'], x['longitude'], ] for x in weather_data])
    weather_data_time = np.array([pd.Timestamp(x, tz='UTC') for x in weather_data[0]['hourly']['time']])
    temperature_80m = np.array([x['hourly']['temperature_80m'] for x in weather_data])
    wind_speed_80m = np.array([x['hourly']['wind_speed_80m'] for x in weather_data]) * 1000 / 3600
    wind_direction_80m = np.array([x['hourly']['wind_direction_80m'] for x in weather_data])

    ##
    data_to_keep = weather_data_time >= now_utc
    weather_data_time = weather_data_time[data_to_keep]
    temperature_80m = temperature_80m[:, data_to_keep]
    wind_speed_80m = wind_speed_80m[:, data_to_keep]
    wind_direction_80m = wind_direction_80m[:, data_to_keep]

    ##
    # maps weather data to spans and computes ratings
    ratings_span = []
    for span in spans:
        dist = np.sqrt((span.lat - weather_data_lalo[:, 0]) ** 2 + (span.lon - weather_data_lalo[:, 1]) ** 2)
        idx = np.argmin(dist)

        # lat, lon, elevation, time, temperature, wind_speed, wind_direction):
        wr = WeatherRecord(spans[0].lat, spans[0].lon, 80.0, weather_data_time, temperature_80m[idx, :],
                           wind_speed_80m[idx, :], wind_direction_80m[idx, :])
        wr.incidence = np.mod(wind_direction_80m[idx, :] - span.azimuth, 360)

        # ratings
        nor, lte, ste = [], [], []
        for i, _ in enumerate(wr.temperature):
            nor.append(thermal_rating_steady_state(wr.temperature[i], wr.wind_speed[i], wr.incidence[i], 1000,
                                                   span.conductor, RATINGS_TEMPS['nor'])[0])
            lte.append(thermal_rating_steady_state(wr.temperature[i], wr.wind_speed[i], wr.incidence[i], 1000,
                                                   span.conductor, RATINGS_TEMPS['lte'])[0])
            ste.append(thermal_rating_steady_state(wr.temperature[i], wr.wind_speed[i], wr.incidence[i], 1000,
                                                   span.conductor, RATINGS_TEMPS['ste'])[0])

        nor = np.array(nor)
        lte = np.array(lte)
        ste = np.array(ste)

        rs = RatingsSpan(span, wr)
        rs.ratings = {'nor': nor, 'lte': lte, 'ste': ste}
        ratings_span.append(rs)

    ## prepares data output
    ratings_line = pd.DataFrame()
    ratings_line['time'] = wr.time

    # ratings for the whole lines
    rating_type = RATINGS_TEMPS.keys()
    for rt in rating_type:
        r = np.array([x.ratings[rt] for x in ratings_span])
        mle_idx = np.nanargmin(r, axis=0)
        mle_rating = np.nanmin(r, axis=0)
        col_name_rating = f'{rt}'
        col_name_mle = f'{rt}_mle'
        ratings_line[col_name_rating] = mle_rating
        ratings_line[col_name_mle] = mle_idx

    ## writes results
    file_name = f'./forecasts/{now_utc.strftime("%Y_%m_%d_%H")}.csv'
    ratings_line.to_csv(file_name, index=False, float_format='%.1f')


if __name__ == '__main__':
    run_dlr()
