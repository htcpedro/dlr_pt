import numpy as np
from functools import partial
from scipy import optimize


def dynamic_viscosity(ambient_temperature, conductor_temperature):
    """From section 4.5.1, eq 13a, valid for SI units."""

    Tfilm = (conductor_temperature + ambient_temperature) / 2

    return 1.458e-6 * (Tfilm + 273.0) ** 1.5 / (Tfilm + 383.4)


def air_density(ambient_temperature, conductor_temperature, elevation):
    """From section 4.5.2, eq 14a, valid for SI units."""

    Tfilm = (conductor_temperature + ambient_temperature) / 2

    return (1.293 - 1.525e-4 * elevation + 6.379e-9 * elevation ** 2) / (
            1 + 0.00367 * Tfilm
    )


def thermal_conductivity_of_air(ambient_temperature, conductor_temperature):
    """Section 4.5.3, eq 15a, valid for SI units."""
    Tfilm = (conductor_temperature + ambient_temperature) / 2

    return 2.424e-2 + 7.477e-5 * Tfilm - 4.407e-9 * Tfilm ** 2


def reynolds_number(
        ambient_temperature,
        wind_speed,
        conductor,
        conductor_temperature,
        elevation,
):
    """Section 4.4.3, eq 2c, page 10."""
    return (
            conductor.diameter
            * air_density(ambient_temperature, conductor_temperature, elevation)
            * wind_speed
            / dynamic_viscosity(ambient_temperature, conductor_temperature)
    )


def forced_convection(
        ambient_temperature,
        wind_speed,
        angle_of_attack,
        conductor,
        conductor_temperature,
        elevation,
        return_parts=False,
):
    """Section 4.4.3.1, page 11."""
    Kangle = (
            1.194
            - np.cos(angle_of_attack)
            + 0.194 * np.cos(2 * angle_of_attack)
            + 0.368 * np.sin(2 * angle_of_attack)
    )

    Nre = reynolds_number(
        ambient_temperature, wind_speed, conductor, conductor_temperature, elevation
    )

    kf = thermal_conductivity_of_air(ambient_temperature, conductor_temperature)

    qc1 = (
            Kangle
            * (1.01 + 1.35 * Nre ** 0.52)
            * kf
            * (conductor_temperature - ambient_temperature)
    )

    qc2 = (
            Kangle * 0.754 * Nre ** 0.6 * kf * (conductor_temperature - ambient_temperature)
    )

    if return_parts:
        return np.maximum(qc1, qc2), qc1, qc2, Kangle

    return np.maximum(qc1, qc2)


def natural_convection(
        ambient_temperature,
        conductor,
        conductor_temperature,
        elevation,
):
    """Section 4.4.3.2, eq 5a 5b, page 12"""
    return (
            3.645
            * air_density(ambient_temperature, conductor_temperature, elevation) ** 0.5
            * conductor.diameter ** 0.75
            * (conductor_temperature - ambient_temperature) ** 1.25
    )


def convective_heat_loss(
        ambient_temperature,
        wind_speed,
        angle_of_attack,
        conductor,
        conductor_temperature,
        elevation,
):
    """The convective heat loss is the highest of forced and natural convection

    From section 4.4.3 in the standard, page 10.
    """

    forced = forced_convection(
        ambient_temperature,
        wind_speed,
        angle_of_attack,
        conductor,
        conductor_temperature,
        elevation,
    )

    natural = natural_convection(
        ambient_temperature, conductor, conductor_temperature, elevation
    )

    qc = np.maximum(forced, natural)

    return qc


def radiated_heat_loss(
        ambient_temperature,
        conductor,
        conductor_temperature,
):
    """Section 4.4.4, eq 7a 7b, page 12"""
    return (
            17.8
            * conductor.diameter
            * conductor.emissivity
            * (
                    ((conductor_temperature + 273) / 100) ** 4
                    - ((ambient_temperature + 273) / 100) ** 4
            )
    )


def solar_heat_gain(solar_irradiation, conductor):
    return conductor.absorptivity * solar_irradiation * conductor.diameter


def thermal_rating_steady_state(ambient_temperature, wind_speed, angle_of_attack, solar_irradiation, conductor,
                                conductor_temperature, horizontal_angle=0, elevation=500):
    """Calculate the rating using IEEE738.
    ambient_temperature:  temperature of air in [°C]
    wind_speed:           in [m/s]
    angle_of_attack:      the angle between the wind and the conductor in [°]. 0° is parallel wind, 90° is perpendicular
    solar_irradiation:    in [W/m^2]
    conductor:            the conductor structure with details about the material. from conductor
    conductor_temperature: the target conductor temperature [°C]
    horizontal_angle:     not used in IEEE738
    elevation:            the see level elevation in [m]
    """

    # the angle must be in the range 0-90
    angle_of_attack = np.deg2rad(90 - np.abs((angle_of_attack % 180) - 90))

    qc = convective_heat_loss(ambient_temperature, wind_speed, angle_of_attack, conductor, conductor_temperature,
                              elevation)
    qr = radiated_heat_loss(ambient_temperature, conductor, conductor_temperature)
    qs = solar_heat_gain(solar_irradiation, conductor)

    current = np.sqrt((qc + qr - qs) / conductor.resistance(conductor_temperature))

    return current, qs, qr, qc


def residual_max_current(current_guess, temp_goal, temp_initial, conductor, ambient_temp, wind_speed, wind_incidence,
                         solar_radiation, elevation, time_interval=15 * 60):
    t_guess = unsteady_conductor_temp(current_guess, temp_initial, ambient_temp, wind_speed, wind_incidence,
                                      solar_radiation, conductor, elevation, time_interval)
    return t_guess - temp_goal


def thermal_rating_unsteady_state(temp_goal, temp_initial, temp_ambient, time_interval, wind_speed, angle_of_attack,
                                  solar_radiation, conductor, elevation=500):
    max_allowed_temp = 400
    f = partial(residual_max_current, temp_goal=temp_goal, temp_initial=temp_initial, conductor=conductor,
                ambient_temp=temp_ambient, wind_speed=wind_speed, wind_incidence=angle_of_attack,
                solar_radiation=solar_radiation, elevation=elevation, time_interval=time_interval)
    current_lb = thermal_rating_steady_state(temp_ambient, wind_speed, angle_of_attack, solar_radiation, conductor,
                                             temp_initial, 0, elevation)[0]
    current_ub = thermal_rating_steady_state(temp_ambient, wind_speed, angle_of_attack, solar_radiation, conductor,
                                             max_allowed_temp, 0, elevation)[0]
    try:
        current_ste = optimize.bisect(f, current_lb, current_ub)
    except ValueError:
        current_ste = np.nan

    return current_ste


def unsteady_conductor_temp(current_final, temp_initial, ambient_temp, wind_speed, wind_incidence,
                            solar_radiation, conductor, elevation, max_t=60 * 15, delta_t=10):
    n_steps = int(np.round(max_t / delta_t))
    mCp = 0
    for m in conductor.materials_heat:
        mCp += m.mass_per_unit_length * m.specific_heat_20deg * (1 + m.beta * (temp_initial - 20))

    conductor_temp = temp_initial
    for i in range(n_steps):
        current, qs, qr, qc = thermal_rating_steady_state(ambient_temp, wind_speed, wind_incidence, solar_radiation,
                                                          conductor, conductor_temp, 0, elevation)
        delta_temp = (conductor.resistance(conductor_temp) * current_final ** 2 + qs - qc - qr) * delta_t / mCp
        conductor_temp += delta_temp
    return conductor_temp
