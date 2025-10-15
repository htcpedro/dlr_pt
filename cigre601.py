import numpy as np
from functools import partial
from scipy import optimize


def wind_direction_correction_stranded(angle_of_attack):
    correction_low = 0.42 + 0.68 * np.sin(angle_of_attack / 180 * np.pi) ** 1.08
    correction_high = 0.42 + 0.58 * np.sin(angle_of_attack / 180 * np.pi) ** 0.90

    correction = np.where(angle_of_attack <= 24, correction_low, correction_high)

    return correction


def nusselt_smooth_conductor(re, angle_of_attack):
    conditions = [re < 5000, (5000 <= re) & (re < 50000), (50000 <= re) & (re < 200000)]

    B_options = [0.583, 0.148, 0.0208]
    n_options = [0.471, 0.633, 0.814]

    B = np.select(conditions, B_options)
    n = np.select(conditions, n_options)

    wind_direction_correction = (
                                        np.sin(angle_of_attack / 180 * np.pi) ** 2
                                        + 0.0169 * np.cos(angle_of_attack / 180 * np.pi) ** 2
                                ) ** 0.225

    return B * re ** n * wind_direction_correction


def nusselt_stranded_small_Rs_conductor(re, angle_of_attack):
    conditions = [re < 2650, (2650 <= re) & (re < 50000), (50000 <= re) & (re < 200000)]

    B_options = [0.641, 0.178, 0.0208]
    n_options = [0.471, 0.633, 0.814]

    B = np.select(conditions, B_options)
    n = np.select(conditions, n_options)

    wind_direction_correction = wind_direction_correction_stranded(angle_of_attack)

    return B * re ** n * wind_direction_correction


def nusselt_stranded_high_Rs_conductor(re, angle_of_attack):
    conditions = [re < 2650, (2650 <= re) & (re < 50000), (50000 <= re) & (re < 200000)]

    B_options = [0.641, 0.048, 0.0208]
    n_options = [0.471, 0.800, 0.814]

    B = np.select(conditions, B_options)
    n = np.select(conditions, n_options)

    wind_direction_correction = wind_direction_correction_stranded(angle_of_attack)

    nusselt = B * re ** n * wind_direction_correction

    return nusselt


def get_nusselt_function(conductor=None, stranded=None, high_rs=None):
    """Returns the appropriate function based on if the conductor is stranded, smooth and has high rs.

    Page 25 and 26.
    """

    if conductor:
        stranded = conductor.stranded
        high_rs = conductor.high_rs

    nusselt_functions = {
        (True, False): nusselt_stranded_small_Rs_conductor,
        (True, True): nusselt_stranded_high_Rs_conductor,
        (False, False): nusselt_smooth_conductor,
        (False, True): nusselt_smooth_conductor,
    }

    return nusselt_functions[(stranded, high_rs)]


def horizontal_correction(conductor, angle):
    """Equation 24, page 28"""
    if conductor.stranded:
        return 1 - 1.76e-6 * angle ** 2.5
    else:
        return 1 - 1.58e-4 * angle ** 1.5


def specific_heat(conductor, temperature):
    total_mass = 0.0
    total_heat = 0.0

    for metal in conductor.materials_heat:
        c = metal.specific_heat_20deg
        m = metal.mass_per_unit_length
        beta = metal.beta

        c_modified = c * (1 + beta * (temperature - 20.0))

        total_mass += m
        total_heat += m * c_modified

    return total_heat / total_mass


def temperature_film(conductor_temperature, ambient_temperature):
    return 0.5 * (conductor_temperature + ambient_temperature)


def thermal_conductivity_of_air(t_f):
    """Section 3.5, eq 18, page 24."""
    return 2.368e-2 + 7.23e-5 * t_f - 2.763e-8 * t_f ** 2


def dynamic_viscosity(t_f):
    """Eq 19, page 25"""
    return (17.239 + 4.635e-2 * t_f - 2.03e-5 * t_f ** 2) * 1e-6


def air_density(t_f, elevation):
    """Eq 20, page 25"""
    return (1.293 - 1.525e-4 * elevation + 6.379e-9 * elevation ** 2) / (
            1 + 0.00367 * t_f
    )


def kinematic_viscosity(t_f, elevation):
    return dynamic_viscosity(t_f) / air_density(t_f, elevation)


def reynolds_number(wind_speed, conductor, t_f, elevation):
    """Page 25, in text."""
    return wind_speed * conductor.diameter / kinematic_viscosity(t_f, elevation)


def forced_convection(
        ambient_temperature,
        wind_speed,
        angle_of_attack,
        conductor,
        conductor_temperature,
        elevation,
):
    t_f = temperature_film(conductor_temperature, ambient_temperature)

    # Page 25, in text
    re = reynolds_number(wind_speed, conductor, t_f, elevation)

    nusselt_number = get_nusselt_function(conductor)(re, angle_of_attack)

    # Eq 17, page 24
    forced_convec = (
            np.pi
            * thermal_conductivity_of_air(t_f)
            * (conductor_temperature - ambient_temperature)
            * nusselt_number
    )

    return forced_convec


def grashof(conductor, conductor_temperature, ambient_temperature, t_f, elevation):
    return (
            conductor.diameter ** 3
            * (conductor_temperature - ambient_temperature)
            * 9.807
            / ((t_f + 273) * kinematic_viscosity(t_f, elevation) ** 2)
    )


def prandtl(conductor, conductor_temperature, t_f):
    return (
        # TODO: Verify this assumption
            1005.0  # specific_heat(conductor, conductor_temperature)  # This should be the specific heat of air?,
            # not the metal
            * dynamic_viscosity(t_f)
            / thermal_conductivity_of_air(t_f)
    )


def natural_convection(
        ambient_temperature, conductor, conductor_temperature, horizontal_angle, elevation
):
    t_f = temperature_film(conductor_temperature, ambient_temperature)

    # From List of Symbols on page 7
    gr = grashof(conductor, conductor_temperature, ambient_temperature, t_f, elevation)

    # From List of Symbols on page 7
    pr = prandtl(conductor, conductor_temperature, t_f)

    gp = gr * pr

    condition = [gp < 1e2, gp < 1e4, gp < 1e7, gp < 1e12]

    A_choices = [1.02, 0.850, 0.480, 0.125]
    m_choices = [0.148, 0.188, 0.250, 0.333]

    A = np.select(condition, A_choices)
    m = np.select(condition, m_choices)

    nusselt_number_natural = (
            A * gp ** m * horizontal_correction(conductor, horizontal_angle)
    )

    natural_convec = (
            np.pi
            * thermal_conductivity_of_air(t_f)
            * (conductor_temperature - ambient_temperature)
            * nusselt_number_natural
    )

    return natural_convec


def power_convective(
        ambient_temperature,
        wind_speed,
        angle_of_attack,
        conductor,
        conductor_temperature,
        horizontal_angle,
        elevation,
):
    """Convective cooling is the highest of forced and natural convection.

    Based on text in "Low wind speeds" on page 28.
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
        ambient_temperature,
        conductor,
        conductor_temperature,
        horizontal_angle,
        elevation,
    )

    P = np.maximum(forced, natural)

    return P


def power_radiation(ambient_temperature, conductor, conductor_temperature):
    """Eq 27, page 30"""
    return (
            np.pi
            * conductor.diameter
            * 5.6697e-8  # Stefan-Boltzmann
            * conductor.emmisivity
            * ((conductor_temperature + 273) ** 4 - (ambient_temperature + 273) ** 4)
    )


# def power_joule(self, current):
#    # TODO: Account for Skin Effect
#    return current ** 2 * self.material.resistance(self.temperature)


def power_solar(solar_irradiation, conductor):
    """Section 3.3, Eq 8, page 18."""
    return conductor.absortivity * solar_irradiation * conductor.diameter


def thermal_rating_steady_state(ambient_temperature, wind_speed, angle_of_attack, solar_irradiation, conductor,
                                conductor_temperature, horizontal_angle=0, elevation=500):
    """Calculate the rating using CIGRE-601.
    ambient_temperature:  temperature of air in [°C]
    wind_speed:           in [m/s]
    angle_of_attack:      the angle between the wind and the conductor in [°]. 0° is parallel wind, 90° is perpendicular
    solar_irradiation:    in [W/m^2]
    conductor:            the conductor structure with details about the material. from conductor
    conductor_temperature: the target conductor temperature [°C]
    horizontal_angle:     not used
    elevation:            the see level elevation in [m]
    """

    angle_of_attack = 90 - np.abs((angle_of_attack % 180) - 90)

    Pc = power_convective(ambient_temperature, wind_speed, angle_of_attack, conductor, conductor_temperature,
                          horizontal_angle, elevation)
    Pr = power_radiation(ambient_temperature, conductor, conductor_temperature)
    Ps = power_solar(solar_irradiation, conductor)

    current = np.sqrt((Pr + Pc - Ps) / conductor.resistance(conductor_temperature))

    return current, Pr, Pc, Ps


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
        current, Pr, Pc, Ps = thermal_rating_steady_state(ambient_temp, wind_speed, wind_incidence, solar_radiation,
                                                          conductor, conductor_temp, 0, elevation)
        delta_temp = (conductor.resistance(conductor_temp) * current_final ** 2 + Ps - Pc - Pr) * delta_t / mCp
        conductor_temp += delta_temp
    return conductor_temp
