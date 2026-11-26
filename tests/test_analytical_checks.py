"""
Tests of Section 6: the analytical answers themselves (so that the evaluation compares against correct numbers).
"""

from config import Scales                                           # reference scales
from physics_neural_solver import physical_equations as eq          # the checks

SCALES = Scales()


def test_sloshing_period_of_the_box():
    period = eq.check_sloshing_period(SCALES.box.length / SCALES.length, 0.25 / SCALES.length) * SCALES.time
    assert abs(period - 1.3976) < 1e-3                              # first mode, 1 m box, 25 cm water: 1.40 s


def test_deeper_water_sloshes_faster():
    shallow = eq.check_sloshing_period(4.0, 0.4)                    # 10 cm water
    deep = eq.check_sloshing_period(4.0, 1.4)                       # 35 cm water
    assert deep < shallow


def test_terminal_velocity_of_a_7cm_granite_stone():
    v = eq.check_terminal_velocity(2500 / SCALES.water.density, 0.035 / SCALES.length) * SCALES.speed
    assert abs(v - 1.712) < 0.01                                    # m/s


def test_floating_fraction():
    assert eq.check_floating_fraction(0.5) == 0.5
