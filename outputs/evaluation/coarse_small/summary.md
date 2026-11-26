# Evaluation of run 'coarse_small'

## still (C1, C5): FAIL
- max_water_speed_m_s: 2.078
- max_volume_error: 0.8181
- mean_hydrostatic_pressure_error: 0.4387

## slosh (C2): FAIL
- period_s: 4.257
- theory_period_s: 1.398
- relative_error: 2.046

## sink (C3): PASS
- max_sinking_speed_m_s: 1.104
- theory_terminal_speed_m_s: 1.712
- reached_floor: True

## float (C4): FAIL
- fraction_under_water: 1
- theory_fraction: 0.5009

## roll (direction): PASS
- start_x_m: 0.25
- end_x_m: -0.465
