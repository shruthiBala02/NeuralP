# Evaluation of run 'coarse_small_v2' (checkpoint.pt)

## still (C1, C5): FAIL
- max_water_speed_m_s: 5.519
- max_volume_error: 0.5518
- mean_hydrostatic_pressure_error: 0.2101

## slosh (C2): FAIL
- period_s: nan
- theory_period_s: 1.398
- relative_error: nan

## sink (C3): PASS
- max_sinking_speed_m_s: 1.107
- theory_terminal_speed_m_s: 1.712
- reached_floor: True

## float (C4): FAIL
- fraction_under_water: 0.3108
- theory_fraction: 0.5009

## roll (direction): PASS
- start_x_m: 0.25
- end_x_m: -0.465
