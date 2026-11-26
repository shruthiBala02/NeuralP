# Evaluation of run 'coarse_small_v2' (checkpoint_end_of_tilt.pt)

## still (C1, C5): FAIL
- max_water_speed_m_s: 0.6977
- max_volume_error: 0.04431
- mean_hydrostatic_pressure_error: 0.06773

## slosh (C2): FAIL
- period_s: nan
- theory_period_s: 1.398
- relative_error: nan

## sink (C3): PASS
- max_sinking_speed_m_s: 1.104
- theory_terminal_speed_m_s: 1.712
- reached_floor: True

## float (C4): PASS
- fraction_under_water: 0.5374
- theory_fraction: 0.5009

## roll (direction): FAIL
- start_x_m: 0.25
- end_x_m: 0.3124
