# Evaluation of run 'coarse_small_v3' (checkpoint.pt)

## still (C1, C5): FAIL
- max_water_speed_m_s: 2.417
- max_volume_error: 0.02177
- mean_hydrostatic_pressure_error: 0.1347
- fastest_water_where: next to a wall
- fastest_water_depth_under_surface_m: 0.07819
- fastest_water_distance_to_wall_m: 0.0125

## slosh (C2): FAIL
- period_s: nan
- theory_period_s: 1.398
- relative_error: nan

## sink (C3): PASS
- max_sinking_speed_m_s: 1.131
- theory_terminal_speed_m_s: 1.712
- reached_floor: True

## float (C4): FAIL
- fraction_under_water: 0.3098
- theory_fraction: 0.5009

## roll (direction): PASS
- start_x_m: 0.25
- end_x_m: -0.465

## splash (stone): FAIL
- peak_water_speed_m_s: 2.87
- water_speed_last_second_m_s: 2.15
- max_volume_error: 0.02419
