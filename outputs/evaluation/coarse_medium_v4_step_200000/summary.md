# Evaluation of run 'coarse_medium_v4' (checkpoint_step_200000.pt)

## still (C1, C5): FAIL
- max_water_speed_m_s: 0.8831
- max_volume_error: 5.96e-08
- mean_hydrostatic_pressure_error: 0.01046
- fastest_water_where: next to a wall
- fastest_water_depth_under_surface_m: 0.1676
- fastest_water_distance_to_wall_m: 0.0125

## slosh (C2): FAIL
- period_s: 1.632
- theory_period_s: 1.398
- relative_error: 0.1676

## sink (C3): PASS
- max_sinking_speed_m_s: 1.103
- theory_terminal_speed_m_s: 1.712
- reached_floor: True

## float (C4): PASS
- fraction_under_water: 0.4535
- theory_fraction: 0.5009

## roll (direction): PASS
- start_x_m: 0.25
- end_x_m: -0.465

## splash (stone): FAIL
- peak_water_speed_m_s: 0.7169
- water_speed_last_second_m_s: 0.6787
- max_volume_error: 0.0007501
