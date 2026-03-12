# Experiments Naming Conventions

Use these names consistently across experiment variants.

## Grid / Domain
- `nx`, `ny`: grid resolution in x/y.
- `lx`, `ly`: domain lengths in x/y.
- `grid_n`: scalar node count (`nx * ny`).

## State Matrices
- `full_state_matrix`: concatenated `[u_flat | v_flat]` over all timesteps.
- `window_state_matrix`: concatenated `[u_flat | v_flat]` within current window.
- `window_basis_matrix`: basis matrix from window-local SSPOR/POD model.
- `global_basis_matrix`: basis matrix from full-history POD.

## Models / Sensors
- `window_sspor_model`: SSPOR model fit on `window_state_matrix`.
- `global_sspor_model`: SSPOR model fit on `full_state_matrix`.
- `window_qr_nodes`: node indices selected by window SSPOR/QR.
- `static_qr_nodes`: fixed node indices selected once from global SSPOR/QR.

## Sensor Coordinates
- `fixed_sensor_positions`
- `lagrangian_sensor_positions`
- `moving_pod_qr_sensor_positions`
- `window_qr_target_positions`

## Labels
- Methods/placements: `Fixed`, `Lagrangian`, `Moving POD-QR`, `QR teleport`.
- Basis labels: `Window POD`, `Global POD`.
