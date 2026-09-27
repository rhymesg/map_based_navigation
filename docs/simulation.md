# Simulation and input reference

This guide explains how to run and adapt the point-pattern simulation in [map_based_navigation](../README.md). Complete the [installation](../README.md#installation) first and run commands from the repository root.

## Small example

Run the noiseless view used by `test_a_case`, with explicit output and no plot:

```bash
MPLBACKEND=Agg .venv/bin/python example.py
```

The example seeds NumPy with zero and uses the supplied point database. Both noise amplitudes are zero; the seed also fixes the random draws that the generator still consumes.

Observed output with Python 3.12.14 and the pinned dependencies on macOS arm64:

```text
#. matched points: 10/12
min_theta_std: 0.00
Image objects: 12
Valid: True
Estimated position (m): (120.406, 92.201)
Position error (m): 0.006058
```

The displayed `min_theta_std` is the standard deviation of combined matching residuals, not position uncertainty. This is a smoke example, not a reproduction of either publication's reported accuracy; the position error is an observation for this input, not a general tolerance or accuracy guarantee.

## Inputs and coordinates

| Entry point | Contract |
|---|---|
| `Image(size_x, size_y)` | Mutable bounds and an initially empty `objects` list; no raster data |
| `GroundObject(x, y)`, `Building(x, y)` | Point coordinates; `Building` is an empty subclass, and matching does not inspect its type |
| `Image.add_object(obj)` | Asserts inclusive bounds on both coordinates |
| `Image.resize(new_size_x, new_size_y)` | Scales each axis and mutates all points and bounds in place |
| `generate_database_1(size_x, size_y)` | Returns the embedded building points rescaled to the requested bounds |
| `get_aerial_image(...)` | Returns a new `Image` containing copies of visible map objects in pixel coordinates |
| `find_position(database, image)` | Returns `{'valid': bool, 'x': value_or_None, 'y': value_or_None, 'num_matches': int}` and usually prints diagnostics |

- Database positions, camera `x`, `y`, `z`, and horizontal output coordinates use the same length unit; the supplied examples interpret it as meters.
- Image positions and `pixel_error_std` use pixels, with the center at `(size_x/2, size_y/2)` and axes aligned with the database axes.
- `attitude_error_std_rad` is in radians; independent roll/pitch draws shift all objects by `z*tan(error)` before projection.
- `get_fov` returns footprint **half-extents**: `fov_x = z*tan(fov_x_deg*pi/180)`, `fov_y = size_y/size_x*fov_x`; despite its name, the angle acts as a half-angle, without division by two.
- Projection crops to the footprint, scales to pixels, adds independent pixel noise, and excludes points within the configured border margin.
- The matcher receives neither height nor camera field of view; those are simulator inputs only.

Use positive image dimensions and nondegenerate point configurations. Input validation is limited to coordinate assertions; see [failure behavior](limitations.md#matcher-limitations).

## Parameters and randomness

- Change example geometry and noise arguments in [example.py](../example.py); settings for the historical routines remain local to their functions in [main.py](../main.py).
- Matcher thresholds are defined at the start of `find_position`: `min_num_obj`, `tol_theta`, `tol_r`, `tol_r_perc`, `tol_theta_circle`, and `max_iter`.
- `tol_theta` and `tol_theta_circle` are angular tolerances in radians; `tol_r` compares dimensionless radius ratios.
- `tol_r_perc` excludes points near the center relative to image or database width; the two exclusions operate in different coordinate systems.
- Both modules use NumPy's global random state; the original routines do not set a seed.
- `sig_circle` is unused; it does not implement the journal's candidate weighting.

## Optional plot and historical Monte Carlo routine

On a machine with a graphical Matplotlib backend, display the database footprint and the simulated image points:

```bash
.venv/bin/python -c 'from main import test_a_case; test_a_case()'
```

`test_a_case` prints matching diagnostics and shows a two-panel figure; it does not plot the estimated position or return the match result. This graphical workflow was not manually verified.

The historical `run_monte_carlo_simulation` samples positions, applies noise, and attempts to report match counts, false positives, and error standard deviation. Its [error calculation omits the y component](limitations.md#monte-carlo-statistics); repair and validate it before interpreting those statistics as horizontal position errors.
