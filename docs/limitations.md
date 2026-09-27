# Provenance and implementation limitations

This reference describes the scope and reuse limits of [map_based_navigation](../README.md), based on source inspection and the checks listed below. It separates the earlier Python simulation from the later journal system.

## Provenance

- Source baseline: [af79bf2e42d84f98d875d0cc8b6dcc9af0f02b7a](https://github.com/rhymesg/map_based_navigation/tree/af79bf2e42d84f98d875d0cc8b6dcc9af0f02b7a).
- The existing README and source headers associate this author-owned repository with the research note and describe it as a foundation for the journal paper; both are in the [canonical citation section](../README.md#citation).
- `generate_database_1` embeds building centers; this checkout supplies no underlying aerial image, geographic coordinates, or dataset provenance record for those points.
- The [MIT license](../LICENSE) and source copyright notices are retained; the locally supplied journal PDF is not distributed with these changes.
- No release tags or software DOI were present in the inspected checkout; publication DOIs identify the papers, not a version of this software.

## Matcher limitations

| Source location in [main.py](../main.py) | Observed behavior and consequence |
|---|---|
| `object_idx_pairs` initialization | A single combinations iterator is created outside the image-pair loop; the first eligible image pair exhausts it, leaving later image pairs without map candidates |
| Map-pair enumeration | Uses combinations rather than the publications' permutations; reversed assignments are absent |
| `del_Theta` calculation | References image `theta1` instead of the map pair's reference angle; rotated configurations are not handled as specified by the published relative-angle geometry |
| `object_indices_set` in the remaining-point loop | Recreated for each image point; different image objects may count the same map object as a match |
| Object matching | Does not check object labels or classes |
| Best-candidate update | Requires both nondecreasing match count and decreasing residual standard deviation; does not return the journal's weighted average |
| `get_intersections` and radius loop | Some degeneracies return `None` and trigger an assertion; near-tangent rounding can produce a square-root domain error; duplicate points and arbitrary geometry are not robustly handled |
| Additional-point angle comparison | Subtracts individually wrapped angles without wrapping the resulting difference; the branch cut can reject nearby directions |

Fewer than `min_num_obj` image points produce an early invalid result with null coordinates and zero matches. Otherwise, `valid` is only a heuristic acceptance flag: it does not establish uniqueness, calibrated uncertainty, or correct localization.

## Monte Carlo statistics

- The error expression is `np.linalg.norm(np.array(res['x'] - x_true, res['y'] - y_true))`.
- The second positional argument to [`np.array`](https://numpy.org/doc/2.3/reference/generated/numpy.array.html) is a dtype, not another coordinate; in the verified run, NumPy accepts the NumPy scalar there as a dtype and computes only the absolute x error. With ordinary Python floats, the same expression raises `TypeError`.
- Even after repairing that expression, the routine's intended reported error standard deviation excludes invalid matches and errors above its false-positive threshold; it is not an all-sample RMSE.
- The original routine has no seed, and the default footprint leaves no variation in the sampled y position apart from floating-point roundoff.

These behaviors are documented without changing the existing algorithm or selecting new scientific assumptions.

## Research reproducibility

- The [algorithm reference](pattern-matching.md) maps the included geometry to the publications and identifies the missing journal weighting, detection, and filtering components.
- No training data, model weights, flight logs, RTK ground truth, ROS 2 nodes, or paper-figure reproduction scripts are included.
- The [journal paper](https://doi.org/10.3390/drones8080375), Data Availability Statement, says its supporting raw data can be requested from the authors; those data are not bundled here and were not obtained for this check.
- Neither publication's performance results have been reproduced by this documentation update.
- The [small example](simulation.md#small-example) exercises a noiseless, axis-aligned case; it does not validate rotation invariance, noise robustness, or real-world navigation.

## Verification status

Checked on macOS arm64 with Python 3.12.14 and the [pinned dependencies](../requirements.txt):

- The headless example completed with the [recorded output](simulation.md#small-example).
- The original 500-sample Monte Carlo routine completed with NumPy seed zero; its reported error statistics omit y and are not evidence of two-dimensional accuracy.
- An isolated check of `np.array(np.float64(3), np.float64(4))` produced a scalar with norm 3, confirming the metric defect; ordinary Python floats raised `TypeError`.
- Python compilation, local Markdown links/anchors, and whitespace checks passed.
- [CITATION.cff](../CITATION.cff) validated against the official [CFF 1.2.0 schema](https://github.com/citation-file-format/citation-file-format/blob/1.2.0/schema.json).
- Syntax-tree comparison confirmed that `main.py` and `image.py` retain their original executable code; only their provenance headers changed.
- The journal PDF remains in ignored `ref/`; no tracked files depend on it.

Broader Python/platform compatibility and interactive plotting remain unverified.
