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
| Object matching | Does not check object labels or classes |
| Best-candidate update | Requires both nondecreasing match count and decreasing residual standard deviation; does not return the journal's weighted average |
| `get_intersections` and radius loop | Some degeneracies return `None` and trigger an assertion; near-tangent rounding can produce a square-root domain error; duplicate points and arbitrary geometry are not robustly handled |

Fewer than `min_num_obj` image points produce an early invalid result with null coordinates and zero matches. Otherwise, `valid` is only a heuristic acceptance flag: it does not establish uniqueness, calibrated uncertainty, or correct localization.

## Monte Carlo statistics

- The routine's reported error standard deviation excludes invalid matches and errors above its false-positive threshold; it is not an all-sample RMSE.
- The original routine has no seed, and the default footprint leaves no variation in the sampled y position apart from floating-point roundoff.

The matcher now enumerates fresh ordered map pairs for every image pair, uses the map reference angle, wraps angular differences, and prevents reusing a map point within a hypothesis. Monte Carlo position errors now use both coordinates via `math.hypot`. Remaining search and scoring choices above are unchanged.

## Research reproducibility

- The [algorithm reference](pattern-matching.md) maps the included geometry to the publications and identifies the missing journal weighting, detection, and filtering components.
- The full source code for the journal system cannot be provided. Use the [paper and method guide](pattern-matching.md#scene-information-and-reference-database) to understand the approach and develop an independent implementation.
- No training data, model weights, flight logs, RTK ground truth, ROS 2 nodes, or paper-figure reproduction scripts are included.
- The [journal paper](https://doi.org/10.3390/drones8080375), Data Availability Statement, says its supporting raw data can be requested from the authors; those data are not bundled here and were not obtained for this check.
- Neither publication's performance results have been reproduced by this documentation update.
- The [small example](simulation.md#small-example) exercises a noiseless, axis-aligned case. [Regression checks](../tests/integration/matching/README.md) additionally cover rotation and observation order on eight synthetic landmarks; noise robustness and real-world accuracy remain unverified.

## Verification status

Checked on macOS arm64 with Python 3.12.14 and the [pinned dependencies](../requirements.txt):

- The headless example completed with the [recorded output](simulation.md#small-example).
- The original 500-sample Monte Carlo routine completed with NumPy seed zero; this historical run omitted y and is not evidence for the corrected metric.
- An isolated check of `np.array(np.float64(3), np.float64(4))` produced a scalar with norm 3, confirming the metric defect; ordinary Python floats raised `TypeError`.
- Python compilation, local Markdown links/anchors, and whitespace checks passed.
- [CITATION.cff](../CITATION.cff) validated against the official [CFF 1.2.0 schema](https://github.com/citation-file-format/citation-file-format/blob/1.2.0/schema.json).
- The journal PDF remains in ignored `ref/`; no tracked files depend on it.

Broader Python/platform compatibility and interactive plotting remain unverified.
