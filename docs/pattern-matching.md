# Ground-object pattern matching

This reference maps the aerial localization geometry to [main.py](../main.py) in [map_based_navigation](../README.md). The [research note and journal citations](../README.md#citation) describe the method's origin and subsequent development; this checkout implements the earlier simulation with the departures listed below.

## Problem and representation

- Input: two [Image](../image.py) instances containing object centers, one in map coordinates and one in image pixels.
- Output: the map position corresponding to the image center, plus a match count and validity flag.
- The matcher consumes point coordinates, not raster images, descriptors, or trained network outputs.
- The published geometry assumes a downward-looking view or points projected onto a ground-parallel plane; no attitude rectification is implemented here.

## Geometry and procedure

Both publications describe a random sample consensus (RANSAC)-based approach to testing position hypotheses. This code deterministically enumerates pairs rather than randomly sampling them; its [map-pair iterator limitation](limitations.md#matcher-limitations) further restricts the search.

For image center $c$ and object $p_i$, `cart_to_polar` computes $r_i=\|p_i-c\|$ and $\theta_i=\operatorname{atan2}(p_{iy}-c_y,p_{ix}-c_x)$. A candidate map center must explain the pair's radius ratio $q=r_j/r_i$ and wrapped angular difference.

For map-object separation $d$, the circle radii are $R_i$ and $qR_i$, with bounds

$$
R_{i,\min}=\frac{d}{1+q},\qquad
R_{i,\max}=\frac{d}{|1-q|}\quad(q\ne1).
$$

The journal's Eq. (4) states the circle-intersection condition underlying these bounds. The implementation caps the upper bound at the map width and uses that width directly when the ratio is close to one.

1. `find_position` forms image pairs and converts each pair to polar coordinates about the image center.
2. It searches map pairs and bisects the radius interval; `get_intersections` supplies the two possible map centers.
3. It chooses an intersection by angular sign, then checks angular agreement within `tol_theta_circle`.
4. Remaining points are compared using normalized radius and angle tolerances; the lowest-error eligible map object is counted for each image object.
5. A candidate replaces the current best only when its match count is at least the current best and its error standard deviation is lower.
6. The result is valid only with at least `min_num_obj` matches, strictly more than half the image objects matched, and non-null coordinates.

`find_position` uses a normalized residual, `abs(angle_error)/pi + abs(radius_ratio_error)/image_radius_ratio`, before taking its standard deviation. This differs from the unnormalized sum in both publications' Algorithm 1.

## Paper-to-code mapping

| Publication location | Implementation | Boundary |
|---|---|---|
| Research note §II-B, Algorithm 1 | `find_position` | One best candidate; pair enumeration and angle-reference limitations below |
| Research note §II-C–II-D | No implementation | Proposed velocity estimation and probabilistic data fusion absent |
| Journal §2.2.1–2.2.2, Algorithm 1 | Pair hypotheses and additional-point matching | No label checks, ROI selection, or candidate collection |
| Journal Algorithm 2; §2.2.3, Eq. (4) | Radius loop and `get_intersections` | Width cap, sign selection, iteration limit, and stopping rules differ |
| Journal §2.2.2, Eqs. (1)–(3) | No implementation | Gaussian/match-count weighting and weighted position average absent |
| Journal §2.1, §2.3 and Appendix A | No implementation | Object detector, coordinate preprocessing, and inertial/Kalman fusion absent |

The code uses `theta1 - theta2`, following the research note's sign convention; the journal defines the reversed difference. More significantly, additional map-point comparisons use the image's `theta1` instead of the candidate's map reference angle, so this implementation does not establish the publications' rotation invariance.

## Use and limitations

Run [example.py](../example.py) using the [simulation guide](simulation.md). The example exercises one noiseless, axis-aligned view and the actual matcher.

The single-use map-pair iterator, lack of reversed map pairs, reusable map matches, and geometry failures are detailed in [implementation limitations](limitations.md#matcher-limitations). Treat those as constraints on this checkout rather than properties of the published method.
