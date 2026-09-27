# Implementation and provenance

Reference for the supplied Python point-pattern simulation. Use the [method guide](pattern-matching.md) to connect it to the journal navigation approach and the [citation section](../README.md#citation) to select the relevant publication.

## Provenance

Youngjoo Kim's [earlier source](https://github.com/rhymesg/map_based_navigation/tree/af79bf2e42d84f98d875d0cc8b6dcc9af0f02b7a) accompanies the research note. `generate_database_1` supplies building-center coordinates for the geometry example; the code and author notices use the [MIT license](../LICENSE).

## Matching procedure

The matcher enumerates image-point pairs and ordered map-point pairs. It skips coincident map pairs and radius intervals without intersections, then searches for a center matching the image's angle and radius ratio. Each candidate assigns a map point at most once.

Candidate comparison uses nondecreasing match count and decreasing residual standard deviation. The `valid` flag reports the configured match-count criterion; the journal's weighted candidate estimate is described separately in the [paper method reference](pattern-matching.md).

Supply positive image dimensions and finite coordinates. Fewer than six image detections return an invalid result with null coordinates and zero matches. The [coordinate guide](simulation.md#inputs-and-coordinates) defines map and image units.

## Monte Carlo statistics

The historical experiment reports match counts, false positives, and the standard deviation of Euclidean position errors for accepted matches within its error threshold. Position error uses both horizontal coordinates; interpret the reported standard deviation as a conditional statistic.

## Journal method

The full source was developed as part of company research and cannot be publicly released. The [paper and method guide](pattern-matching.md#scene-information-and-reference-database) describe scene extraction, reference-database preparation, and matching for an independent implementation.

## Checks

The [small example](simulation.md#small-example) reports the estimated position and geometric error. [Regression checks](../tests/integration/matching/README.md) cover rotation, observation order, repeated detections, duplicate map landmarks, and circle intersections.
