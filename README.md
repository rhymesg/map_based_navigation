# map_based_navigation

## Overview

Research guide to aerial map-based navigation using deep-learning scene information and database matching, with an illustrative Python geometry simulation.

The main idea is:

1. **Extract scene information:** use deep learning to identify ground objects and represent them by their labels and center locations.
2. **Prepare the reference database:** apply the same extraction method to georeferenced reference imagery, storing the objects in the same representation with geographic coordinates.
3. **Match the patterns:** compare the observed objects' spatial arrangement with the database to estimate the camera's horizontal location.

Read [“Aerial Map-Based Navigation by Ground Object Pattern Matching,” Drones (2024)](https://doi.org/10.3390/drones8080375) for the method and flight experiments. The [method guide](docs/pattern-matching.md#scene-information-and-reference-database) connects scene extraction, database preparation, and geometric matching to the paper.

The full source code for the journal system cannot be provided. This [repository](https://github.com/rhymesg/map_based_navigation) contains Youngjoo Kim's earlier Python simulation associated with the [2021 research note](#citation), demonstrating the geometric matching stage using point coordinates.

The associated navigation research has a [granted Korean patent](#related-patent).

| Intended use | Starting point |
|---|---|
| Understanding the navigation approach | Read the [journal paper](https://doi.org/10.3390/drones8080375) and [method guide](docs/pattern-matching.md) |
| Developing an independent implementation | Follow the paper's scene representation, database preparation, matching, and filtering procedures |
| Exploring the matching geometry | Run the [small simulation](example.py); see its [scope and limitations](docs/limitations.md#matcher-limitations) |

## Installation

The optional geometry simulation uses Python 3.12 as its verified environment; [requirements.txt](requirements.txt) pins the NumPy and Matplotlib versions used for the example. No external imagery or trained model is required.

Clone the repository:

```bash
git clone https://github.com/rhymesg/map_based_navigation.git
```

Enter the repository root:

```bash
cd map_based_navigation
```

Create an environment:

```bash
python3 -m venv .venv
```

Install dependencies:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

These commands use a POSIX shell; on Windows, use the environment's `Scripts/python.exe` executable.

## Usage

To explore the matching geometry, run the small, noiseless example without opening a figure window:

```bash
MPLBACKEND=Agg .venv/bin/python example.py
```

[example.py](example.py) prints the match count, validity, estimated map position, and Euclidean position error. See the [simulation guide](docs/simulation.md) for expected output, coordinate conventions, parameters, and the optional plot.

The original `main.py` entry point runs a Monte Carlo simulation with [conditional error statistics](docs/limitations.md#monte-carlo-statistics); it is not the quick-start command.

## Development

Run the example above as a smoke check and check Python syntax:

```bash
.venv/bin/python -m compileall -q main.py image.py example.py
```

[Synthetic regression checks](tests/integration/matching/README.md) cover rotation, detection order, and repeated points; they do not validate real-world accuracy. [Verification status](docs/limitations.md#verification-status) records what was checked and what remains unverified.

Report issues through the [issue tracker](https://github.com/rhymesg/map_based_navigation/issues), including the commit, dependency versions, input coordinates, random seed, and traceback or unexpected result. [Repository metadata](docs/repository-metadata.md) contains a proposed GitHub description and topics.

## Algorithms and source

| Capability | Publication location | Source and example |
|---|---|---|
| Point-pattern position hypotheses and matching | Research note §II-B, Algorithm 1; journal §2.2, Algorithms 1–2 | [main.py](main.py): `find_position`, `get_intersections`; [example.py](example.py) |
| Simulated meta images and attitude/pixel noise | Research note §III-A | [image.py](image.py): `Image`, `get_aerial_image`, `generate_database_1`; [simulation guide](docs/simulation.md) |
| Journal weighted candidate estimate, Eqs. (1)–(3) | Journal §2.2.2 | Not implemented; see [paper-to-code mapping](docs/pattern-matching.md) |

The [algorithm reference](docs/pattern-matching.md) explains the geometry, implementation choices, and departures from the publications.

## Citation

For the navigation approach and flight experiments, please cite:

> Youngjoo Kim, Seungho Back, Dongchan Song, and Byung-Yoon Lee. “Aerial Map-Based Navigation by Ground Object Pattern Matching.” *Drones*, 8(8), article 375, 2024. [doi:10.3390/drones8080375](https://doi.org/10.3390/drones8080375).

For the earlier simulation and research note, please cite:

> Youngjoo Kim. “Aerial Map-Based Navigation Using Semantic Segmentation and Pattern Matching.” arXiv:2107.00689, 2021; revised 2022, [version 3](https://arxiv.org/abs/2107.00689v3). [doi:10.48550/arXiv.2107.00689](https://doi.org/10.48550/arXiv.2107.00689).

[CITATION.cff](CITATION.cff) lists the journal paper as the preferred reference and the earlier research note as a related work; citation requests are separate from license obligations.

## Related patent

Related granted Korean patent for the navigation research: [KR102737055B1 — Device and method for determining position of aerial vehicle](https://patents.google.com/patent/KR102737055B1/en).

## License and provenance

The code includes an [MIT license](LICENSE). [Provenance and limitations](docs/limitations.md#provenance) identify the inspected revision and the scope of the supplied point data.
