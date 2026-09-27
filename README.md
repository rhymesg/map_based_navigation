# map_based_navigation

## Overview

Python research simulation of aerial map-based localization using ground-object point patterns: match a simulated aerial view to a two-dimensional map and estimate the camera's horizontal position.

This is Youngjoo Kim's simulation associated with the [2021 research note](#citation), which preceded **“Aerial Map-Based Navigation by Ground Object Pattern Matching,” Drones (2024)**. The [canonical repository](https://github.com/rhymesg/map_based_navigation) contains point generation and geometric matching; the journal's object detector, Kalman filter, ROS 2 system, and flight datasets are not included.

Use it to study meta-image representation, radius ratios, and circle-intersection position hypotheses for vision-based navigation in GNSS-denied settings.

| Intended use | Scope and prerequisites |
|---|---|
| Learning localization geometry | Run the [small example](example.py) and follow the [algorithm reference](docs/pattern-matching.md); its noiseless result does not validate general accuracy |
| Adapting the matcher | Repair and validate the [documented implementation limitations](docs/limitations.md#matcher-limitations) before relying on its output |
| Reproducing the journal experiments | Additional system components and data are required; see [research reproducibility](docs/limitations.md#research-reproducibility) |

## Installation

Python 3.12 is the verified environment; [requirements.txt](requirements.txt) pins the NumPy and Matplotlib versions used for the example. No external imagery, trained model, or local `ref/` material is required.

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

Run the small, noiseless example without opening a figure window:

```bash
MPLBACKEND=Agg .venv/bin/python example.py
```

[example.py](example.py) prints the match count, validity, estimated map position, and Euclidean position error. See the [simulation guide](docs/simulation.md) for expected output, coordinate conventions, parameters, and the optional plot.

The original `main.py` entry point runs a Monte Carlo simulation with a [known error-metric defect](docs/limitations.md#monte-carlo-statistics); it is not the quick-start command.

## Development

Run the example above as a smoke check and check Python syntax:

```bash
.venv/bin/python -m compileall -q main.py image.py example.py
```

No automated scientific validation suite is supplied. [Verification status](docs/limitations.md#verification-status) records what was checked and what remains unverified.

Report issues through the [issue tracker](https://github.com/rhymesg/map_based_navigation/issues), including the commit, dependency versions, input coordinates, random seed, and traceback or unexpected result. [Repository metadata](docs/repository-metadata.md) contains a proposed GitHub description and topics.

## Algorithms and source

| Capability | Publication location | Source and example |
|---|---|---|
| Point-pattern position hypotheses and matching | Research note §II-B, Algorithm 1; journal §2.2, Algorithms 1–2 | [main.py](main.py): `find_position`, `get_intersections`; [example.py](example.py) |
| Simulated meta images and attitude/pixel noise | Research note §III-A | [image.py](image.py): `Image`, `get_aerial_image`, `generate_database_1`; [simulation guide](docs/simulation.md) |
| Journal weighted candidate estimate, Eqs. (1)–(3) | Journal §2.2.2 | Not implemented; see [paper-to-code mapping](docs/pattern-matching.md) |

The [algorithm reference](docs/pattern-matching.md) explains the geometry, implementation choices, and departures from the publications.

## Citation

For this simulation and its original method, please cite:

> Youngjoo Kim. “Aerial Map-Based Navigation Using Semantic Segmentation and Pattern Matching.” arXiv:2107.00689, 2021; revised 2022, [version 3](https://arxiv.org/abs/2107.00689v3). [doi:10.48550/arXiv.2107.00689](https://doi.org/10.48550/arXiv.2107.00689).

For the subsequent journal method and flight experiments, please cite:

> Youngjoo Kim, Seungho Back, Dongchan Song, and Byung-Yoon Lee. “Aerial Map-Based Navigation by Ground Object Pattern Matching.” *Drones*, 8(8), article 375, 2024. [doi:10.3390/drones8080375](https://doi.org/10.3390/drones8080375).

[CITATION.cff](CITATION.cff) provides software metadata, the research note as preferred citation, and the journal paper as a related work; citation requests are separate from license obligations.

## License and provenance

The code includes an [MIT license](LICENSE). [Provenance and limitations](docs/limitations.md#provenance) identify the inspected revision and the scope of the supplied point data.

The prior README links [patent KR102737055B1](https://patents.google.com/patent/KR102737055B1/en) and reports PCT and US filings; the filing statements are retained as provenance and have not been independently verified here. Local research materials in `ref/` are gitignored and are not runtime dependencies.
