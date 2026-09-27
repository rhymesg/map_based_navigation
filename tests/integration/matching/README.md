# Matching regression checks

From the repository root after installing `requirements.txt`:

```bash
MPLBACKEND=Agg python -m unittest discover -s tests/integration/matching -v
```

Eight asymmetric synthetic landmarks check localization under rotation, shuffled and reversed detection order, repeated detections, and duplicate map landmarks. Circle fixtures check tangent and disjoint pairs. Expected positions follow the constructed geometry, with tolerance 0.001 map units.
