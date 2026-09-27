# Matching regression checks

From the repository root after installing `requirements.txt`:

```bash
MPLBACKEND=Agg python -m unittest discover -s tests/integration/matching -v
```

Eight asymmetric synthetic landmarks check noiseless localization under rotation, shuffled and reversed detection order, and repeated detections. Expected positions are set by the constructed geometry, with tolerance 0.001 map units. These checks do not establish noisy or real-world navigation accuracy.
