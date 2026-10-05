# Legacy

Scripts from an earlier version of this project that are **not used** in the
GRSL letter:

- `02_env/` — ERA5 reanalysis download and per-scene sea-state matching.
  Neither benchmark provides acquisition time or footprint metadata, so
  scenes cannot be matched to an external sea-state product; the letter's
  clutter proxy `e` is a purely image-level statistic (see the letter,
  Sec. III-E). Kept for reference only.
