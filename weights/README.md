# Detector checkpoints

The two retrained YOLOv8s models used for every number in the letter are
distributed as GitHub release assets of this repository (release `weights`):

| File | Dataset | Role |
|---|---|---|
| `best_sardet100k.pt` | SARDet-100K ship subset | detector for all SARDet-100K results |
| `best_sarship.pt` | SAR-Ship-Dataset | detector for all SAR-Ship-Dataset results |

Download and verify:

```bash
# from the repository root
wget -O weights/best_sardet100k.pt <RELEASE_URL>/best_sardet100k.pt
wget -O weights/best_sarship.pt    <RELEASE_URL>/best_sarship.pt
sha256sum -c weights/SHA256SUMS
```

Training recipe (reproduces the checkpoints from scratch): YOLOv8s,
Ultralytics default recipe, 100 epochs, seed 42, single class (ship), one
model per dataset — see `scripts/03_train/` and `configs/default.yaml`.
Ungated test AP@0.5: 0.940 (SARDet-100K) / 0.962 (SAR-Ship-Dataset).

## SHA256SUMS

```
<PASTE_SHA256_best_sardet100k.pt>
<PASTE_SHA256_best_sarship.pt>
```

*The checksum lines above are placeholders; replace them after uploading
the release assets (`sha256sum best_*.pt` on the local files).*
