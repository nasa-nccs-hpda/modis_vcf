# Training labels (set 7.0.1, modis_vcf v1.5.2)

The labels are delivered separately from the code, as `VCF_training_labels_7.0.1.tar.gz` (83 MB; `.sha256` next to it). The archive
has its own `README.md` and a `MANIFEST.csv` (every file's name, original name, sha256, size and number of labelled pixels).

## Contents

| Path in the archive | Files | What |
|---|---|---|
| `pcttree/VCF_training_pcttree_<tile>.7.0.1.out` | 63 | Percent tree cover labels |
| `pctbare/VCF_training_pctbare_<tile>.7.0.0.out` | 33 | Percent bare (non-vegetated) labels |
| `pctbare/VCF_training_pctbare_h16v02.7.0.1.out` | 1 | 10,000 points on the Greenland ice sheet, all 100 % bare |
| `manifests/manifest_{tree,bare}_2019.parq` (+ `.json`) | 4 | The exact 2019 samples behind our 2019 models |

Format of every label file: raw, no header, 4800 x 4800 uint8, row-major, on the MODIS sinusoidal 250 m tile grid (231.656 m pixels). 0-100 = percent
cover, **255 = no label**. In the code, pixel (x, y) = (row, column):
`numpy.fromfile(f, numpy.uint8).reshape(4800, 4800)[x, y]`.

## The version numbers

- **7.0.0** = the label set in the 2025 `modis_vcf` repository (`data/training`, removed from the repository in v1.5.2). The 33 bare
  files are byte-identical to it. Its two other bare tiles, h10v02 and h21v08, are not used since v1.5.0.
- **7.0.1** = files that differ from 7.0.0 or are new:
  - tree: 7.0.0 holds the full v5.0.3 label rasters; the models train on **v5.0.3samp**, a subset of the same pixels (the rest set
    to 255, every kept value unchanged). Only 2 of the 63 tiles are identical to 7.0.0;
  - bare h16v02: new. The ice points are a regular grid at least 10 km inside the ice-sheet edge (not water), all labelled 100.
    They teach the bare model that the ice sheet interior is fully bare. Until v1.5.1 they were a separate list (a parquet file of
    pixel keys); in the archive they are a raster like every other tile.

## How labels become training rows

See [RUNNING.md](RUNNING.md), steps 1 and 3. Every labelled pixel is a row, rows are sorted by (tile-year, row, column), and a row is
held out for testing when the pandas hash of its key is 0 mod 5 (about 20%). The predictors at each row are read from that year's
metrics, so a new year needs new splits and new models, but the labels do not change.

## Reproducing our models exactly, and the h16v02 row order

- Built from the 2019 manifests (`trainingManifest build --manifest manifests/manifest_tree_2019.parq ...`) on the same 2019 metrics,
  the splits and models are byte-identical to ours.
- Sampled afresh from the rasters (`trainingManifest sample`), the rows and their train/test parts are the same, but the h16v02 ice
  points sit in tile order (between h16v01 and the next tile). In our runs they were appended after all other rows. A Random Forest's
  bootstrap depends on row order, so the **bare model is statistically equivalent but not byte-identical**. This is expected; tests
  comparing splits or models made from the archive should build from the manifests when they need exact equality.
- The bare manifest's `.json` still names the separate ice-point list it was made with.

## Provenance

Tree and bare labels come from Landsat 30 m classifications of the early 2000s, at sites chosen to be stable over time, aggregated to
the 250 m grid. No commercial (NGA) imagery was used for any label in the archive.
