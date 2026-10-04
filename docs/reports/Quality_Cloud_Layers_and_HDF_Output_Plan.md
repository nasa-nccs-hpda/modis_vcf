# Quality and Cloud layers, and an HDF product file, for the Python VCF (plan)

**Written 2026-10-01. Status: plan only, awaiting approval. Nothing has run apart from the read-only checks below.**

The historical MOD44B granule has 7 SDSs. Once the SD layers are added (`SD_Layer_Plan.md`, `SD_Layer_Results.md`), Python will write 5 of them, still as separate GeoTIFFs. This plan covers the last two layers (`Quality`, `Cloud`) and a single compressed product file laid out like the original. It ends with a recommendation on HDF-EOS2 against HDF5.

## What the historical product holds (checked 2026-10-01, forest203, read-only)

Files: C6 `MOD44B.A2019065.h08v05.006.2020098202855.hdf` (86 MB), C6.1 `MOD44B.A2019065.h08v05.061.2022283140221.hdf` (78 MB), and the C code's 2026-09-19 output `full63_c_vs_py_20260919-0806/C/production/build_context/runtime/b3_output/MOD44B.A2019065.h08v05.061.hdf` (82 MB).

- **Format:** HDF4 with an HDF-EOS2 grid `MOD44B_250m_GRID` (4800 x 4800, sinusoidal), and ECS metadata from the MCF through the SDP toolkit (`MOD_PR44B3/create_output_file.c`, `make_grid.c`). The C code's output is also HDF4 (file signature `0e 03 13 01`). The comment at `create_output_file.c:642` that mentions "plain HDF5" is wrong.
- **Compression:** every SDS uses DEFLATE level 8, in both C6 and the C code (`make_grid.c`: `GDsettilecomp(..., HDFE_COMP_DEFLATE, compparm)`, `compparm[0] = 8`). The tile size is the MOD44C grid size. Ratios on h08v05 for the C code: tree cover 3.9:1, SDs 1.6-2.5:1, `Quality` 8:1, `Cloud` 70:1.

| SDS | Type | C6.1 attributes |
|---|---|---|
| `Percent_Tree_Cover` | uint8 | valid_range 0-100, Water_Value 200, _FillValue 253 |
| `Percent_NonTree_Vegetation` | uint8 | same |
| `Percent_NonVegetated` | uint8 | same |
| `Quality` | uint8 | long_name "Quality Flags", valid_range 0-255, _FillValue 0 |
| `Percent_Tree_Cover_SD` | uint16 | scale_factor 0.01, valid_range 0-275 (wrong, see SD plan), _FillValue 0 |
| `Percent_NonVegetated_SD` | uint16 | same |
| `Cloud` | uint8 | long_name "Cloud Cover in Composites", valid_range 0-255, _FillValue 0 |

C6 has **no SDS attributes at all**. C6.1 and the C code carry them (4-5 per SDS).

- **Granule QA metadata** (`QAPERCENTGOODQUALITY`, `QAPERCENTCLOUDCOVER`, …) is a placeholder in the C code ("set Passed as default, will decide later"). h08v05 C6 shows 100% good and 0% cloud.

**Python now** (bit-15 RF, `rf_aggregation_maps_20261001/out`, h08v05): 5 single-layer GeoTIFFs, **Int16, LZW, 1-row strips, no nodata set**, 135 MB in total. That is 1.6x the C code's 82 MB for 7 layers. The causes are the Int16 type (the product is uint8) and LZW on row strips.

## Quality and Cloud: what the C code computes (`MOD_PR44B1/make_metrics.c:58-104`)

For each of the 8 composite periods n (3 input days each; the last has 2):
- **`Quality` bit n** is set when no input day passed QA (`state` flag == 1). The period then falls back to the plain mean of all days.
- **`Cloud` bit n** is set only when the period fell back **and** no input day had the clear cloud flag.
- Water and out-of-projection pixels get 0 in both (`MOD44B3.c:388, 471`).
- Neither layer is used in prediction.

On h08v05 C6, `Quality` is 0 for 20.2 M pixels and 255 (all 8 periods fell back) for 111 k.

**What Python has** (`model/CompositeDayFile.py`):
- `outQa` (line 282) is the same per-period fallback flag. It is saved as `MOD44-<tile>-<date>-Band_<b>-qa.npy` beside each composite in `2-Composites/`: 8 periods x 8 bands, all present for h08v05 in `qaC_full63_retrain_20260929/Python/metrics_C`.
- `outCloud` (line 285) has three defects:
  1. It is inverted. The input cloud mask is 1 = clear, so `cloudTotal == 3` flags periods whose days were all clear.
  2. It is not tied to the fallback.
  3. It can never fire in the last period, which has only 2 inputs.
- Neither flag is gathered into a product layer.

## Proposed method

1. **`Quality`:** pack the 8 per-period `qa` masks into bits 0-7, with period 1 in bit 0, as in the C code. One band's mask is enough if the masks are identical across bands; step 0 checks this.
2. **Fix `outCloud`** to the C code's meaning: `fellBack & (number of clear inputs == 0)`. Then pack it into bits 0-7 in the same way.
3. **Water and out-of-projection = 0** in both layers, as in the C code.
4. **The value layers are unchanged.** Composites and predictions do not read these flags. Only the saved `-cloud.npy` changes, and nothing in prediction reads it. Existing metrics folders can be packed without being recomputed: `Quality` straight from the saved masks, and `Cloud` from the corrected flag. The corrected flag needs the per-day cloud masks, so it is recomputed from the inputs for the test tiles.

## A single product file like the original

One file per tile and year, with the historical name and layout: `MOD44B.A<yyyy>065.<tile>.<collection>.<production time>.hdf`.
- **All 7 SDSs**, with the historical names, types and order:
  - value layers uint8, with water 200;
  - out-of-projection code (Python 250, attribute says 253; decided in step 0);
  - SDs uint16 with scale 0.01;
  - `Quality` and `Cloud` uint8.
- **Attributes on every SDS** as in C6.1, apart from the wrong SD `valid_range`, which is not copied. Add `scale_factor` to the SDs only.
- **Compression on every SDS:** DEFLATE. Step 2 picks the level (6 or 9) and chunk size (whole tile vs 480 x 480) by measuring size and read time. Expected size: about 80 MB per tile against 135 MB now, even with the two extra layers.
- **Grid metadata** (`StructMetadata.0`): sinusoidal grid, corners taken from the MOD44C input (as `make_grid.c` does), so GDAL and HEG read the georeferencing.
- **ECS metadata** (`CoreMetadata.0`, `ArchiveMetadata.0`): built from `SHR_MCF/MOD44B.C7.mcf`. This is the hardest part. The C code gets it from the SDP toolkit, and Python would have to write the ODL text itself (see "How to write it" below).
- **GeoTIFFs** stay as an optional output (`--geotiff`) for analysis runs. When kept, write them as uint8/uint16, DEFLATE, tiled 512 x 512, with nodata set.

### How to write HDF-EOS2 from Python (decision needed)

The container (`modis-vcf-latest`) has GDAL with HDF4 read support, `h5py` 3.12 and `rasterio`. It has **no `pyhdf`**. GDAL's `HDF4Image` writer produces plain HDF4 without the EOS grid, so it is ruled out.

| Option | How | For | Against |
|---|---|---|---|
| **A. `pyhdf` (recommended)** | Add `pyhdf` to the container. Write the SDSs with DEFLATE through `SD`, build the `MOD44B_250m_GRID` Vgroup ("Data Fields", "Grid Attributes"), and write `StructMetadata.0`, `CoreMetadata.0` and `ArchiveMetadata.0` as ODL text | Pure Python; no SDP toolkit | The EOS structure is hand-built, so it must be validated |
| B. C packer | A small C tool based on `make_grid.c` and `create_output_file.c` packs raw arrays from Python | Reuses code with a known-good EOS structure and toolkit metadata | Keeps the SDP toolkit and C build in the production path, which the move to Python is meant to remove |

## Steps (forest nodes only, outputs in a new run root `FullRuns/quality_hdf_<date>/`)

**Code location:** a copy of the Python code in the new run root. Nothing goes on the delivery branch until the results are approved. Nothing in rlgill's checkouts is touched.

0. **Cheap checks (h08v05, forest203, about 10 minutes, then stop for review):**
   - Are the 8 bands' `-qa.npy` masks identical within each period?
   - Does Python `Quality`, packed from the existing masks, match the C code's `Quality` on h08v05? Report the exact-match share and the per-bit agreement. Inputs differ (bit 15 against the C code's `state`), so close agreement is the target, not exact.
   - Which code does the C code write for out-of-projection in the value layers (250 or 253)? Check on a tile that has some (e.g. h31v11).
   - Can `pyhdf` be installed in a test container overlay, and can it read the C code's file?
1. **Code:**
   - the `outCloud` fix;
   - a `packPeriodFlags` helper;
   - `Quality` and `Cloud` written by `VcfPredict` next to the SDs;
   - unit checks: the bit packing on synthetic masks, and the cloud rule on all 8 input cases of 3 days (2 for the last period).
2. **Writer:**
   - an `Mod44bHdfWriter` class (option A) with a writer-independent layer description, so an HDF5 writer can be added later without touching `VcfPredict`;
   - compression tests on h08v05 (levels 6 and 9, two chunk sizes): file size and full-read time.
3. **Validation on the 5 SD test tiles** (h08v05, h12v09, h19v11, h20v06, h12v02):
   - GDAL opens the file as `HDF4_EOS:EOS_GRID` with the right projection and corners (compared with the C code's file);
   - `hdp dumpsds` shows DEFLATE on all 7 SDSs;
   - every SDS equals the GeoTIFF it replaces (values exact);
   - `Quality`/`Cloud` against the C code's layers (agreement by bit, and the share of pixels with 3 or more fallback periods);
   - ECS metadata parses, and its fields match the MCF.
4. **Report:** `Reports/Quality_Cloud_Layers_and_HDF_Output_Results.md`. If approved, a separate step adds the code to the delivery branch.

## Expected cost

About 1.5-2 hours of wall time: step 0 about 10 minutes, code and writer about 1 hour, validation on 5 tiles about 20 minutes. No prediction reruns are needed. Packing reads existing masks and maps.

## HDF-EOS2 or HDF5? (suggestion)

**Recommendation: keep HDF-EOS2 (HDF4) for the MOD44B Collection 7 product, but build the writer so that HDF5 is a second back end and not a rewrite.**

Reasons to stay with HDF-EOS2 for this product:
- **Collection continuity.** MOD44B has been HDF-EOS2 for its whole record, as have the other MODIS land products. Users' readers, HEG, AppEEARS and the existing MCF (`SHR_MCF/MOD44B.C7.mcf`) all expect it. A format change in one product mid-record is a cost to users with little gain in science.
- **Ingest.** Whatever MODAPS / LP DAAC expects for C7 decides this. **Confirm with them before building.** If they will take HDF5 for C7, the case changes.
- **Size is not a reason to switch.** Both formats use the same DEFLATE. HDF5 adds a shuffle filter, which would help the uint16 SDs a little (perhaps 10-20%), but the uint8 layers dominate the file size.

Reasons HDF5 is the better format in general:
- HDF4 is legacy. Python write support is thin (`pyhdf` only, not in the container), and there is no maintained Python HDF-EOS2 writer.
- `h5py` is already in the container. HDF5 (as HDF-EOS5, or netCDF-4/CF) supports chunked, partial reads from cloud object storage, which is where Earthdata distribution is going.
- The VIIRS land products that continue MODIS use HDF5-based formats. A future VIIRS VCF from this code would want HDF5 anyway.

**So:** C7 MOD44B in HDF-EOS2 (option A writer), with a layer description that an HDF5 / netCDF-4 writer can reuse. Revisit only if the DAAC accepts HDF5 for C7, or when the code is pointed at VIIRS.

## Not in scope

- Changing how composites fall back (that is the cloud-composite work in `Cloud_Composite_Alternatives_Plan.md`).
- Filling the granule-level QA percentages (`QAPERCENTGOODQUALITY`, …) with real values. It is easy to add from `Quality` once the writer exists. The C code never did it.
- Using `Quality` or `Cloud` to mask or weight predictions.
- Re-running predictions on the 63 tiles; the files are packed from existing outputs.
