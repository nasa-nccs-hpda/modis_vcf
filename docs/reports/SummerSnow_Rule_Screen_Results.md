# Summer-snow ice rule: cheap screen (results, 2026-10-04 evening)

**Question (project lead, decision 19.5):** the candidate rule sets non-vegetated = 100 and tree cover = 0 where the summer snow share is
>= 0.25. Where does it trigger outside Greenland, and does it hurt tree cover there? Full tiles only if the screen is clean.

**Run:** `FullRuns/summersnow_screen_20261004` (forest201, 19:22-19:31, 69 tiles, read-only, no failures). Code `code/screen.py`, per
tile `out/<tile>_screen.csv`, all tiles `out/ALL_screen.csv`, printed tables `out/summary.txt`.

**Method.** Per pixel: the share of present MOD44CQ `state` inputs (2019 window, 23 inputs; 0 = no observation, not counted) with the
snow flag set (MOD44C state bit 5; see the mod44c-state-bits skill), as in `SnowFlag_IceMargin_Results.md`. Two summers:
- **JJA:** inputs dated 2019161-2019241 everywhere (the definition tested on h16v01);
- **HEMI:** JJA for v <= 08, December-February (2019353, 2020001-049) for v >= 09.

Thresholds 0.25 and 0.5. Compared with the v1.1.0 maps (`cand60med_20261004` CAND), C6 2019, MCD12Q1 2019, and the held-out tree
(bit-15 split C) and non-vegetated labels at the triggered pixels. "Greenland/Iceland" = h15v02, h16v00-02, h17v00-02.

## Results (threshold 0.25)

| | Tiles triggering outside Greenland/Iceland | Pixels outside G/I | Share of their land |
|---|---|---|---|
| JJA | 22 of 62 | 254,420 | 0.023% |
| HEMI | 21 of 62 | 92,579 | 0.008% |

At 0.5: JJA 15 tiles / 66,458 pixels; HEMI 14 tiles / 21,270 pixels.

**1. JJA everywhere fails: h12v12 (Patagonian Andes, southern winter).** 161,841 pixels (0.8% of the tile's land) trigger; v1.1.0 tree
cover there averages 15.5 (33% of pixels >= 10, 19% >= 30; 9% MCD12Q1 forest), C6 tree 11.1. The rule would remove about 2.5 M tree
cover points. This is seasonal snow, not ice. **If the rule goes ahead it must use the local summer (HEMI).**

**2. HEMI outside Greenland/Iceland: tree cover is barely touched.** 92,579 pixels, about 16,000 tree points in total. 90% of them are
on h11v02 (Canadian Arctic Archipelago ice caps: 83,122 pixels, 74% MCD12Q1 snow/ice, 22% barren; v1.1.0 tree 0.09, bare 98.3).
The rest are small clusters (<= 4,884 pixels per tile) in mountains: Rockies, Alps, Caucasus/Central Asia, Tibet, Siberia. A few tiny
clusters have real tree cover: h09v05 12 pixels (tree 32), h08v04 24 pixels (tree 20), h08v05 499 pixels (tree 5; 15% >= 10).

**3. The labels outside Greenland do not support the rule** (few labels; read as a warning, not a verdict):

| Tile | Labels | Mean label | v1.1.0 MAE | Rule MAE |
|---|---|---|---|---|
| h11v02 bare | 185 | 92.3 | 5.70 | 7.69 |
| h08v05 bare | 11 | 58.0 | 3.73 | 42.00 |
| h12v02 bare | 4 | 100 | 0.25 | 0.00 |
| h13v01 bare | 1 | 3 | 46 | 97 |
| tree (h11v02, h12v02, h13v01, h08v05) | 37 | 0-8.7 | 0.0-8.0 | 0.0-8.7 |

On h16v01 the rule changes nothing (v1.1.0 is already 100 at the labels: 88,502 bare labels, MAE 0 both).

**4. Greenland/Iceland: what the rule would change.** v1.1.0 bare at the triggered pixels already averages 99.4-100 (h16v02 99.51,
h15v02 99.42, h17v02 99.46), and tree cover is 0. The gain is the h16v02 90-99 band (`SnowFlag_IceMargin_Results.md`), a few points on
bare. h16v02 has no labels.

## Conclusion

- **Not clean as specified** (JJA): it would remove real tree cover in the Patagonian Andes (h12v12).
- **HEMI is almost clean for tree cover**, but outside Greenland it also changes ice caps and mountain snow where the few bare labels
  side with v1.1.0 (h11v02 +2 MAE; h08v05 +38 on 11 labels).
- The gain is limited to a few bare points on the Greenland ice sheet (mainly h16v02), where v1.1.0 already reads 99.4-99.9.
- **Options for the project lead:** (a) drop the rule; (b) HEMI rule on Greenland/Iceland tiles only (a location list, gain a few bare
  points on ~2 M h16v02 pixels, no labels to test it); (c) HEMI everywhere at 0.5 (21,270 pixels outside G/I) after a look at the
  h11v02 and h08v05 labels. Suggested: (a), or (b) if the h16v02 shortfall matters for delivery. No full-tile run was started.
