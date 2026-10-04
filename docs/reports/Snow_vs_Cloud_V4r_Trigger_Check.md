# Does the V4r trigger fire on snow and low sun at high latitude? (check, h12v03 against h12v09)

**2026-10-03. Read-only check; run folder `FullRuns/snow_flag_check_20261003` (script `code/snow_flags.py`, outputs `out/`, first pass
without the zenith columns in `out_v1/`).** Part of the cloudy-pixel experiments on v1.0.0 (`Product_Versions_and_Work_Map.md`).

## Question

V4r replaces the most composites on northern tiles: h12v03 has 44% of land pixel-periods replaced, against a median of 7% across the
63 tiles. Is that clouds, or snow and low sun flagged as cloud? Does MOD44C carry a snow flag that would separate tropical clouds from
high-latitude snow?

## What MOD44C `state` can say

From the PGE72 investigation (`MOD44C_PGE72_Investigation/Reports/MOD44C_State_Aerosol_Bit_Collision.md`):
- **Bit 5 = MOD09 bit 12, the MOD35 snow/ice flag.** This is the only snow information in MOD44C.
- MOD09's internal snow mask (bit 15) is not carried into MOD44C.
- **Bits 0-1 come from MOD09's internal cloud-algorithm flag when that flag is set**, which gives value 1. Otherwise they come from
  MOD09's cloud state.

## Method

For h12v03 (central Canada, Manitoba / Hudson Bay, about 50-60 N, 96 W) and h12v09 (eastern Amazon), each of the 8 periods, land pixels only:
- **CUR:** the V4r trigger as in `v4r_compose.py`. It reproduces the stored counts exactly on both tiles (0 pixels differ).
- **SNOWX:** the same trigger, but an input with the snow bit set does not count as cloud.
- **ZENX:** the same trigger, but only where at least one input has 0 < solar zenith < 65. In other words, the fallback could be
  caused by cloud rather than low sun.
- **BOTH:** SNOWX and ZENX together.

Shares are of land pixels in each period.

## Results

| Period | h12v03 fallback | CUR | SNOWX | ZENX | BOTH | Triggered pixels with every input at zenith ≥ 65 |
|---|---|---|---|---|---|---|
| 2019 065 (Mar) | 0.89 | 0.85 | 0.63 | 0.85 | 0.63 | 0.00 |
| 2019 113 (Apr-May) | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.00 |
| 2019 161-257 (Jun-Sep) | ≤ 0.003 | ≤ 0.003 | ≤ 0.003 | ≤ 0.003 | ≤ 0.003 | 0.00 |
| 2019 305 (Nov) | 1.00 | 0.96 | 0.57 | **0.04** | 0.03 | **0.96** |
| 2019 353 (Dec) | 1.00 | 0.79 | 0.47 | **0.00** | 0.00 | **1.00** |
| 2020 033 (Feb) | 1.00 | 0.89 | 0.70 | 0.77 | 0.62 | 0.14 |
| **Year (pixel-periods)** | | **0.44** | 0.30 | **0.21** | 0.17 | |

**h12v09:** CUR, SNOWX, ZENX and BOTH are identical in every period. The snow bit is never set, and no input has zenith ≥ 65.

**Flags on the inputs of triggered h12v03 pixel-periods (winter periods):**
- Cloud value is 1 in 95-100% of inputs. Values 2 and 3 never occur, on either tile.
- The snow bit is set on 12-18% of inputs, and nearly all of those also carry cloud value 1.
- In only 2-3% of triggered pixel-periods do all inputs carry the snow bit.

## Findings

1. **Most of h12v03's firing is low sun, not cloud or snow.**
   - In November and December, every land pixel falls back because all inputs have zenith ≥ 65, the same reason as Greenland's polar
     night.
   - V4r then fires because MOD09 flags those low-sun inputs as cloud.
   - **The zenith test (ZENX) removes this completely and changes nothing in the tropics.** h12v03 drops from 44% to 21% of
     pixel-periods.
2. **The snow bit separates only part of the snow.**
   - It is set on a minority of inputs, mostly together with a cloud flag, and rarely on all inputs.
   - SNOWX removes about a quarter of the March and February triggers.
   - It is a weak discriminator on its own. It is safe for the tropics, where it is never set.
3. **What remains on h12v03 after ZENX is February and March, with adequate sun.** That is about 77-85% of land, snow season, mostly
   flagged as cloud. MOD44C cannot tell cloud from cloud-flagged snow there.
4. **Consistent with earlier observations:** on the northern tiles the V4r picks were brighter than clear pixels (NIR ratio
   1.12-1.33), as expected from snow.

## Implications

- **For the blend and V4r generally:** add the zenith test to the trigger, so V4r does not fire where fallback is caused by low sun.
  This matches the meaning of "V4r only where clouds caused the fallback". It is cheap: zenith is already read for the QA test.
- **For the remaining snow-season triggers,** a snow test from the bands themselves could be tried.
  - NDSI = (green − SWIR) / (green + SWIR), with MODIS Band 4 and Band 6, both in MOD44CH. Snow has high NDSI, around 0.4 or more;
    clouds have low NDSI, because they are bright in the SWIR.
  - Alternatively, use an external snow product (MOD10A1/A2).
  - Not tested yet.
- Applying any of these changes the V4r composites, so it needs the V4r composites, metrics, tables and V4R refit to be rebuilt for 63
  tiles. ZENX alone changes only tiles with low-sun periods.

## Four more northern tiles (added 2026-10-03)

h24v03, h23v03, h11v03 and h12v04 were checked the same way. All reproduce the stored counts exactly (0 pixels differ).

**Annual share of land pixel-periods replaced:**

| Tile | Where | CUR | SNOWX | ZENX | BOTH |
|---|---|---|---|---|---|
| h12v03 | central Canada, about 50-60 N, 96 W | 0.44 | 0.30 | 0.21 | 0.17 |
| h24v03 | eastern Siberia, about 50-60 N, 113 E | 0.39 | 0.29 | 0.20 | 0.17 |
| h23v03 | central Siberia, about 50-60 N, 96 E | 0.38 | 0.24 | 0.18 | 0.14 |
| h11v03 | western Canada, about 50-60 N, 113 W | 0.33 | 0.23 | 0.14 | 0.12 |
| h12v04 | northeastern US, Great Lakes, southern Ontario and Quebec, about 40-50 N, 78 W | 0.33 | 0.28 | 0.29 | 0.25 |
| h12v09 | eastern Amazon (control) | 0.41 | 0.41 | 0.41 | 0.41 |

**By period:**
- **On all four v03 tiles, November and December are pure low sun.** In 97-100% of triggered pixel-periods every input has zenith ≥ 65,
  and ZENX takes those periods to 0-2%. The same holds for 14-26% of February.
- **h12v04 is further south.** Low sun appears only in December (43% of triggered pixel-periods). Its winter firing is mostly with
  adequate sun: 50-82% of land in March, November and February. That is consistent with winter cloud plus snow, so ZENX removes little
  there (0.33 to 0.29).
- **Spring and autumn snow:** the snow bit is set in at least one input of 6-54% of triggered pixel-periods. It is higher in November
  to February (20-54%) than in spring (6-24%). SNOWX removes 10-50% of the triggers in those periods.
- **Summer** (June-September) triggers are small (0.3-7% of land) and almost never snow-flagged.

**Conclusion over the six tiles:**
- **The zenith test is the reliable separator:** it removes the low-sun firing (about half of all firing on the v03 tiles) and nothing
  in the tropics.
- **The snow bit is a partial, safe add-on:** never set in the tropics, but it catches only part of the snow.
- After both, 12-25% of land pixel-periods on these tiles still fire, mostly February-March and November. MOD44C alone cannot say
  whether that is cloud or cloud-flagged snow.

## Open
- **NDSI on the February and March triggers of h12v03:** not run.
