"""Cloud-fallback composites and snow-aware cloud counts in the package (cloud-fallback tree model inputs), from the HDF inputs, without daily files.

The arithmetic is that of v1.4.0's original script (blend/the original script, removed in v1.5.2, at tag v1.5.1), unchanged; only the
source of the inputs differs.  the original script read the unsubstituted
(low-sun mode off) dailies and standard composites of a separate mode-off root.  Here each period's inputs are read directly
in mode off (BandDayFile, as that root was built), the mode-off standard composite is made with CompositeDayFile's own
compositing, and the cloud-fallback composite, masks and counts are written.  Periods are independent and run in parallel processes.

Per period (3 inputs; the last period has 2 and the original script's never-read 24th slot), from MOD44CQ state (bits 0-1, 2, 5, 6) and
solar zenith, the mode-off Band_1 / Band_2 inputs and the composite's fallback flag (Band_1 -qa.npy, which does not depend on
the low-sun mode):
- trigger = every present input cloudy or cloud-adjacent, and the period fell back; pick = the input with the lowest red among
  present, not-shadow, valid, NDVI >= 0, NIR >= 1500 inputs; use = trigger and some input qualifies;
- cloud-fallback composite of each of the 8 bands = the picked input where use, else the mode-off standard composite;
- counts CUR_Z (use and some input with 0 < zenith < 65), CUR_ZS (... and not NDSI snow on the cloud-fallback composite), CUR_ZB (... and
  state bit 5 of the pick not set), and the number of periods replaced per pixel.
Self-checks (fatal, as in the original script): the pick's Band_1 equals the cloud-fallback Band_1 at every counted pixel-period; the mean of all
inputs equals the mode-off standard composite at fallback pixels (here over all fallback pixels; the original script used land only).

Outputs: <cloud-fallback root>/<tile>/<year>/2-Composites (cloud-fallback composites with the -qa.npy / -cloudfb.npy of the standard root), a low-sun
stamp 'off' on that root, and in <counts dir>: <tile>_counts_snow.npz (CUR_Z, CUR_ZS, CUR_ZB), <tile>_cloudfallback_used_count.npy and
<tile>_cloudfallback_periods.npz (per-period trigger and use bits, zenith-ok counts: for the statistics computed after the map).

Usage: python -m modis_vcf.model.CloudFallback build <tile> <year> <standard metrics root> <cloud-fallback metrics root> <counts dir> [workers]
       python -m modis_vcf.model.CloudFallback compose <tile> <year> <cloud-fallback metrics root> <standard metrics root> <counts dir> <tree map>
         (after the map: <tile>_compose.json statistics from the tree map, no C6; decision D5)
"""
import json
import logging
import multiprocessing
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

BANDS = ['Band_1', 'Band_2', 'Band_3', 'Band_4', 'Band_5', 'Band_6', 'Band_7', 'Band_31']
NIR_FLOOR = 1500
ROWS = COLS = 4800


# ----------------------------------------------------------------------------
# slots
# ----------------------------------------------------------------------------
def slots(year: int) -> list:
    """the original script's 24 input slots: DOY 065-353 of the year, 001-049 of the next, and the never-read 065 of the next."""

    s = [(year, d) for d in range(65, 366, 16)] + [(year + 1, d) for d in (1, 17, 33, 49)] + [(year + 1, 65)]
    assert len(s) == 24
    return s


def _key(yd) -> str:
    return '%d%03d' % yd


# ----------------------------------------------------------------------------
# _period (one worker process)
# ----------------------------------------------------------------------------
def _period(tile: str, year: int, p: int, stdComp: str, cloudFallbackComp: str, logLevel: int) -> dict:

    logging.basicConfig(level=logLevel, stream=sys.stdout)
    from modis_vcf.model import LowSun
    LowSun.configure('off')                              # the cloud-fallback inputs are unsubstituted (Q3)
    from modis_vcf.model.BandDayFile import BandDayFile
    from modis_vcf.model.CompositeDayFile import CompositeDayFile
    from modis_vcf.model.ProductTypeMod44 import ProductTypeMod44
    from modis_vcf.model.VcfPredict import MOD44_DIR

    BandDayFile.setWriteDailies(False)
    log = logging.getLogger('cloudFallback')
    pt = ProductTypeMod44(MOD44_DIR)
    stdComp, cloudFallbackComp = Path(stdComp), Path(cloudFallbackComp)
    S = slots(year)
    sl = S[3 * p: 3 * p + 3]
    pk = _key(sl[0])
    tmpDir = cloudFallbackComp.parent / '1-Days'                   # BandDayFile needs an existing folder; nothing is written there

    def bdf(yd, band):
        return BandDayFile().initFromParams(pt, band, tile, yd[0], yd[1], tmpDir, log)

    # State and zenith per slot, read once (the original script's sub(); the 24th slot is never read).
    st, zen = [], []

    for yd in sl:
        if yd == S[-1]:
            st.append(None); zen.append(None); continue
        try:
            b = bdf(yd, 'Band_1')
            st.append(b.getState()); zen.append(b.getSolz())
        except RuntimeError:
            st.append(None); zen.append(None)

    avail = [s is not None for s in st]

    # The composite's days and its per-day masks, exactly as CompositeDayFile._getRaster builds them.
    cdf = CompositeDayFile().initFromParams(pt, 'Band_1', tile, sl[0][0], sl[0][1], stdComp, logger=log, dayDir=tmpDir)
    daysToFind = cdf._getDaysToFind()
    masks = {}

    for yd in daysToFind:
        try:
            masks[yd] = cdf._getMasks(bdf(yd, 'Band_1'), yd[0], yd[1])
        except RuntimeError:
            masks[yd] = None

    def offDaily(yd, band):
        """The mode-off daily of a slot, or None (the original script's day())."""
        if yd == S[-1] or yd not in daysToFind:
            return None
        try:
            return bdf(yd, band).raster(applyQa=band != 'Band_31')
        except RuntimeError:
            return None

    def offComposite(ins: dict):
        """CompositeDayFile._getRaster with these inputs: missing day or mask -> substitute day."""
        rw = []
        for yd in daysToFind:
            if ins.get(yd) is None or masks[yd] is None:
                r = np.full((ROWS, COLS), CompositeDayFile.SUBSTITUTE_PIXEL_VALUE, dtype=np.int16)
                rw.append((r, np.zeros_like(r), np.zeros_like(r)))
            else:
                rw.append((ins[yd], masks[yd][0], masks[yd][1]))
        comp, _, _ = cdf._computeComposite(rw)
        return comp

    cloudy = np.ones((ROWS, COLS), bool)
    zenOk = np.zeros((ROWS, COLS), bool)

    for i, s in enumerate(st):
        if s is not None:
            cloudy &= ((s & 3) != 0) | ((s & 64) != 0)
            z = zen[i].astype(np.int32)
            zenOk |= (z > 0) & (z < 65)

    qa = np.load(stdComp / ('MOD44-%s-%s-Band_1-qa.npy' % (tile, pk))).astype(bool)
    trig = cloudy & qa & any(avail)
    days = {b: {yd: offDaily(yd, b) for yd in sl} for b in ('Band_1', 'Band_2')}
    b1 = [days['Band_1'][yd] for yd in sl]
    b2 = [days['Band_2'][yd] for yd in sl]
    red = np.full((len(sl), ROWS, COLS), np.iinfo(np.int32).max, np.int64)

    for i in range(len(sl)):
        if st[i] is None or b1[i] is None:
            continue
        r, n = b1[i].astype(np.int64), b2[i].astype(np.int64)
        sh = (st[i] & 4) != 0
        inv = (r <= -100) | (n <= -100) | (r >= 10000) | (n >= 10000)
        nd = np.where(r + n != 0, (n - r) / np.where(r + n != 0, r + n, 1), -1.0)
        ok = ~sh & ~inv & ~(nd < 0) & ~(n < NIR_FLOOR)
        red[i] = np.where(ok, r, red[i])

    anyok = red.min(0) < np.iinfo(np.int32).max
    pick = red.argmin(0)
    use = trig & anyok
    del red
    align, comps = [], {}

    for b in BANDS:
        ins = days[b] if b in days else {yd: offDaily(yd, b) for yd in sl}
        old = offComposite(ins)
        insl = [ins[yd] for yd in sl]
        present = [x if x is not None else np.full((ROWS, COLS), 10000, np.int16)
                   for x, s in zip(insl, sl) if x is not None or s != S[-1]]
        if present:
            allmean = np.mean(np.stack(present).astype(np.float64), 0).astype(np.int16)
            align.append(dict(period=pk, band=b, fallback_px=int(qa.sum()),
                              identical=float((allmean[qa] == old[qa]).mean()) if qa.any() else 1.0))
        stack = np.stack([x if x is not None else np.zeros((ROWS, COLS), np.int16) for x in insl])
        picked = np.take_along_axis(stack, pick[None], 0)[0]
        new = np.where(use, picked, old).astype(np.int16)
        new.tofile(cloudFallbackComp / ('MOD44-%s-%s-%s.bin' % (tile, pk, b)))
        for suf in ('-qa.npy', '-cloudfb.npy'):
            src = stdComp / ('MOD44-%s-%s-%s%s' % (tile, pk, b, suf))
            if src.exists():
                shutil.copy(src, cloudFallbackComp)
        if b in ('Band_1', 'Band_2', 'Band_4', 'Band_6'):
            comps[b] = new
        if b == 'Band_1':
            stackb1 = stack
        del stack, picked, ins, insl, present

    cu = use & zenOk
    pickMism = int((np.take_along_axis(stackb1, pick[None], 0)[0][cu] != comps['Band_1'][cu]).sum())
    c2, c4, c6 = (comps[b].astype(np.float64) for b in ('Band_2', 'Band_4', 'Band_6'))
    ndsi = (c4 - c6) / np.where(c4 + c6 != 0, c4 + c6, 1)
    snowNdsi = (ndsi >= 0.4) & (c2 >= 1100) & (c4 + c6 != 0)
    sstack = np.stack([s if s is not None else np.zeros((ROWS, COLS), np.uint16) for s in st]).astype(np.int64)
    snowBit = (np.take_along_axis(sstack, pick[None], 0)[0] & 32) != 0
    return dict(p=p, period=pk, inputs_available=int(sum(avail)), use=use, trig=trig, cu=cu,
                cuZs=cu & ~snowNdsi, cuZb=cu & ~snowBit, snowNdsi=snowNdsi, snowBit=snowBit,
                align=align, pick_mismatch=pickMism)


# ----------------------------------------------------------------------------
# buildTile
# ----------------------------------------------------------------------------
def buildTile(tile: str, year: int, stdRoot: Path, cloudFallbackRoot: Path, countsDir: Path, workers: int = 8,
              logger: logging.Logger = None) -> dict:
    """Cloud-fallback composites, counts and per-period masks of one tile-year.  Refuses an existing cloud-fallback tile-year folder."""

    log = logger or logging.getLogger('cloudFallback')
    stdComp = Path(stdRoot) / tile / str(year) / '2-Composites'
    out = Path(cloudFallbackRoot) / tile / str(year)

    if (out / '2-Composites').exists() and any((out / '2-Composites').iterdir()):
        raise RuntimeError('Cloud-fallback composites exist, refusing: ' + str(out))

    (out / '2-Composites').mkdir(parents=True, exist_ok=True)
    (out / '1-Days').mkdir(parents=True, exist_ok=True)
    countsDir = Path(countsDir); countsDir.mkdir(parents=True, exist_ok=True)

    # Stamp the cloud-fallback root 'off' before anything is written in it (the cloud-fallback tree model reads it in mode off).
    from modis_vcf.model import LowSun
    from modis_vcf.model.Metrics import Metrics
    from modis_vcf.model.ProductTypeMod44 import ProductTypeMod44
    from modis_vcf.model.VcfPredict import MOD44_DIR
    Metrics(tile, year, ProductTypeMod44(MOD44_DIR), Path(cloudFallbackRoot), log, lowSun='off')

    ctx = multiprocessing.get_context('spawn')
    res = []

    with ProcessPoolExecutor(max(1, min(workers, 8)), mp_context=ctx) as ex:
        futs = [ex.submit(_period, tile, year, p, str(stdComp), str(out / '2-Composites'), log.getEffectiveLevel())
                for p in range(8)]
        res = sorted((f.result() for f in futs), key=lambda r: r['p'])

    C = {k: np.zeros((ROWS, COLS), np.uint8) for k in ('CUR_Z', 'CUR_ZS', 'CUR_ZB')}
    nused = np.zeros((ROWS, COLS), np.uint8)
    trigBits = np.zeros((ROWS, COLS), np.uint8); useBits = np.zeros((ROWS, COLS), np.uint8)
    snowNdsiBits = np.zeros((ROWS, COLS), np.uint8); snowBitBits = np.zeros((ROWS, COLS), np.uint8)
    cuBits = np.zeros((ROWS, COLS), np.uint8)

    for r in res:
        nused += r['use']; C['CUR_Z'] += r['cu']; C['CUR_ZS'] += r['cuZs']; C['CUR_ZB'] += r['cuZb']
        bit = np.uint8(1 << r['p'])
        trigBits |= np.where(r['trig'], bit, 0).astype(np.uint8); useBits |= np.where(r['use'], bit, 0).astype(np.uint8)
        cuBits |= np.where(r['cu'], bit, 0).astype(np.uint8)
        snowNdsiBits |= np.where(r['snowNdsi'], bit, 0).astype(np.uint8)
        snowBitBits |= np.where(r['snowBit'], bit, 0).astype(np.uint8)

    np.save(countsDir / ('%s_cloudfallback_used_count.npy' % tile), nused)
    np.savez_compressed(countsDir / ('%s_counts_snow.npz' % tile), **C)
    np.savez_compressed(countsDir / ('%s_cloudfallback_periods.npz' % tile), trig=trigBits, use=useBits, cu=cuBits,
                        snow_ndsi=snowNdsiBits, snow_bit=snowBitBits)
    align = [a for r in res for a in r['align']]
    summary = dict(tile=tile, year=year, alignment_min_identical=min([a['identical'] for a in align] or [1.0]),
                   pick_band1_mismatch=sum(r['pick_mismatch'] for r in res),
                   inputs_available=[r['inputs_available'] for r in res])
    (countsDir / ('%s_cloudfallback_build.json' % tile)).write_text(json.dumps(dict(summary=summary, align=align), indent=1))

    if summary['alignment_min_identical'] < 0.999:
        raise RuntimeError('Cloud-fallback alignment failed: %s' % summary)

    if summary['pick_band1_mismatch']:
        raise RuntimeError('Cloud-fallback pick mismatch: %s' % summary)

    return summary


# ----------------------------------------------------------------------------
# composeStats (after the map; decision D5)
# ----------------------------------------------------------------------------
def composeStats(tile: str, year: int, cloudFallbackRoot: Path, stdRoot: Path, countsDir: Path, treeMap: Path) -> dict:
    """<tile>_compose.json statistics (diagnostic only; nothing reads them back), from the saved per-period bits, the
    cloud-fallback composites, the standard composites' fallback flags and the year's tree map: land = map <= 100, dense = map 60-100.
    No external reference is read (until v1.5.1, land and dense also used C6 MOD44B when present).  The alignment
    figure is the build's (all fallback pixels, not land only)."""

    from osgeo import gdal
    countsDir = Path(countsDir)
    S = slots(year)
    vc = Path(cloudFallbackRoot) / tile / str(year) / '2-Composites'
    sc = Path(stdRoot) / tile / str(year) / '2-Composites'
    if Path(treeMap).exists():
        G = gdal.Open(str(treeMap)).ReadAsArray()
    else:                                                # an h5-only run: the tree layer of the C7 file
        from modis_vcf.model import Mod44bH5
        G = Mod44bH5.readMapLayer(Path(treeMap).parent, tile, year, 'Percent_Tree_Cover')
    land = G <= 100
    dense = land & (G >= 60)
    sets = dict(land=land, dense=dense)
    bits = np.load(countsDir / ('%s_cloudfallback_periods.npz' % tile))
    build = json.loads((countsDir / ('%s_cloudfallback_build.json' % tile)).read_text())
    C = dict(np.load(countsDir / ('%s_counts_snow.npz' % tile)))
    nused = np.load(countsDir / ('%s_cloudfallback_used_count.npy' % tile))
    ndv = lambda r, n: (n.astype(float) - r) / np.maximum(n.astype(float) + r, 1)
    stats, per = [], []

    for p in range(8):
        pk = _key(S[3 * p])
        bit = np.uint8(1 << p)
        trig, use, cu = ((bits[k] & bit) != 0 for k in ('trig', 'use', 'cu'))
        snowNdsi, snowBit = ((bits[k] & bit) != 0 for k in ('snow_ndsi', 'snow_bit'))
        qa = np.load(sc / ('MOD44-%s-%s-Band_1-qa.npy' % (tile, pk))).astype(bool)
        rd = lambda b: np.fromfile(vc / ('MOD44-%s-%s-%s.bin' % (tile, pk, b)), np.int16).reshape(ROWS, COLS)
        n1, n2 = rd('Band_1'), rd('Band_2')
        o2 = n2                                         # the cloud-fallback composite is the standard one wherever use is off (use implies qa)
        rec = dict(period=pk, inputs_available=build['summary']['inputs_available'][p])

        for sname, m in sets.items():
            u = use & m
            rec[sname] = dict(px=int(m.sum()), trig_share=float((trig & m).sum() / max(1, m.sum())),
                              used_share=float(u.sum() / max(1, m.sum())), used_px=int(u.sum()),
                              new_med=dict(red=float(np.median(n1[u])), nir=float(np.median(n2[u])),
                                           ndvi=float(np.median(ndv(n1[u], n2[u])))) if u.any() else None,
                              clear_ref=dict(nir=float(np.median(o2[m & ~qa]))) if (m & ~qa).any() else None)

        stats.append(rec)
        u = cu & land
        nn = max(1, int(u.sum()))
        per.append(dict(period=pk, curz_share=float(u.sum() / max(1, land.sum())),
                        ndsi_snow_share_of_curz=float((u & snowNdsi).sum() / nn),
                        bit_snow_share_of_curz=float((u & snowBit).sum() / nn)))

    ratios = [s['dense']['new_med']['nir'] / s['dense']['clear_ref']['nir'] for s in stats
              if s['dense']['new_med'] and s['dense']['clear_ref'] and s['dense']['used_px'] >= 10000]
    gate = float(np.median(ratios)) if ratios else None
    summary = dict(tile=tile, year=year, alignment_min_identical=build['summary']['alignment_min_identical'],
                   gate_nir_ratio=gate, gate_flag=bool(gate is not None and gate < 0.6),
                   pick_band1_mismatch=build['summary']['pick_band1_mismatch'],
                   land_pixel_periods_replaced_share=float(nused[land].sum() / max(1, 8 * land.sum())),
                   land_pixels_any_replaced_share=float((nused[land] > 0).mean()) if land.any() else 0.0,
                   land_dist={k: np.bincount(v[land], minlength=9).tolist() for k, v in C.items()})
    (countsDir / ('%s_compose.json' % tile)).write_text(json.dumps(dict(summary=summary, stats=stats,
                                                                        align=build['align'], periods=per), indent=1))
    return summary


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main() -> int:

    a = sys.argv[1:]
    logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

    if len(a) >= 6 and a[0] == 'build':
        s = buildTile(a[1], int(a[2]), Path(a[3]), Path(a[4]), Path(a[5]), int(a[6]) if len(a) > 6 else 8)
    elif len(a) == 7 and a[0] == 'compose':
        s = composeStats(a[1], int(a[2]), Path(a[3]), Path(a[4]), Path(a[5]), Path(a[6]))
    else:
        print(__doc__)
        return 2

    print(json.dumps(s))
    return 0


if __name__ == '__main__':
    sys.exit(main())
