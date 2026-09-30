#!/usr/bin/python
"""Predict many tiles in ONE process: the two models are loaded once, then tiles are taken from a shared queue.

Tiles are claimed with mkdir (<claimDir>/predict_<tile>) so several nodes can run this at the same time; a marker file
<markerDir>/predict_<tile>.ok or .failed records the result.  A tile that already has a claim is skipped, so a restarted
run never redoes or overwrites finished work.
"""
import argparse
import sys
import traceback
from pathlib import Path

from modis_vcf.model.VcfPredict import MOD44_DIR
from modis_vcf.model.VcfPredict import VcfPredict


def main():

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('-o', type=Path, required=True, help='Directory for the predicted tifs.')
    ap.add_argument('--metricsDir', type=Path, required=True, help='Root holding <tile>/<year>/3-Metrics.')
    ap.add_argument('--modisDir', type=Path, default=MOD44_DIR)
    ap.add_argument('--tc', type=Path, required=True)
    ap.add_argument('--tcn', type=Path, required=True)
    ap.add_argument('--nv', type=Path, required=True)
    ap.add_argument('--nvn', type=Path, required=True)
    ap.add_argument('--tilesFile', type=Path, required=True)
    ap.add_argument('--claimDir', type=Path, required=True)
    ap.add_argument('--markerDir', type=Path, required=True)
    ap.add_argument('--nJobs', type=int, default=1)
    ap.add_argument('-y', type=int, required=True)
    args = ap.parse_args()

    tiles = args.tilesFile.read_text().split()
    args.claimDir.mkdir(parents=True, exist_ok=True)
    args.markerDir.mkdir(parents=True, exist_ok=True)

    vcfp = VcfPredict(args.tc, args.tcn, args.nv, args.nvn, args.o, args.metricsDir, args.modisDir,
                      nJobs=args.nJobs)

    for tile in tiles:

        try:
            (args.claimDir / ('predict_' + tile)).mkdir()
        except FileExistsError:
            continue

        try:
            vcfp.run(tids=[tile], years=[args.y])
            (args.markerDir / ('predict_%s.ok' % tile)).write_text('ok\n')
            print('PREDICT OK', tile, flush=True)

        except Exception:                                                # noqa: BLE001
            (args.markerDir / ('predict_%s.failed' % tile)).write_text(traceback.format_exc()[-1500:])
            print('PREDICT FAILED', tile, flush=True)
            traceback.print_exc()


if __name__ == '__main__':
    sys.exit(main())
