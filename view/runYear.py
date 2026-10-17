import argparse
import logging
from pathlib import Path
import sys

from modis_vcf.model.YearRun import YearRun


# -----------------------------------------------------------------------------
# main
#
# One driver per year (model/YearRun.py).  Each node of a run executes
#   runYear.py run --spec <spec.json>
# from the run's detached lane script (claims, guards and watchers stay in
# the lane wrapper); the nodes share the tasks through <runRoot>/status.
#   runYear.py metrics --spec <spec.json> --tile hXXvYY   one metrics task (started by run)
#   runYear.py status --spec <spec.json>                   task counts and the failed tasks
# -----------------------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)

    for name in ('run', 'metrics', 'status'):
        p = sub.add_parser(name)
        p.add_argument('--spec', type=Path, required=True)
        if name == 'metrics':
            p.add_argument('--tile', required=True)
        if name == 'run':
            p.add_argument('--keepIntermediates', action='store_true',
                           help='Keep every tile\'s 1-Days, 2-Composites and 3-Metrics (no clean-up).')

    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    yr = YearRun(args.spec, logger)

    if args.cmd == 'run':
        if args.keepIntermediates:
            yr.cleanupMode = 'off'
        return yr.run()

    if args.cmd == 'metrics':
        yr.warmMetrics(args.tile)
        return 0

    done = [t for t in yr.tasks if yr._done(t)]
    failed = [t for t in yr.tasks if yr._failed(t)]
    print('%d tasks: %d done, %d failed, %d to go' % (len(yr.tasks), len(done), len(failed),
                                                      len(yr.tasks) - len(done) - len(failed)))
    for t in failed:
        print('FAILED', t)
    return 0


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
