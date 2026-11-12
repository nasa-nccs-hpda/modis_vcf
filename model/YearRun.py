"""The year driver: one processing year (metrics -> splits -> fits -> maps) as tasks shared by every node of a run.

Each node runs YearRun(spec).run() (through view/runYear.py).  The nodes share one task list through marker files in
<runRoot>/status:
  claims/<task>/                  taken (made with mkdir, so a task runs only once)
  <task>.done, <task>.failed      finished
  STOP, STOP_<host>               stop every node, or one node
  fatal_<task>.failed             a fatal failure: every node stops

Tasks, with the tasks each one waits for:
  metrics:<tile>  the predictor metrics of every model for one tile                 (none)
  split:<model>   the model's training split (TrainingManifest.build)               metrics of the tiles in its manifest
  fit:<model>     the fit (view/fitAltModel.py), then its compact copy              split:<model>
  map:<tile>      the 7 layers (view/vcfPredict.py, with the cloud-fallback tree    metrics:<tile> and the fits it uses
                  blend and the bare-priority rule inside)
A rule tile (rows v15-v17, model/RuleTile.py) has only its map task, written from MOD44W, with no metrics and no models.
A ready-made model (spec "fixed") has no split or fit task.  Metrics, split and fit failures are fatal (the year cannot be
completed); a failed map is retried once, then recorded.

The cloud-fallback tree model (spec "cloudFallbackTree"): for its blend tiles, and for its training tiles when it is fitted in the
run, the metrics task also builds the cloud-fallback composites, the cloud counts and the cloud-fallback metrics
(model/CloudFallback.py, low-sun mode off).  split:cloudFallbackTree takes its training rows from those metrics; fit:cloudFallbackTree
fits it.  After the map of a blend tile, the cloud-fallback statistics are written (CloudFallback compose, diagnostic only).

Timing: every attempt adds one row to <runRoot>/timing/tasks.csv: task, host, start, end, seconds, CPU seconds (user + system of the
task and the processes it waited for), peak memory of the task's processes, bytes read and written, and the return code.  Memory and
bytes are sampled every 5 s, so very short processes may be undercounted.

Spec (JSON):
  year, runRoot, lowSun                       the processing year, the run folder, the low-sun mode (le0)
  models.tree, models.bare                    either {manifest, predictors, trainingType, fit {...}} (fitted in the run;
                                              "splitOnly": true builds the split without a fit, and then no mapTiles are
                                              allowed) or {trainingType, fixed: {model, topN, sha256}} (ready-made; the
                                              optional sha256 file is checked before the first task)
  cloudFallbackTree (optional)                {tiles, manifest, predictors, trainingType, fit} (fitted in the run),
                                              {tiles, fixed: {model, topN}} (ready-made), or {tiles, model, topN,
                                              metricsDir, counts} (borrowed from another run: no composites built here)
  mapTiles                                    the tiles to map
  metricsTiles (optional)                     default: the manifests' tiles and the map tiles
  workers {metrics, split, mapJobs, cloudFallback}, fitHosts, slots {host prefix: n}, memGB {task type: GB}
  cleanup (optional) {mode: after_map (default) | off, keepTiles: [...]}, maxBacklog (optional)
  writeDailiesTiles (optional)                tiles whose daily inputs are also written to 1-Days (none by default)
  mapFormat (optional), h5Attrs (optional)    the map output format (h5 by default) and extra .h5 root attributes

Clean-up: once every task that reads a tile's metrics is done (its map, and the splits of the manifests
it is in), that tile's 1-Days, 2-Composites and 3-Metrics are deleted, unless the tile is in keepTiles.
A tile whose map failed is kept.  Only real folders inside the run's metrics root are deleted (a
symbolic link or a path outside runRoot is refused); the low-sun stamp and counts stay.  Each deletion
is logged to status/cleanup.log with the GB freed.  The metrics of training tiles are claimed first, and
maxBacklog holds new metrics claims of map-only tiles while that many of them wait for their map.
"""

import csv
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

from modis_vcf.model import RuleTile


TIMING_COLUMNS = ['task', 'host', 'attempt', 'start', 'end', 'seconds', 'cpu_seconds', 'peak_rss_gb', 'read_gb',
                  'write_gb', 'rc']


# ----------------------------------------------------------------------------
# _tree
# ----------------------------------------------------------------------------
def _tree(pid: int) -> list:
    """pid and its descendants (from /proc)."""

    children = {}

    for d in os.listdir('/proc'):

        if not d.isdigit():
            continue

        try:
            with open('/proc/' + d + '/stat') as f:
                ppid = int(f.read().rsplit(')', 1)[1].split()[1])
            children.setdefault(ppid, []).append(int(d))

        except (OSError, ValueError, IndexError):
            continue

    out, todo = [], [pid]

    while todo:
        p = todo.pop()
        out.append(p)
        todo.extend(children.get(p, []))

    return out


# ----------------------------------------------------------------------------
# _sample
# ----------------------------------------------------------------------------
def _sample(pids: list) -> dict:
    """{pid: (rss bytes, read bytes, write bytes, cpu seconds)}."""

    tick = os.sysconf('SC_CLK_TCK')
    page = os.sysconf('SC_PAGE_SIZE')
    out = {}

    for p in pids:

        try:
            with open('/proc/%d/statm' % p) as f:
                rss = int(f.read().split()[1]) * page
            io = {}
            with open('/proc/%d/io' % p) as f:
                for line in f:
                    k, v = line.split(':')
                    io[k] = int(v)
            with open('/proc/%d/stat' % p) as f:
                fields = f.read().rsplit(')', 1)[1].split()
            cpu = (int(fields[11]) + int(fields[12])) / tick
            # rchar / wchar: bytes through read / write calls, which counts network filesystems (panfs);
            # read_bytes / write_bytes count block devices only (a map read 0.04 GB that way).
            out[p] = (rss, io.get('rchar', 0), io.get('wchar', 0), cpu)

        except (OSError, ValueError, IndexError):
            continue

    return out


# ----------------------------------------------------------------------------
# YearRun
# ----------------------------------------------------------------------------
class YearRun(object):
    """One processing year as a task graph shared by the nodes of a run (module docstring): the spec, the task list and
    dependencies, the task commands, the metrics task itself (warmMetrics) and the driver loop (run).
    """
    def __init__(self, specFile: Path, logger: logging.Logger = None):

        self.specFile = Path(specFile)
        self.spec = json.loads(self.specFile.read_text())
        self.root = Path(self.spec['runRoot'])
        self.year = int(self.spec['year'])
        self.host = socket.gethostname().split('.')[0]
        self.st = self.root / 'status'
        self.logs = self.root / 'logs'
        self.timing = self.root / 'timing'
        self.metricsRoot = self.root / 'Python' / 'metrics'

        for d in (self.st / 'claims', self.logs, self.timing, self.metricsRoot, self.root / 'out' / 'maps'):
            d.mkdir(parents=True, exist_ok=True)

        self.logger = logger or logging.getLogger('runYear')
        self.models = self.spec['models']
        # Keys renamed in v1.9.0: stop rather than run without the blend.
        if 'partner' in self.spec or 'v4r' in self.spec.get('workers', {}):
            raise ValueError('Spec keys renamed in v1.9.0: "partner" is now "cloudFallbackTree", workers "v4r" is now '
                             'workers "cloudFallback".')

        self.cloudFallbackTree = self.spec.get('cloudFallbackTree')
        self.cloudFallbackTreeMode = None if not self.cloudFallbackTree else 'borrowed' if 'metricsDir' in self.cloudFallbackTree else \
            'fixed' if 'fixed' in self.cloudFallbackTree else 'fit'
        self.cloudFallbackRoot = self.root / 'Python' / 'cloudfallback'
        self.countsDir = self.root / 'out' / 'cloudfallback_counts'
        cleanup = self.spec.get('cleanup', {})
        self.cleanupMode = cleanup.get('mode', 'after_map')

        if self.cleanupMode not in ('after_map', 'off'):
            raise ValueError('cleanup mode must be after_map or off')

        self.keepTiles = set(cleanup.get('keepTiles', []))
        self.maxBacklog = self.spec.get('maxBacklog')
        self.tasks, self.deps, self.kind = self._taskList()
        self._cleaned = set()

    # ------------------------------------------------------------------------
    # manifest tiles
    # ------------------------------------------------------------------------
    def _model(self, name: str) -> dict:
        return self.cloudFallbackTree if name == 'cloudFallbackTree' else self.models[name]

    def _manifestTiles(self, name: str) -> list:

        meta = json.loads(Path(self._model(name)['manifest'] + '.json').read_text())
        tiles = set()

        for part in meta['parts'].values():
            tiles |= set(part['rows_per_tile'])

        return sorted(tiles)

    # ------------------------------------------------------------------------
    # taskList
    # ------------------------------------------------------------------------
    def _taskList(self) -> tuple:

        fitted = [m for m in ('tree', 'bare') if 'fixed' not in self.models[m]]
        mapTiles = self._withMod44w(list(self.spec['mapTiles']))
        self.ruleTiles = {t for t in mapTiles if RuleTile.isRuleTile(t)}
        mt = self.spec.get('metricsTiles')

        if self.cloudFallbackTreeMode == 'fit':
            fitted.append('cloudFallbackTree')

        self.trainingTiles = set().union(*[self._manifestTiles(m) for m in fitted])
        cloudFallbackTreeTiles = set(self.cloudFallbackTree['tiles']) if self.cloudFallbackTree else set()
        self.cloudFallbackCompositeTiles = set() if self.cloudFallbackTreeMode in (None, 'borrowed') else \
            (cloudFallbackTreeTiles & set(mapTiles)) | (set(self._manifestTiles('cloudFallbackTree')) if self.cloudFallbackTreeMode == 'fit' else set())

        if mt is None:
            mt = sorted((set(mapTiles) - self.ruleTiles) | self.trainingTiles | self.cloudFallbackCompositeTiles)

        tasks, deps, kind = [], {}, {}

        # Training tiles first, so the splits and fits can start while the map-only tiles are computed.
        for t in sorted(mt, key=lambda t: t not in self.trainingTiles):
            k = 'metrics:' + t
            tasks.append(k); deps[k] = []; kind[k] = 'metrics'

        splitOnly = [m for m in fitted if self._model(m).get('splitOnly')]

        if splitOnly and mapTiles:
            raise ValueError('splitOnly (%s) builds the split without a model: give no mapTiles.' % ', '.join(splitOnly))

        for m in fitted:
            k = 'split:' + m
            tasks.append(k); deps[k] = ['metrics:' + t for t in self._manifestTiles(m)]; kind[k] = 'split'
            if m in splitOnly:
                continue
            k = 'fit:' + m
            tasks.append(k); deps[k] = ['split:' + m]; kind[k] = 'fit'

        for t in mapTiles:
            k = 'map:' + t
            tasks.append(k)
            deps[k] = [] if t in self.ruleTiles else (['metrics:' + t] if t in mt else []) + \
                ['fit:' + m for m in fitted if m != 'cloudFallbackTree' or t in cloudFallbackTreeTiles]
            kind[k] = 'map'

        return tasks, deps, kind

    # ------------------------------------------------------------------------
    # withMod44w
    #
    # A tile with no MOD44W file for the year cannot be masked, so it gets no
    # map task: it is skipped (not failed) and listed in
    # status/skipped_no_mod44w.txt (h18v14 in 2019).
    # ------------------------------------------------------------------------
    def _withMod44w(self, tiles: list) -> list:

        from modis_vcf.model.ProductTypeMod44W import ProductTypeMod44W
        from modis_vcf.model.Settings import MOD44W_DIR

        pt, missing = ProductTypeMod44W(MOD44W_DIR), []

        for t in tiles:
            try:
                pt.findFile(t, self.year, 1, ProductTypeMod44W.WATER_MASK)
            except RuntimeError:
                missing.append(t)

        if missing:
            (self.st / 'skipped_no_mod44w.txt').write_text(
                ''.join(t + ' no MOD44W in ' + str(MOD44W_DIR / str(self.year) / '001') + '\n' for t in missing))
            self.logger.warning('No MOD44W, map skipped: ' + ' '.join(missing))

        return [t for t in tiles if t not in missing]

    # ------------------------------------------------------------------------
    # file names
    # ------------------------------------------------------------------------
    @staticmethod
    def _key(task: str) -> str:
        return task.replace(':', '_')

    def _done(self, task: str) -> bool:
        return (self.st / (self._key(task) + '.done')).exists()

    def _failed(self, task: str) -> bool:
        return (self.st / (self._key(task) + '.failed')).exists()

    def _finished(self, task: str) -> bool:
        return self._done(task) or self._failed(task)

    def _fatal(self) -> str:

        for f in [self.st / 'STOP', self.st / ('STOP_' + self.host)] + sorted(self.st.glob('fatal_*.failed')):
            if f.exists():
                return str(f)

        return ''

    def _mark(self, task: str, rc: int, note: str = '') -> None:

        k = self._key(task)

        if rc == 0:
            (self.st / (k + '.done')).write_text(time.strftime('%F %T') + '\n')

        else:
            (self.st / (k + '.failed')).write_text('rc=%d host=%s %s\n' % (rc, self.host, note))

            if self.kind[task] in ('metrics', 'split', 'fit'):
                (self.st / ('fatal_' + k + '.failed')).write_text('%s failed (rc=%d): the year cannot be completed\n'
                                                                    % (task, rc))

    def _log(self, msg: str) -> None:

        with open(self.st / ('lane_' + self.host + '.status'), 'a') as f:
            f.write('[' + time.strftime('%F %T') + '] ' + msg + '\n')

    # ------------------------------------------------------------------------
    # model paths
    # ------------------------------------------------------------------------
    def modelPaths(self, name: str) -> tuple:
        """(model, topN) for vcfPredict: the compact copy of a fitted model, or the fixed one."""

        m = self._model(name)

        if name == 'cloudFallbackTree' and self.cloudFallbackTreeMode == 'borrowed':
            return m['model'], m['topN']

        if 'fixed' in m:
            return m['fixed']['model'], m['fixed']['topN']

        d = self.root / 'Python' / 'models' / name
        return str(d / (m['trainingType'] + '.cf')), str(d / (m['trainingType'] + 'TopN.bin'))

    # ------------------------------------------------------------------------
    # commands
    # ------------------------------------------------------------------------
    def _commands(self, task: str) -> list:

        kind, arg = task.split(':', 1)
        py = [sys.executable, '-W', 'ignore']
        ls = ['--lowSun', self.spec.get('lowSun', 'le0')]
        w = self.spec.get('workers', {})

        if kind == 'metrics':
            return [py + ['-m', 'modis_vcf.view.runYear', 'metrics', '--spec', str(self.specFile), '--tile', arg]]

        m = self._model(arg) if kind in ('split', 'fit') else {}

        if kind == 'split':
            # The cloud-fallback tree model's rows come from the cloud-fallback root, built in low-sun mode off.
            root, sls = (self.cloudFallbackRoot, ['--lowSun', 'off']) if arg == 'cloudFallbackTree' else (self.metricsRoot, ls)
            return [py + ['-m', 'modis_vcf.view.trainingManifest', 'build', '--manifest', m['manifest'],
                          '--predictors', m['predictors'], '--root', str(root), '-y', str(self.year),
                          '-o', str(self.root / 'Python' / 'splits' / arg), '--workers', str(w.get('split', 6))] + sls]

        if kind == 'fit':
            s, d, f = self.root / 'Python' / 'splits' / arg, self.root / 'Python' / 'models' / arg, m['fit']
            return [py + ['-m', 'modis_vcf.view.fitAltModel', '--algo', 'rf', '-o', str(d),
                          '--xTrainPath', str(s / 'xTrain.parq'), '--yTrainPath', str(s / 'yTrain.parq'),
                          '--trainingType', m['trainingType'], '--predictorFile', m['predictors'],
                          '--nEstimators', str(f['nEstimators']), '--maxFeatures', str(f['maxFeatures']),
                          '--nJobs', str(f.get('nJobs', 12)), '--randomState', str(f['randomState']),
                          '--minSamplesLeaf', str(f.get('minSamplesLeaf', 1))],
                    py + ['-m', 'modis_vcf.model.CompactForest', str(d / (m['trainingType'] + '.bin')),
                          str(d / (m['trainingType'] + '.cf'))]]

        # Map output (VcfPredict.MAP_FORMATS): the C7 file by default; analysis runs set "mapFormat": "both".
        mf = ['--format', self.spec.get('mapFormat', 'h5')] + \
            [a for k, v in sorted(self.spec.get('h5Attrs', {}).items()) for a in ('--h5Attr', '%s=%s' % (k, v))]

        if kind == 'map' and arg in self.ruleTiles:
            return [py + ['-m', 'modis_vcf.view.vcfPredict', '-o', str(self.root / 'out' / 'maps'), '-t', arg,
                          '-y', str(self.year)] + mf]

        if kind == 'map':
            tc, tcn = self.modelPaths('tree')
            nv, nvn = self.modelPaths('bare')
            cmd = py + ['-m', 'modis_vcf.view.vcfPredict', '-o', str(self.root / 'out' / 'maps'),
                        '--metricsDir', str(self.metricsRoot), '-t', arg, '--tc', tc, '--tcn', tcn, '--nv', nv,
                        '--nvn', nvn, '-y', str(self.year), '--nJobs', str(w.get('mapJobs', 8)), '--tcAgg', 'median',
                        '--nvAgg', 'median'] + ls + mf

            blend = self.cloudFallbackTree and arg in self.cloudFallbackTree['tiles']

            if blend:
                pm, pn = self.modelPaths('cloudFallbackTree')
                borrowed = self.cloudFallbackTreeMode == 'borrowed'
                cmd += ['--cloudFallbackTree', pm, '--cloudFallbackTreeTopN', pn,
                        '--cloudFallbackMetricsDir', self.cloudFallbackTree['metricsDir'] if borrowed else str(self.cloudFallbackRoot),
                        '--counts', self.cloudFallbackTree['counts'] if borrowed else str(self.countsDir)]

            if self.spec.get('lowSunCheck') == 'off':
                cmd += ['--lowSunCheck', 'off']

            cmds = [cmd]

            if blend and arg in self.cloudFallbackCompositeTiles:
                cmds.append(py + ['-m', 'modis_vcf.model.CloudFallback', 'compose', arg, str(self.year), str(self.cloudFallbackRoot),
                                  str(self.metricsRoot), str(self.countsDir),
                                  str(self.root / 'out' / 'maps' / ('Percent_Tree_Cover-%s-%d.tif' % (arg, self.year)))])
            return cmds

        raise ValueError(task)

    # ------------------------------------------------------------------------
    # metrics task (in a subprocess of the driver)
    # ------------------------------------------------------------------------
    def warmMetrics(self, tile: str) -> None:
        """Every predictor of every model, cached in the run's metrics root (one Metrics, one tile)."""

        import pickle
        from modis_vcf.model import LowSun
        from modis_vcf.model.BandDayFile import BandDayFile
        from modis_vcf.model.Metrics import Metrics
        from modis_vcf.model.ProductTypeMod44 import ProductTypeMod44
        from modis_vcf.model.VcfPredict import MOD44_DIR

        LowSun.configure(self.spec.get('lowSun', 'le0'))
        # Dailies stay in memory unless the tile is listed for diagnostics (the composite workers inherit this).
        BandDayFile.setWriteDailies(tile in self.spec.get('writeDailiesTiles', []))
        names = []

        for name in ('tree', 'bare'):

            m = self.models[name]

            if 'fixed' in m:
                with open(m['fixed']['topN'], 'rb') as f:
                    cand = list(pickle.load(f))
            else:
                cand = Path(m['predictors']).read_text().split()

            names += [n for n in cand if n not in names]

        Metrics.compositeWorkers = max(1, int(self.spec.get('workers', {}).get('metrics', 8)))
        mt = Metrics(tile, self.year, ProductTypeMod44(MOD44_DIR), self.metricsRoot, self.logger)

        for n in names:
            mt.getMetricFromRf(n)

        # Release this phase's composites and metric cache before the cloud-fallback build, which needs its own (a Metrics object
        # may be held in a reference cycle, so collect it now rather than at the next garbage collection).
        del mt
        gc.collect()

        if tile not in self.cloudFallbackCompositeTiles:
            return

        # Cloud-fallback composites and counts, then the cloud-fallback tree model's predictors on the cloud-fallback root (mode off).
        from modis_vcf.model import CloudFallback
        done = self.countsDir / ('%s_cloudfallback_build.json' % tile)
        comp = self.cloudFallbackRoot / tile / str(self.year) / '2-Composites'

        if not done.exists():
            if comp.exists() and any(comp.iterdir()):
                # An interrupted attempt (e.g. a guard kill): move its partial cloud-fallback composites aside, then rebuild.
                aside = self.root / 'Python' / 'failed_attempts' / ('cloudfallback_%s_%d' % (tile, int(time.time())))
                aside.mkdir(parents=True)
                shutil.move(str(comp), str(aside / '2-Composites'))
            CloudFallback.buildTile(tile, self.year, self.metricsRoot, self.cloudFallbackRoot, self.countsDir,
                          int(self.spec.get('workers', {}).get('cloudFallback', 8)), self.logger)

        p = self.cloudFallbackTree

        if 'fixed' in p:
            with open(p['fixed']['topN'], 'rb') as f:
                pnames = list(pickle.load(f))
        else:
            pnames = Path(p['predictors']).read_text().split()

        vm = Metrics(tile, self.year, ProductTypeMod44(MOD44_DIR), self.cloudFallbackRoot, self.logger, lowSun='off')

        for n in pnames:
            vm.getMetricFromRf(n)

    # ------------------------------------------------------------------------
    # start / poll
    # ------------------------------------------------------------------------
    def _start(self, task: str, attempt: int) -> dict:

        log = open(self.logs / ('%s_a%d.log' % (self._key(task), attempt)), 'w')
        cmds = self._commands(task)
        self._log('START %s attempt %d (free %d GB)' % (task, attempt, self._freeGb()))
        p = subprocess.Popen(cmds[0], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             cwd=str(self.root / 'out'), start_new_session=True)
        return dict(task=task, attempt=attempt, cmds=cmds, i=0, proc=p, log=log, start=time.time(),
                    peak=0, seen={}, cpu=0.0)

    def _kill(self, r: dict, grace: float = 30.0) -> None:
        """End a running task and everything it started: each task runs in its own process group (session), which
        gets SIGTERM, then SIGKILL for whatever is left after the grace period.  A split's pool workers are in the
        group, so none of them is left holding memory after a stop."""

        pid = r['proc'].pid

        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return

        end = time.time() + grace

        while time.time() < end:

            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass

            try:
                os.killpg(pid, 0)
            except ProcessLookupError:
                self._log('ended %s (process group %d)' % (r['task'], pid))
                return

            time.sleep(1)

        try:
            os.killpg(pid, signal.SIGKILL)
            self._log('killed %s (process group %d) after %d s' % (r['task'], pid, grace))
        except ProcessLookupError:
            pass

        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass

    def _poll(self, r: dict) -> bool:
        """Sample the task's process tree; start its next command when one ends; True when the task ended."""

        s = _sample(_tree(r['proc'].pid))
        r['peak'] = max(r['peak'], sum(v[0] for v in s.values()))

        for p, v in s.items():
            r['seen'][(r['i'], p)] = v

        # wait4: the exit status and the CPU of the command, including the descendants it reaped.
        pid, status, ru = os.wait4(r['proc'].pid, os.WNOHANG)

        if pid == 0:
            return False

        rc = os.waitstatus_to_exitcode(status)
        r['proc'].returncode = rc
        r['cpu'] += ru.ru_utime + ru.ru_stime

        if rc == 0 and r['i'] + 1 < len(r['cmds']):
            r['i'] += 1
            r['proc'] = subprocess.Popen(r['cmds'][r['i']], stdout=r['log'], stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, cwd=str(self.root / 'out'), start_new_session=True)
            return False

        r['rc'] = rc
        r['log'].close()
        return True

    def _record(self, r: dict) -> None:

        end = time.time()
        v = list(r['seen'].values())
        row = dict(task=r['task'], host=self.host, attempt=r['attempt'],
                   start=time.strftime('%F %T', time.localtime(r['start'])),
                   end=time.strftime('%F %T', time.localtime(end)), seconds=round(end - r['start'], 1),
                   cpu_seconds=round(r['cpu'], 1), peak_rss_gb=round(r['peak'] / 2 ** 30, 2),
                   read_gb=round(sum(x[1] for x in v) / 2 ** 30, 2), write_gb=round(sum(x[2] for x in v) / 2 ** 30, 2),
                   rc=r['rc'])
        f = self.timing / 'tasks.csv'
        new = not f.exists()

        with open(f, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=TIMING_COLUMNS, lineterminator='\n')
            if new:
                w.writeheader()
            w.writerow(row)

    @staticmethod
    def _freeGb() -> int:

        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable'):
                    return int(line.split()[1]) // 2 ** 20

        return 0

    # ------------------------------------------------------------------------
    # clean-up and backlog
    # ------------------------------------------------------------------------
    def _readers(self, tile: str) -> list:
        """The tasks that read a tile's metrics."""

        k = 'metrics:' + tile
        return [t for t in self.tasks if k in self.deps[t]]

    def _cleanupLog(self, msg: str) -> None:

        with open(self.st / 'cleanup.log', 'a') as f:
            f.write('[' + time.strftime('%F %T') + '] ' + self.host + ' ' + msg + '\n')

    def _cleanup(self) -> None:
        """Delete the intermediates of every tile whose readers are all done (once per tile, over all nodes)."""

        if self.cleanupMode == 'off':
            return

        rootReal = self.root.resolve()

        for k in self.tasks:

            if self.kind[k] != 'metrics':
                continue

            tile = k.split(':', 1)[1]

            if tile in self._cleaned or tile in self.keepTiles or not self._done(k):
                continue

            readers = self._readers(tile)

            if not readers or not all(self._done(t) for t in readers):
                continue

            self._cleaned.add(tile)

            try:
                (self.st / 'claims' / ('cleanup_' + tile)).mkdir()
            except FileExistsError:
                continue

            freed, gone = 0, []

            for base, name in [(b, n) for b in (self.metricsRoot, self.cloudFallbackRoot)
                               for n in ('1-Days', '2-Composites', '3-Metrics')]:

                d = base / tile / str(self.year) / name

                if base == self.cloudFallbackRoot:
                    name = 'cloudfallback/' + name

                if not d.exists() and not d.is_symlink():
                    continue

                if d.is_symlink() or not d.is_dir() or rootReal not in d.resolve().parents:
                    self._cleanupLog('REFUSED %s %s (a link, not a folder, or outside the run root)' % (tile, d))
                    continue

                size = 0

                for dirpath, dirnames, filenames in os.walk(d):
                    for f in filenames:
                        try:
                            size += os.lstat(os.path.join(dirpath, f)).st_size
                        except OSError:
                            pass

                shutil.rmtree(d)
                freed += size
                gone.append(name)

            self._cleanupLog('%s deleted %s, %.2f GB (%d bytes)' % (tile, ' '.join(gone) or 'nothing', freed / 2 ** 30,
                                                                     freed))
            (self.st / ('cleanup_' + tile + '.done')).write_text('%s %.2f GB\n' % (time.strftime('%F %T'),
                                                                                 freed / 2 ** 30))

    def _backlog(self) -> int:
        """Map-only tiles whose metrics are claimed or done while their map is not finished."""

        n = 0

        for k in self.tasks:

            if self.kind[k] != 'map':
                continue

            tile = k.split(':', 1)[1]
            m = 'metrics:' + tile

            if tile in self.trainingTiles or m not in self.kind or self._finished(k):
                continue

            if self._done(m) or (self.st / 'claims' / self._key(m)).exists():
                n += 1

        return n

    # ------------------------------------------------------------------------
    # run (one node's share)
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    # checkFixedModels
    # ------------------------------------------------------------------------
    def checkFixedModels(self) -> None:
        """A ready-made model may name a SHA-256 file ("fixed": {..., "sha256": <file>}), in the format of sha256sum: lines
        '<hash>  <path>', paths relative to that file's folder.  Every file of the model (a compact forest folder or a joblib
        file) and its top-N file must be listed, and every listed file must match.  Checked once, before the first task; fatal.
        """
        fixed = [(n, m['fixed']) for n, m in list(self.models.items()) +
                 ([('cloudFallbackTree', self.cloudFallbackTree)] if self.cloudFallbackTree else []) if 'fixed' in m]

        for name, f in fixed:

            if 'sha256' not in f:
                continue

            sums = Path(f['sha256'])
            listed = {}

            for line in sums.read_text().split('\n'):
                if line.strip():
                    h, p = line.split(None, 1)
                    listed[(sums.parent / p.strip().lstrip('*')).resolve()] = h

            model = Path(f['model'])
            need = [p.resolve() for p in (sorted(model.iterdir()) if model.is_dir() else [model])] + [Path(f['topN']).resolve()]
            missing = [str(p) for p in need if p not in listed]

            if missing:
                raise RuntimeError('Model %s: %s does not list %s' % (name, sums, ', '.join(missing)))

            for p, h in listed.items():
                if self._sha256(p) != h:
                    raise RuntimeError('Model %s: %s does not match its SHA-256 in %s' % (name, p, sums))

            self._log('model %s: SHA-256 of %d files match %s' % (name, len(listed), sums))

    @staticmethod
    def _sha256(path: Path) -> str:

        h = hashlib.sha256()

        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(1 << 24), b''):
                h.update(block)

        return h.hexdigest()

    def run(self) -> int:
        """Drive the year on this node until every task is finished or the run stops: check fixed models, then repeatedly claim
        tasks whose dependencies are done (up to this host's slots, when its free memory allows), start them, record their results
        and timing, retry a failed map once, and clean up finished tiles.  Returns 1 when the run stops (a fatal failure or a STOP
        marker), 0 when every task has finished (a map that failed twice is recorded, not fatal).
        """
        self.checkFixedModels()
        slots = 1

        for prefix, n in self.spec.get('slots', {}).items():
            if self.host.startswith(prefix):
                slots = int(n)

        mem = self.spec.get('memGB', {})
        fitHosts = self.spec.get('fitHosts', [])
        running, attempts = [], {}
        self._log('=== runYear START pid=%d host=%s slots=%d tasks=%d ===' % (os.getpid(), self.host, slots,
                                                                             len(self.tasks)))

        while True:

            for r in [r for r in running if self._poll(r)]:

                running.remove(r)
                self._record(r)
                t = r['task']

                if r['rc'] != 0 and self.kind[t] == 'map' and r['attempt'] == 1 and not self._fatal():
                    aside = self.root / 'Python' / 'failed_attempts' / self._key(t)
                    aside.mkdir(parents=True, exist_ok=True)
                    tile = t.split(':')[1]
                    maps = self.root / 'out' / 'maps'
                    for f in list(maps.glob('*-' + tile + '-%d*' % self.year)) + \
                            list(maps.glob('MOD44B.A%d065.%s.*.h5*' % (self.year, tile))):
                        shutil.move(str(f), str(aside / f.name))
                    running.append(self._start(t, 2))
                    continue

                self._mark(t, r['rc'], 'log ' + str(self.logs / ('%s_a%d.log' % (self._key(t), r['attempt']))))
                self._log(('OK ' if r['rc'] == 0 else 'FAILED ') + t)

            self._cleanup()
            stop = self._fatal()

            if stop:
                for r in running:
                    self._kill(r)
                self._log('stopped: ' + stop)
                return 1

            if all(self._finished(t) for t in self.tasks):
                break

            for t in self.tasks:

                if len(running) >= slots:
                    break

                if self._finished(t) or (self.st / 'claims' / self._key(t)).exists():
                    continue

                if self.kind[t] == 'fit' and fitHosts and self.host not in fitHosts:
                    continue

                if not all(self._done(d) for d in self.deps[t]):
                    continue

                if self.maxBacklog is not None and self.kind[t] == 'metrics' and \
                   t.split(':', 1)[1] not in self.trainingTiles and self._backlog() >= int(self.maxBacklog):
                    continue

                if self._freeGb() < int(mem.get(self.kind[t], 40)):
                    continue

                try:
                    (self.st / 'claims' / self._key(t)).mkdir()
                except FileExistsError:
                    continue

                (self.st / 'claims' / self._key(t) / 'host').write_text(self.host + '\n')
                running.append(self._start(t, 1))

            time.sleep(5)

        self._log('all tasks finished')

        if not (self.st / 'ALL.done').exists():
            failed = [t for t in self.tasks if self._failed(t)]
            (self.st / 'ALL.done').write_text(time.strftime('%F %T') + (' failed: ' + ' '.join(failed) if failed else '')
                                              + '\n')
        return 0
