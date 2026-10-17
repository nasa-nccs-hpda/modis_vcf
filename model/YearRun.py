import csv
import json
import logging
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time


# ----------------------------------------------------------------------------
# YearRun
#
# One driver per year (v1.4.1 P5): metrics -> splits -> fits -> maps, as tasks
# shared by every node of a run.  Each node runs YearRun(spec).run() (through
# view/runYear.py, started by the run's detached lane script); the nodes share
# one task list through marker files in <runRoot>/status:
#   claims/<task>/  taken (mkdir, so a task runs once)
#   <task>.done / <task>.failed
#   STOP, STOP_<host>, fatal_<task>.failed   stop this node or every node
# Tasks (dependencies in brackets):
#   metrics:<tile>   the predictor metrics of every model, one tile          []
#   split:<model>    TrainingManifest.build (S5: tiles in parallel)          [metrics of the manifest's tiles]
#   fit:<model>      fitAltModel.py, then the compact copy (CompactForest)   [split:<model>]
#   map:<tile>       vcfPredict, one step (blend A + BP10 inside)            [metrics:<tile>, fit:tree, fit:bare, fit:partner]
#                    then, on blend tiles with V4r built in the run, the V4r statistics (model/V4r.py compose)
# A model can instead be given ready-made (spec "models"/<name>/"fixed"); it then has no split or fit task.
# Blend A partner (spec "partner"; S7 / D4): V4r composites, counts and V4r metrics are built in metrics:<tile> (model/V4r.py,
# low-sun mode off) for the partner's blend tiles and, when the partner is fitted, its training tiles; split:partner builds the
# partner's training rows from the V4r root, fit:partner fits it.
# metrics, split and fit failures are fatal (they break the year); a map is retried once, then recorded.
# Every attempt appends one row to <runRoot>/timing/tasks.csv: task, host, start, end, seconds, CPU seconds
# (user + system, from wait4: the commands and the descendants they reaped), peak RSS of the task's process tree
# and bytes read / written (/proc/<pid>/io rchar / wchar: through read and write calls, incl. page-cache hits), both sampled every 5 s (short-lived processes may be undercounted), return code.  The partner's V4r metrics and the cloud counts are inputs
# (blend/v4r_year.py) only in the "borrowed" partner form below.
#
# Spec (JSON): year, runRoot, lowSun, models {tree, bare: {manifest, predictors, trainingType, fit {...}} or
# {fixed: {model, topN}}}, partner (optional): {tiles, manifest, predictors, trainingType, fit} (fitted in the run),
# {tiles, fixed: {model, topN}} (fixed model, V4r built in the run) or {tiles, model, topN, metricsDir, counts} (borrowed: no V4r
# built), mapTiles,
# metricsTiles (optional; default: the manifests' tiles and mapTiles), workers {metrics, split, mapJobs},
# fitHosts, slots {host prefix: n}, memGB {task type: GB},
# cleanup (optional) {mode: after_map (default) | off, keepTiles: [...]}, maxBacklog (optional),
# writeDailiesTiles (optional: tiles whose daily inputs are also written to 1-Days; by default none are).
#
# Clean-up: once every task that reads a tile's metrics is done (its map, and the splits of the manifests
# it is in), that tile's 1-Days, 2-Composites and 3-Metrics are deleted, unless the tile is in keepTiles.
# A tile whose map failed is kept.  Only real folders inside the run's metrics root are deleted (a
# symbolic link or a path outside runRoot is refused); the low-sun stamp and counts stay.  Each deletion
# is logged to status/cleanup.log with the GB freed.  The metrics of training tiles are claimed first, and
# maxBacklog holds new metrics claims of map-only tiles while that many of them wait for their map.
# ----------------------------------------------------------------------------
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
        self.partner = self.spec.get('partner')
        self.partnerMode = None if not self.partner else 'borrowed' if 'metricsDir' in self.partner else \
            'fixed' if 'fixed' in self.partner else 'fit'
        self.v4rRoot = self.root / 'Python' / 'v4r'
        self.countsDir = self.root / 'out' / 'v4r_counts'
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
        return self.partner if name == 'partner' else self.models[name]

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
        mapTiles = list(self.spec['mapTiles'])
        mt = self.spec.get('metricsTiles')

        if self.partnerMode == 'fit':
            fitted.append('partner')

        self.trainingTiles = set().union(*[self._manifestTiles(m) for m in fitted])
        partnerTiles = set(self.partner['tiles']) if self.partner else set()
        self.v4rTiles = set() if self.partnerMode in (None, 'borrowed') else \
            (partnerTiles & set(mapTiles)) | (set(self._manifestTiles('partner')) if self.partnerMode == 'fit' else set())

        if mt is None:
            mt = sorted(set(mapTiles) | self.trainingTiles | self.v4rTiles)

        tasks, deps, kind = [], {}, {}

        # Training tiles first, so the splits and fits can start while the map-only tiles are computed.
        for t in sorted(mt, key=lambda t: t not in self.trainingTiles):
            k = 'metrics:' + t
            tasks.append(k); deps[k] = []; kind[k] = 'metrics'

        for m in fitted:
            k = 'split:' + m
            tasks.append(k); deps[k] = ['metrics:' + t for t in self._manifestTiles(m)]; kind[k] = 'split'
            k = 'fit:' + m
            tasks.append(k); deps[k] = ['split:' + m]; kind[k] = 'fit'

        for t in mapTiles:
            k = 'map:' + t
            tasks.append(k)
            deps[k] = (['metrics:' + t] if t in mt else []) + \
                ['fit:' + m for m in fitted if m != 'partner' or t in partnerTiles]
            kind[k] = 'map'

        return tasks, deps, kind

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

        if name == 'partner' and self.partnerMode == 'borrowed':
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
            # The partner's rows come from the V4r root, built in mode off (Q3).
            root, sls = (self.v4rRoot, ['--lowSun', 'off']) if arg == 'partner' else (self.metricsRoot, ls)
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

        if kind == 'map':
            tc, tcn = self.modelPaths('tree')
            nv, nvn = self.modelPaths('bare')
            cmd = py + ['-m', 'modis_vcf.view.vcfPredict', '-o', str(self.root / 'out' / 'maps'),
                        '--metricsDir', str(self.metricsRoot), '-t', arg, '--tc', tc, '--tcn', tcn, '--nv', nv,
                        '--nvn', nvn, '-y', str(self.year), '--nJobs', str(w.get('mapJobs', 8)), '--tcAgg', 'median',
                        '--nvAgg', 'median'] + ls

            blend = self.partner and arg in self.partner['tiles']

            if blend:
                pm, pn = self.modelPaths('partner')
                borrowed = self.partnerMode == 'borrowed'
                cmd += ['--partner', pm, '--partnerTopN', pn,
                        '--partnerMetricsDir', self.partner['metricsDir'] if borrowed else str(self.v4rRoot),
                        '--counts', self.partner['counts'] if borrowed else str(self.countsDir)]

            if self.spec.get('lowSunCheck') == 'off':
                cmd += ['--lowSunCheck', 'off']

            cmds = [cmd]

            if blend and arg in self.v4rTiles:
                cmds.append(py + ['-m', 'modis_vcf.model.V4r', 'compose', arg, str(self.year), str(self.v4rRoot),
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

        if tile not in self.v4rTiles:
            return

        # V4r composites and counts (S7), then the partner's predictors on the V4r root (mode off).
        from modis_vcf.model import V4r
        done = self.countsDir / ('%s_v4r_build.json' % tile)
        comp = self.v4rRoot / tile / str(self.year) / '2-Composites'

        if not done.exists():
            if comp.exists() and any(comp.iterdir()):
                # An interrupted attempt (e.g. a guard kill): move its partial V4r composites aside, then rebuild.
                aside = self.root / 'Python' / 'failed_attempts' / ('v4r_%s_%d' % (tile, int(time.time())))
                aside.mkdir(parents=True)
                shutil.move(str(comp), str(aside / '2-Composites'))
            V4r.buildTile(tile, self.year, self.metricsRoot, self.v4rRoot, self.countsDir,
                          int(self.spec.get('workers', {}).get('v4r', 8)), self.logger)

        p = self.partner

        if 'fixed' in p:
            with open(p['fixed']['topN'], 'rb') as f:
                pnames = list(pickle.load(f))
        else:
            pnames = Path(p['predictors']).read_text().split()

        vm = Metrics(tile, self.year, ProductTypeMod44(MOD44_DIR), self.v4rRoot, self.logger, lowSun='off')

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
                             cwd=str(self.root / 'out'))
        return dict(task=task, attempt=attempt, cmds=cmds, i=0, proc=p, log=log, start=time.time(),
                    peak=0, seen={}, cpu=0.0)

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
                                         stdin=subprocess.DEVNULL, cwd=str(self.root / 'out'))
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

            for base, name in [(b, n) for b in (self.metricsRoot, self.v4rRoot)
                               for n in ('1-Days', '2-Composites', '3-Metrics')]:

                d = base / tile / str(self.year) / name

                if base == self.v4rRoot:
                    name = 'v4r/' + name

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
    def run(self) -> int:

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
                    for f in (self.root / 'out' / 'maps').glob('*-' + t.split(':')[1] + '-%d*' % self.year):
                        shutil.move(str(f), str(aside / f.name))
                    running.append(self._start(t, 2))
                    continue

                self._mark(t, r['rc'], 'log ' + str(self.logs / ('%s_a%d.log' % (self._key(t), r['attempt']))))
                self._log(('OK ' if r['rc'] == 0 else 'FAILED ') + t)

            self._cleanup()
            stop = self._fatal()

            if stop:
                for r in running:
                    r['proc'].terminate()
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
