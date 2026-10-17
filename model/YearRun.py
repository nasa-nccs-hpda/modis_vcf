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
#   map:<tile>       vcfPredict, one step (blend A + BP10 inside)            [metrics:<tile>, fit:tree, fit:bare]
# A model can instead be given ready-made (spec "models"/<name>/"fixed"); it then has no split or fit task.
# metrics, split and fit failures are fatal (they break the year); a map is retried once, then recorded.
# Every attempt appends one row to <runRoot>/timing/tasks.csv: task, host, start, end, seconds, CPU seconds
# (user + system, from wait4: the commands and the descendants they reaped), peak RSS of the task's process tree
# and bytes read / written (/proc/<pid>/io rchar / wchar: through read and write calls, incl. page-cache hits), both sampled every 5 s (short-lived processes may be undercounted), return code.  The partner's V4r metrics and the cloud counts are inputs
# (blend/v4r_year.py); computing them in the run is S7.
#
# Spec (JSON): year, runRoot, lowSun, models {tree, bare: {manifest, predictors, trainingType, fit {...}} or
# {fixed: {model, topN}}}, partner (optional) {model, topN, metricsDir, counts, tiles}, mapTiles,
# metricsTiles (optional; default: the manifests' tiles and mapTiles), workers {metrics, split, mapJobs},
# fitHosts, slots {host prefix: n}, memGB {task type: GB}.
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
        self.tasks, self.deps, self.kind = self._taskList()

    # ------------------------------------------------------------------------
    # manifest tiles
    # ------------------------------------------------------------------------
    def _manifestTiles(self, name: str) -> list:

        meta = json.loads(Path(self.models[name]['manifest'] + '.json').read_text())
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

        if mt is None:
            mt = sorted(set(mapTiles).union(*[self._manifestTiles(m) for m in fitted]))

        tasks, deps, kind = [], {}, {}

        for t in mt:
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
            deps[k] = (['metrics:' + t] if t in mt else []) + ['fit:' + m for m in fitted]
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

        m = self.models[name]

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

        m = self.models.get(arg, {})

        if kind == 'split':
            return [py + ['-m', 'modis_vcf.view.trainingManifest', 'build', '--manifest', m['manifest'],
                          '--predictors', m['predictors'], '--root', str(self.metricsRoot), '-y', str(self.year),
                          '-o', str(self.root / 'Python' / 'splits' / arg), '--workers', str(w.get('split', 6))] + ls]

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

            if self.partner and arg in self.partner['tiles']:
                cmd += ['--partner', self.partner['model'], '--partnerTopN', self.partner['topN'],
                        '--partnerMetricsDir', self.partner['metricsDir'], '--counts', self.partner['counts']]

            if self.spec.get('lowSunCheck') == 'off':
                cmd += ['--lowSunCheck', 'off']

            return [cmd]

        raise ValueError(task)

    # ------------------------------------------------------------------------
    # metrics task (in a subprocess of the driver)
    # ------------------------------------------------------------------------
    def warmMetrics(self, tile: str) -> None:
        """Every predictor of every model, cached in the run's metrics root (one Metrics, one tile)."""

        import pickle
        from modis_vcf.model import LowSun
        from modis_vcf.model.Metrics import Metrics
        from modis_vcf.model.ProductTypeMod44 import ProductTypeMod44
        from modis_vcf.model.VcfPredict import MOD44_DIR

        LowSun.configure(self.spec.get('lowSun', 'le0'))
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
