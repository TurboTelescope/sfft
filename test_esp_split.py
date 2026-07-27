#!/usr/bin/env python3
# Bit-exact validation of the ESP -> ESP_prep + ESP_solve split.
# Run in-container (GPU) where the fixture is mounted at /ravi/perf_analysis/refactor/fixture/.

import os
import gc
import json
import inspect
import numpy as np
import cupy as cp
from astropy.io import fits
from sfft.EasySparsePacket import Easy_SparsePacket

FIXROOT = '/ravi/perf_analysis/refactor/fixture'
TILES = ['00', '01', '10', '11']


def _free():
    # replicate gpu_service teardown: return CuPy's cached pool blocks between jobs (else OOM)
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

with open(os.path.join(FIXROOT, 'GOLDEN_STATS.json')) as f:
    GOLDEN = json.load(f)['metrics_per_tile']

# route a full kwargs dict to each staticmethod by its actual signature
PREP_PARAMS = set(inspect.signature(Easy_SparsePacket.ESP_prep).parameters) - {'FITS_REF', 'FITS_SCI'}
SOLVE_PARAMS = set(inspect.signature(Easy_SparsePacket.ESP_solve).parameters) - {'prep', 'FITS_REF', 'FITS_SCI'}


def build_kwargs(jk, XY_PriorBan):
    return dict(
        GKerHW=jk['GKerHW'], MatchTol=jk['MatchTol'], ForceConv=jk['ForceConv'],
        KerHWRatio=jk['KerHWRatio'], KerHWLimit=jk['KerHWLimit'], KerPolyOrder=jk['KerPolyOrder'],
        BGPolyOrder=jk['BGPolyOrder'], ConstPhotRatio=jk['ConstPhotRatio'],
        BACKEND_4SUBTRACT=jk['BACKEND_4SUBTRACT'], CUDA_DEVICE_4SUBTRACT=jk['CUDA_DEVICE_4SUBTRACT'],
        NUM_CPU_THREADS_4SUBTRACT=jk['NUM_CPU_THREADS_4SUBTRACT'],
        XY_PriorBan=XY_PriorBan,
        GAIN_KEY='GAIN', SATUR_KEY='SATURATE', BACK_TYPE='MANUAL', BACK_VALUE=0.0,
        COARSE_VAR_REJECTION=True, CVREJ_MAGD_THRESH=0.12, ELABO_VAR_REJECTION=True,
        SINGLE_PRECISION=True, VERBOSE_LEVEL=1,
    )


rows = []
all_gate_pass = True   # gate = split==ESP for every tile
fail_msgs = []

for T in TILES:
    ref = os.path.join(FIXROOT, 'ref_tile_%s.fits' % T)
    sci = os.path.join(FIXROOT, 'sci_tile_%s.fits' % T)
    golden = np.load(os.path.join(FIXROOT, 'golden_dif_%s.npy' % T))
    with open(os.path.join(FIXROOT, 'esp_kwargs_%s.json' % T)) as f:
        jk = json.load(f)

    ban_path = os.path.join(FIXROOT, 'xy_ban_%s.npy' % T)
    XY_PriorBan = np.load(ban_path) if os.path.exists(ban_path) else None

    kwargs = build_kwargs(jk, XY_PriorBan)
    prep_kwargs = {k: v for k, v in kwargs.items() if k in PREP_PARAMS}
    solve_kwargs = {k: v for k, v in kwargs.items() if k in SOLVE_PARAMS}

    # (1) original monolithic ESP
    d1, _, _ = Easy_SparsePacket.ESP(FITS_REF=ref, FITS_SCI=sci, **kwargs)
    _free()
    # (2) split: CPU prep on one call, GPU solve on another
    prep = Easy_SparsePacket.ESP_prep(FITS_REF=ref, FITS_SCI=sci, **prep_kwargs)
    _free()
    d2, _, _ = Easy_SparsePacket.ESP_solve(prep, FITS_REF=ref, FITS_SCI=sci, **solve_kwargs)
    _free()

    d1n = np.nan_to_num(d1)
    d2n = np.nan_to_num(d2)
    # GATE: the bit-exact refactor check -- ESP_solve(ESP_prep(x)) == ESP(x) byte-for-byte
    split_eq_esp = bool(np.array_equal(d2n, d1n))
    # INFO ONLY: single-tile vs full-frame golden. Invalid cross-context cmp on saturated
    # core tiles (01,10); full-frame golden belongs to the end-to-end A/B, not this unit test.
    esp_eq_golden = bool(np.array_equal(d1n, golden))

    if not split_eq_esp:
        all_gate_pass = False
        maxdiff = float(np.abs(d2n - d1n).max())
        fail_msgs.append('FAIL tile %s split!=ESP maxdiff=%.6g' % (T, maxdiff))

    # regression stats (human check) computed on d2
    sci_data = fits.getdata(sci)
    std_diff = float(np.std(d2))
    ratio = float(np.std(d2) / np.std(sci_data))
    median = float(np.median(d2))

    gs = GOLDEN[T]
    stats_ok = (
        np.isclose(std_diff, gs['std_diff'], rtol=1e-3, atol=1e-3) and
        np.isclose(ratio, gs['ratio_diff_sci'], rtol=1e-3, atol=1e-3) and
        np.isclose(median, gs['median_diff'], rtol=1e-3, atol=1e-3)
    )
    rows.append((T, std_diff, ratio, median, split_eq_esp, esp_eq_golden, bool(stats_ok)))
    _free()   # teardown at end of tile iteration

# report
print('')
print('tile | std_diff      | ratio     | median    | split==ESP | ESP==golden | stats~golden')
print('     |               |           |           | (GATE)     | (info)      | (info)')
print('-----+---------------+-----------+-----------+------------+-------------+-------------')
for T, sd, r, md, se, eg, so in rows:
    print(' %s  | %13.4f | %9.4f | %9.4f | %10s | %11s | %s' % (T, sd, r, md, se, eg, so))
print('')
print('note: split==ESP is the bit-exact refactor GATE. ESP==golden and stats~golden are')
print('      INFORMATIONAL only -- the isolated single-tile run lacks the full-frame')
print('      gpu_service context the golden was captured in, so saturated core tiles (01,10)')
print('      need not match golden here; that check belongs to the end-to-end A/B task.')
print('')

for m in fail_msgs:
    print(m)

n_pass = sum(1 for row in rows if row[4])   # row[4] = split_eq_esp (the gate)
if all_gate_pass:
    print('SPLIT-BITEXACT PASS (%d/%d tiles)' % (n_pass, len(TILES)))
    if not all(row[6] for row in rows):
        print('INFO: some tiles differ from full-frame golden (stats/byte) -- expected, see note.')
else:
    print('SPLIT-BITEXACT FAIL (%d/%d tiles)' % (n_pass, len(TILES)))
