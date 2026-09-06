#!/usr/bin/env python3
"""
Classify nozomi diffusion-MRI substrates into Set 1 / Set 2 / Set 3 and
(optionally) gather each substrate's FULL folder into data_full/.

Read-only over source data. Originals are never moved/deleted; --copy only
duplicates full substrate folders into data_full/set1|set2|set3.

Rules (agreed with user, 2026-08-26)
------------------------------------
Filters (ALL sets): keep only substrates with >= 500 fibers.
Diameter: use d2.58 as generated; EXCLUDE any substrate generated with d=2.5.
  Grid = d{1.68, 2.58, 3.5, 4.5} x K{2,4,7,10,20,200}.

Set 1 (Healthy WM): six Set1/3 folders, bead == 0.3 ONLY.
  - K in {10,20,200}: keep achieved avf in [0.55, 0.75].
  - K in {2,4,7}: keep as-achieved VF (high dispersion packs to lower VF).
Set 3 (TBI pathology): six Set1/3 folders, bead in {0.3,0.5,0.83,1.24}, all d/K/VF.
  (Set 1 is a subset of Set 3.)
Set 2 (Axonal loss): full dataVF03 (VF30 arm) + data (VF70 arm) folders, matched
  on (d,K). Arm is defined by source folder; baseline-beading reproducibility
  runs, so no avf-band clip — only >=500 fibers + in-grid diameter apply.

Usage:
  python3 classify_substrates.py            # analyze + print coverage (no copy)
  python3 classify_substrates.py --copy     # also wipe & rebuild data_full/
"""
import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

WS = Path('/home/nguyt16@ds.vanderbilt.edu')
DATA_FULL = WS / 'nozomi/experiment/aim2-prep/data_full'

SET2_VF30_FOLDER = 'nozomi/experiment/visualization/dataVF03'
SET2_VF70_FOLDER = 'nozomi/experiment/visualization/data'
SET13_FOLDERS = [
    'nozomi/experiment/result/2026-03-22-bead03',
    'nozomi/experiment/result/2026-03-22_bead_05',
    'nozomi/experiment/result/2026-03-18_bead_083_simulated',
    'nozomi/experiment/result/2026-03-22_bead_1.24',
    'nozomi/experiment/aim2-prep/data-high-dispersion/highdisp_20260727_140957',
    'nozomi/experiment/aim2-prep/data-high-dispersion/highdisp_20260727_140958',
]

DIAMS = [1.68, 2.58, 3.5, 4.5]
KAPPAS = [2, 4, 7, 10, 20, 200]
BEADS = [0.3, 0.5, 0.83, 1.24]
K_COHERENT = {10, 20, 200}      # pack to high VF
K_DISPERSED = {2, 4, 7}         # pack to lower VF
MIN_FIBERS = 500
VF70_LO, VF70_HI = 0.55, 0.75    # Set 1 coherent-K healthy band
DIAM_TOL = 0.02                  # match a designed diameter within this tolerance

# Incremental high-dispersion add (--add-highdisp): per-K achieved-VF acceptance
HIGHDISP_FOLDERS = [
    'nozomi/experiment/aim2-prep/data-high-dispersion/highdisp_20260727_140957',
    'nozomi/experiment/aim2-prep/data-high-dispersion/highdisp_20260727_140958',
]
K_VF_TARGET = {7: 0.5, 4: 0.4, 2: 0.3}   # target achieved VF per dispersed K
K_VF_TOL = {2: 0.10, 4: 0.10, 7: 0.04}   # per-K achieved-VF acceptance half-width
MIN_CONVERGED_PER_CONFIG = 3             # >=3 converged in-band substrates per (d,K,bead)


# ---------------------------------------------------------------- parsing ----
def find_pkls(folder):
    p = WS / folder
    if not p.exists():
        return None
    r = subprocess.run(
        ['find', str(p), '-name', 'array*_fibers_*.pkl'],
        capture_output=True, text=True,
    )
    return [x for x in r.stdout.strip().split('\n') if x]


def _f(pattern, text):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def parse_nfib(name):
    m = re.search(r'array(\d+)_fibers', name)
    return int(m.group(1)) if m else None


def parse_avf(name):
    return _f(r'avf_(\d+\.\d+)', name)


def parse_d(name):
    return _f(r'_d(\d+\.\d+)_', name)


def parse_K(name):
    m = re.search(r'_K(\d+)_', name)
    return int(m.group(1)) if m else None


def parse_odi(name):
    return _f(r'ODI_(\d+\.\d+)', name)


def parse_bead(full_path):
    """Beading from the folder chain. Returns float or None (baseline)."""
    # Primary: explicit decimal form e.g. bead_0.83 / bead0.3
    m = re.search(r'bead_?(\d+\.\d+)', full_path)
    if m:
        return float(m.group(1))
    # Fallback: compact integer forms bead03->0.3, bead05->0.5, bead083->0.83
    m = re.search(r'bead_?0(\d{1,3})(?!\d)', full_path)
    if m:
        digits = m.group(1)
        return float('0.' + digits) if len(digits) <= 2 else float('0.' + digits)
    return None


def get_cv(pkl_path):
    stats = Path(pkl_path).parent.parent / 'figs' / 'substrate_stats'
    if not stats.exists():
        return None
    for png in stats.glob('*CV_outer_diameter*CVmean_*.png'):
        v = _f(r'CVmean_(\d+\.\d+)', png.name)
        if v is not None:
            return v
    return None


def match_diam(d):
    """Return the designed diameter this d belongs to, or None. d=2.5 -> None."""
    if d is None:
        return None
    for dd in DIAMS:
        if abs(d - dd) <= DIAM_TOL:
            return dd
    return None


def substrate_folder(pkl_path):
    p = Path(pkl_path)
    return p.parent.parent if p.parent.name == 'data' else p.parent


def in_vf_band(rec):
    t = K_VF_TARGET.get(rec.get('K'))
    tol = K_VF_TOL.get(rec.get('K'))
    if t is None or tol is None or rec.get('avf') is None:
        return False
    return (t - tol) <= rec['avf'] <= (t + tol)


def has_diffusion_sim(pkl_path):
    sim = substrate_folder(pkl_path) / 'sim'
    if not sim.exists():
        return False
    intra = any(sim.glob('difftime_*diffcoeff_intra_*.png'))
    extra = any(sim.glob('difftime_*diffcoeff_extra_*.png'))
    return bool(intra and extra)


def highdisp_converged_records():
    """Newest pkl per run folder across the 2 highdisp folders = realized avf."""
    best = {}
    for folder in HIGHDISP_FOLDERS:
        for p in find_pkls(folder) or []:
            rf = substrate_folder(p)
            mt = Path(p).stat().st_mtime
            if rf not in best or mt > best[rf][1]:
                best[rf] = (p, mt)
    recs = [make_record(p) for p, _ in best.values()]
    return [r for r in recs if r['K'] in K_VF_TARGET and r['d'] is not None
            and r['bead'] in BEADS and r['n_fibers'] and r['n_fibers'] >= MIN_FIBERS
            and r['avf'] is not None]


def make_record(pkl_path, arm=None):
    name = Path(pkl_path).name
    d_raw = parse_d(name)
    return {
        'path': str(Path(pkl_path).relative_to(WS)),
        'abs_path': str(pkl_path),
        'd_raw': d_raw,
        'd': match_diam(d_raw),
        'K': parse_K(name),
        'avf': parse_avf(name),
        'ODI': parse_odi(name),
        'bead': parse_bead(str(pkl_path)),
        'n_fibers': parse_nfib(name),
        'CV': get_cv(pkl_path),
        'arm': arm,
    }


# --------------------------------------------------------------- classify ----
def collect(folder, arm=None):
    pkls = find_pkls(folder)
    if pkls is None:
        print(f"  !! MISSING FOLDER: {folder}", file=sys.stderr)
        return []
    seen, recs = set(), []
    for p in pkls:
        if p in seen:
            continue
        seen.add(p)
        recs.append(make_record(p, arm=arm))
    return recs


def classify():
    dropped_small, excluded_offgrid = [], []

    # ---- Set 1 & Set 3 sources ------------------------------------------
    set13 = []
    for folder in SET13_FOLDERS:
        set13 += collect(folder)

    set1, set3 = [], []
    for r in set13:
        if r['n_fibers'] is None or r['n_fibers'] < MIN_FIBERS:
            dropped_small.append(r); continue
        if r['d'] is None:
            excluded_offgrid.append(r); continue
        if r['bead'] not in BEADS:
            continue
        # Set 3: coherent-K all VF; dispersed-K only within achieved-VF band
        if r['K'] in K_COHERENT or in_vf_band(r):
            set3.append(r)
        # Set 1: bead 0.3 healthy baseline
        if abs(r['bead'] - 0.3) < 1e-9:
            if r['K'] in K_COHERENT:
                if r['avf'] is not None and VF70_LO <= r['avf'] <= VF70_HI:
                    set1.append(r)
            elif in_vf_band(r):
                set1.append(r)

    # ---- Set 2 sources (VF30 / VF70 arms) -------------------------------
    # Arm is defined by source folder (dataVF03=VF30, data=VF70); these are
    # baseline-beading reproducibility runs, so use the FULL folders with only
    # the universal filters (>=500 fibers, in-grid diameter). No avf-band clip.
    vf30_raw = collect(SET2_VF30_FOLDER, arm='VF30')
    vf70_raw = collect(SET2_VF70_FOLDER, arm='VF70')

    def keep_arm(r):
        if r['n_fibers'] is None or r['n_fibers'] < MIN_FIBERS:
            dropped_small.append(r); return False
        if r['d'] is None:
            excluded_offgrid.append(r); return False
        return True

    vf30 = [r for r in vf30_raw if keep_arm(r)]
    vf70 = [r for r in vf70_raw if keep_arm(r)]

    cells30 = {(r['d'], r['K']) for r in vf30}
    cells70 = {(r['d'], r['K']) for r in vf70}
    paired_cells = cells30 & cells70
    set2 = [r for r in vf30 + vf70 if (r['d'], r['K']) in paired_cells]

    return {
        'set1': set1, 'set2': set2, 'set3': set3,
        'set2_vf30': vf30, 'set2_vf70': vf70, 'paired_cells': sorted(paired_cells),
        'dropped_small': dropped_small, 'excluded_offgrid': excluded_offgrid,
    }


# --------------------------------------------------------------- reports ----
def grid_beadfilter(recs, bead=None):
    grid = defaultdict(int)
    for r in recs:
        if bead is not None and (r['bead'] is None or abs(r['bead'] - bead) > 1e-9):
            continue
        grid[(r['d'], r['K'])] += 1
    return grid


def print_grid(title, cell_fn):
    print(f"\n{title}")
    head = "   d\\K |" + "".join(f"{k:>6}" for k in KAPPAS)
    print(head); print("-" * len(head))
    for d in DIAMS:
        row = f"{d:>6} |"
        for k in KAPPAS:
            row += f"{cell_fn(d, k):>6}"
        print(row)


def report(result):
    s1, s2, s3 = result['set1'], result['set2'], result['set3']
    print("=" * 72)
    print("SUBSTRATE CLASSIFICATION SUMMARY")
    print("=" * 72)
    print(f"Set 1 (Healthy, bead=0.3):     {len(s1):>4} substrates")
    print(f"Set 2 (Axonal loss VF30/70):   {len(s2):>4} substrates "
          f"(VF30={len(result['set2_vf30'])}, VF70={len(result['set2_vf70'])})")
    print(f"Set 3 (TBI, all beading):      {len(s3):>4} substrates")
    print(f"Dropped (<{MIN_FIBERS} fibers):        {len(result['dropped_small']):>4}")
    print(f"Excluded (off-grid d, e.g. 0.5/2.5): {len(result['excluded_offgrid']):>4}")

    # Set 1 coverage
    g1 = grid_beadfilter(s1)
    print_grid("SET 1 coverage (count per d x K, bead=0.3)",
               lambda d, k: g1.get((d, k), 0) or '.')
    cells1 = {(d, k) for (d, k), n in g1.items() if n}
    missing1 = [(d, k) for d in DIAMS for k in KAPPAS if (d, k) not in cells1]
    print(f"  cells covered: {len(cells1)}/24   missing: {missing1 if missing1 else 'none'}")

    # Set 2 coverage
    print("\nSET 2 pair coverage (P=pair, 30=VF30 only, 70=VF70 only, .=none)")
    c30 = {(r['d'], r['K']) for r in result['set2_vf30']}
    c70 = {(r['d'], r['K']) for r in result['set2_vf70']}

    def cell2(d, k):
        a, b = (d, k) in c30, (d, k) in c70
        return 'P' if a and b else '30' if a else '70' if b else '.'
    print_grid("  ", cell2)
    print(f"  complete pairs: {len(result['paired_cells'])}   cells: "
          f"{result['paired_cells']}")

    # Set 3 coverage per bead
    for b in BEADS:
        gb = grid_beadfilter(s3, bead=b)
        cells = {(d, k) for (d, k), n in gb.items() if n}
        print_grid(f"SET 3 coverage bead={b} (count per d x K)",
                   lambda d, k: gb.get((d, k), 0) or '.')
        print(f"  cells covered: {len(cells)}/24")


# ----------------------------------------------------------------- output ----
def clean_name(r, idx):
    d = r['d']; k = r['K']; b = r['bead']; avf = r['avf']
    bead_s = 'baseline' if b is None else f"{b}"
    avf_s = 'NA' if avf is None else f"{avf:.2f}"
    arm = f"_{r['arm']}" if r.get('arm') else ""
    return f"d{d}_K{k}_bead{bead_s}_avf{avf_s}{arm}_{idx:03d}"


def write_csv(path, recs):
    cols = ['d', 'd_raw', 'K', 'ODI', 'avf', 'bead', 'CV', 'n_fibers', 'arm', 'path']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in sorted(recs, key=lambda x: (x['d'] or 0, x['K'] or 0,
                                             x['bead'] or 0, x['avf'] or 0)):
            w.writerow(r)


def copy_full_folders(result):
    for set_name in ['set1', 'set2', 'set3']:
        dst_root = DATA_FULL / set_name
        if dst_root.exists():
            shutil.rmtree(dst_root)
        dst_root.mkdir(parents=True, exist_ok=True)
        recs = result[set_name]
        for idx, r in enumerate(sorted(recs, key=lambda x: (x['d'] or 0, x['K'] or 0,
                                                            x['bead'] or 0, x['avf'] or 0))):
            src = substrate_folder(r['abs_path'])
            if not src.exists():
                print(f"  !! source folder missing: {src}", file=sys.stderr)
                continue
            shutil.copytree(src, dst_root / clean_name(r, idx))
        print(f"  {set_name}: copied {len(list(dst_root.iterdir()))} folders -> {dst_root}")


def _write_rows(path, rows, cols):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def add_highdisp(do_copy):
    """Incrementally add NEW converged K=2/4/7 high-dispersion substrates to
    Set 1 (bead=0.3) & Set 3 (all beads), without wiping existing data_full."""
    inv_path = DATA_FULL / 'inventory.json'
    if not inv_path.exists():
        sys.exit(f"inventory.json not found at {inv_path}; run full classify first.")
    inv = json.loads(inv_path.read_text())
    existing = {r['path'] for r in inv['set1']} | {r['path'] for r in inv['set3']}

    candidates = []
    for folder in HIGHDISP_FOLDERS:
        candidates += collect(folder)

    new_set1, new_set3, skipped, seen = [], [], defaultdict(int), set()
    for r in candidates:
        if r['n_fibers'] is None or r['n_fibers'] < MIN_FIBERS:
            skipped['<500 fibers'] += 1; continue
        if r['d'] is None:
            skipped['off-grid d'] += 1; continue
        if r['K'] not in K_VF_TARGET:
            skipped['K not 2/4/7'] += 1; continue
        if r['bead'] not in BEADS:
            skipped['bead not in set'] += 1; continue
        if not in_vf_band(r):
            skipped['avf out of K-band'] += 1; continue
        if r['path'] in existing:
            skipped['already collected'] += 1; continue
        if r['path'] in seen:
            continue
        seen.add(r['path'])
        r['has_sim'] = has_diffusion_sim(r['abs_path'])
        new_set3.append(r)
        if abs(r['bead'] - 0.3) < 1e-9:
            new_set1.append(r)

    ex_hd = [r for r in inv['set3'] if r.get('K') in K_VF_TARGET]
    ex_out = [r for r in ex_hd if not in_vf_band(r)]

    cfg = defaultdict(int)
    for r in ex_hd:
        if in_vf_band(r):
            cfg[(r['d'], r['K'], r['bead'])] += 1
    for r in new_set3:
        cfg[(r['d'], r['K'], r['bead'])] += 1

    print("=" * 72)
    print("ADD HIGH-DISPERSION SUBSTRATES (incremental, K=2/4/7)")
    print("=" * 72)
    print(f"Candidates scanned: {len(candidates)}")
    print("Skipped:", dict(skipped))
    print(f"NEW -> Set 3: {len(new_set3)}  (Set 1 bead=0.3 subset: {len(new_set1)})")
    bykb = defaultdict(int)
    for r in new_set3:
        bykb[(r['K'], r['bead'])] += 1
    if bykb:
        print("\nNew adds by (K,bead):")
        for k in (7, 4, 2):
            for b in BEADS:
                c = bykb.get((k, b), 0)
                if c:
                    print(f"  K={k} bead={b}: {c}")

    print(f"\nCONFIG COVERAGE (converged in-band; target >={MIN_CONVERGED_PER_CONFIG}, '*'=short)")
    for k in (7, 4, 2):
        print(f"  K={k}  VF {K_VF_TARGET[k]:.2f}+-{VF_TOL}")
        print("    bead\\d " + "".join(f"{d:>7}" for d in DIAMS))
        for b in BEADS:
            row = f"    {b:>5} "
            for d in DIAMS:
                n = cfg.get((d, k, b), 0)
                row += f"{str(n) + ('*' if n < MIN_CONVERGED_PER_CONFIG else ''):>7}"
            print(row)
    gaps = sum(1 for k in (7, 4, 2) for d in DIAMS for b in BEADS
               if cfg.get((d, k, b), 0) < MIN_CONVERGED_PER_CONFIG)
    need_sim = [r for r in new_set3 if not r['has_sim']]
    print(f"\n  configs below target: {gaps} / {4 * 3 * 4}")
    print(f"  existing Set 3 K2/4/7 OUT-of-band (reported, kept): {len(ex_out)}")
    print(f"  new adds missing intra/extra diffusion sim: {len(need_sim)} / {len(new_set3)}")

    if not do_copy:
        print("\n(dry run) re-run with --add-highdisp --copy to copy + update inventory")
        return

    def do_add(recs, set_name):
        dst_root = DATA_FULL / set_name
        dst_root.mkdir(parents=True, exist_ok=True)
        added = 0
        for r in sorted(recs, key=lambda x: (x['d'], x['K'], x['bead'], x['avf'] or 0)):
            src = substrate_folder(r['abs_path'])
            if not src.exists():
                print(f"  !! source missing: {src}", file=sys.stderr); continue
            dst = dst_root / f"d{r['d']}_K{r['K']}_bead{r['bead']}_avf{(r['avf'] or 0):.2f}_{src.name}"
            if dst.exists():
                continue
            shutil.copytree(src, dst)
            added += 1
        return added

    a1 = do_add(new_set1, 'set1')
    a3 = do_add(new_set3, 'set3')
    print(f"\nCopied: Set 1 +{a1}, Set 3 +{a3}")

    clean = lambda r: {k: v for k, v in r.items() if k != 'has_sim'}
    inv['set1'] += [clean(r) for r in new_set1]
    inv['set3'] += [clean(r) for r in new_set3]
    inv_path.write_text(json.dumps(inv, indent=2))
    write_csv(DATA_FULL / 'set1_healthy.csv', inv['set1'])
    write_csv(DATA_FULL / 'set3_beading.csv', inv['set3'])

    _write_rows(DATA_FULL / 'highdisp_config_gaps.csv',
                [{'d': d, 'K': k, 'bead': b,
                  'converged_in_band': cfg.get((d, k, b), 0),
                  'shortfall': max(0, MIN_CONVERGED_PER_CONFIG - cfg.get((d, k, b), 0))}
                 for k in (7, 4, 2) for d in DIAMS for b in BEADS],
                ['d', 'K', 'bead', 'converged_in_band', 'shortfall'])
    write_csv(DATA_FULL / 'existing_out_of_band.csv', ex_out)
    _write_rows(DATA_FULL / 'needs_diffusion_sim.csv',
                [{'d': r['d'], 'K': r['K'], 'bead': r['bead'], 'avf': r['avf'],
                  'path': r['path']} for r in need_sim],
                ['d', 'K', 'bead', 'avf', 'path'])
    print(f"Inventory + CSVs updated; reports written to {DATA_FULL}")


def _inner_pkl_to_dir(set_root):
    m = {}
    for d in set_root.iterdir():
        dd = d / 'data'
        if d.is_dir() and dd.is_dir():
            for pk in dd.glob('*.pkl'):
                m[pk.name] = d
    return m


def resync_highdisp(do_copy):
    """Re-filter Set 1 & Set 3 K=2/4/7 members to EXACTLY the converged in-band
    set (per K_VF_TOL); keep ALL in-band reps. Coherent-K & Set 2 untouched."""
    inv_path = DATA_FULL / 'inventory.json'
    if not inv_path.exists():
        sys.exit(f"inventory.json not found at {inv_path}")
    inv = json.loads(inv_path.read_text())

    in_band = [r for r in highdisp_converged_records() if in_vf_band(r)]
    target3 = {r['path']: r for r in in_band}
    target1 = {p: r for p, r in target3.items() if abs(r['bead'] - 0.3) < 1e-9}
    targets = {'set1': target1, 'set3': target3}

    print("=" * 72)
    print("RESYNC HIGH-DISP (K=2/4/7) -> converged in-band set")
    print("=" * 72)
    for K in (2, 4, 7):
        t, tol = K_VF_TARGET[K], K_VF_TOL[K]
        print(f"  K{K} band [{t - tol:.2f},{t + tol:.2f}]")

    all_removed = []
    for s in ('set1', 'set3'):
        tgt = targets[s]
        cur = inv[s]
        cur_disp = [r for r in cur if r['K'] in K_VF_TARGET]
        cur_disp_paths = {r['path'] for r in cur_disp}
        remove = [r for r in cur_disp if r['path'] not in tgt]
        add = [tgt[p] for p in tgt if p not in cur_disp_paths]
        print(f"\n{s}: dispersed now={len(cur_disp)}  remove={len(remove)}  "
              f"add={len(add)}  -> {len(tgt)}")
        for r in remove:
            all_removed.append({'set': s, 'd': r.get('d'), 'K': r.get('K'),
                                'bead': r.get('bead'), 'avf': r.get('avf'),
                                'path': r.get('path')})
        if not do_copy:
            continue
        pmap = _inner_pkl_to_dir(DATA_FULL / s)
        for r in remove:
            d = pmap.get(Path(r['path']).name)
            if d is not None and d.parent == (DATA_FULL / s) and d.exists():
                shutil.rmtree(d)
        for r in sorted(add, key=lambda x: (x['d'], x['K'], x['bead'], x['avf'] or 0)):
            src = substrate_folder(r['abs_path'])
            if not src.exists():
                print(f"  !! source missing: {src}", file=sys.stderr); continue
            dst = DATA_FULL / s / f"d{r['d']}_K{r['K']}_bead{r['bead']}_avf{(r['avf'] or 0):.2f}_{src.name}"
            if not dst.exists():
                shutil.copytree(src, dst)
        inv[s] = ([r for r in cur if r['K'] not in K_VF_TARGET]
                  + [r for r in cur_disp if r['path'] in tgt]
                  + [tgt[p] for p in tgt if p not in cur_disp_paths])

    cfg = defaultdict(int)
    for r in in_band:
        cfg[(r['d'], r['K'], r['bead'])] += 1
    print("\nConfig coverage (in-band; target >=3):")
    for K in (2, 4, 7):
        ge3 = sum(1 for d in DIAMS for b in BEADS if cfg.get((d, K, b), 0) >= 3)
        short = [(d, b, cfg.get((d, K, b), 0)) for d in DIAMS for b in BEADS
                 if cfg.get((d, K, b), 0) < 3]
        print(f"  K{K}: {ge3}/16 configs >=3" + (f"; <3: {short}" if short else ""))
    need_sim = [r for r in in_band if not has_diffusion_sim(r['abs_path'])]
    print(f"  in-band dispersed total={len(in_band)}; missing intra/extra sim={len(need_sim)}")

    if not do_copy:
        print("\n(dry run) re-run with --resync-highdisp --copy to apply")
        return

    inv_path.write_text(json.dumps(inv, indent=2))
    write_csv(DATA_FULL / 'set1_healthy.csv', inv['set1'])
    write_csv(DATA_FULL / 'set3_beading.csv', inv['set3'])
    _write_rows(DATA_FULL / 'removed_out_of_band.csv', all_removed,
                ['set', 'd', 'K', 'bead', 'avf', 'path'])
    _write_rows(DATA_FULL / 'highdisp_config_gaps.csv',
                [{'d': d, 'K': K, 'bead': b,
                  'converged_in_band': cfg.get((d, K, b), 0),
                  'shortfall': max(0, MIN_CONVERGED_PER_CONFIG - cfg.get((d, K, b), 0))}
                 for K in (2, 4, 7) for d in DIAMS for b in BEADS],
                ['d', 'K', 'bead', 'converged_in_band', 'shortfall'])
    _write_rows(DATA_FULL / 'needs_diffusion_sim.csv',
                [{'d': r['d'], 'K': r['K'], 'bead': r['bead'], 'avf': r['avf'],
                  'path': r['path']} for r in need_sim],
                ['d', 'K', 'bead', 'avf', 'path'])
    post_oob = [r for s in ('set1', 'set3') for r in inv[s]
                if r['K'] in K_VF_TARGET and not in_vf_band(r)]
    write_csv(DATA_FULL / 'existing_out_of_band.csv', post_oob)
    print(f"\nInventory + CSVs updated; reports written to {DATA_FULL}")


def band_scan():
    """Report achieved-VF distributions + band-width trade-off for K=2/4/7 so a
    band can be chosen (K2 & K7 >=3 reps for every (d,bead) config; K4 may lag)."""
    recs = highdisp_converged_records()
    Ks = (2, 4, 7)
    n_configs = len(DIAMS) * len(BEADS)  # 16 per K
    print("=" * 72)
    print(f"BAND SCAN (converged high-disp; {len(recs)} substrates; {n_configs} configs/K)")
    print("=" * 72)
    import statistics as st
    print("\nAchieved avf per K:")
    for K in Ks:
        v = sorted(r['avf'] for r in recs if r['K'] == K)
        print(f"  K{K} (target {K_VF_TARGET[K]}): n={len(v)} min={v[0]:.2f} max={v[-1]:.2f} "
              f"mean={st.mean(v):.3f} sd={st.pstdev(v):.3f} median={st.median(v):.2f}")

    def configs_ge3(K, tol):
        t = K_VF_TARGET[K]
        cfg = defaultdict(int)
        for r in recs:
            if r['K'] == K and (t - tol) <= r['avf'] <= (t + tol):
                cfg[(r['d'], r['bead'])] += 1
        return sum(1 for c in cfg.values() if c >= 3), cfg

    # max feasible configs>=3 at the widest tol (defines true converged gaps)
    wide = 0.20
    feasible = {K: configs_ge3(K, wide)[0] for K in Ks}

    tols = [round(0.04 + 0.005 * i, 3) for i in range(int((0.12 - 0.04) / 0.005) + 1)]
    print("\nConfigs with >=3 in-band (of 16) by uniform tolerance:")
    print(f"  {'tol':>5} | {'K2':>7} | {'K4':>7} | {'K7':>7} | in-band N (K2/K4/K7)")
    for tol in tols:
        cells = {}
        ns = {}
        for K in Ks:
            ge3, cfg = configs_ge3(K, tol)
            cells[K] = ge3
            ns[K] = sum(cfg.values())
        print(f"  {tol:>5.3f} | {cells[2]:>3}/16 | {cells[4]:>3}/16 | {cells[7]:>3}/16 | "
              f"{ns[2]}/{ns[4]}/{ns[7]}")

    print(f"\n  max feasible configs>=3 at tol={wide} (converged-rep ceiling): "
          f"K2={feasible[2]}/16 K4={feasible[4]}/16 K7={feasible[7]}/16")

    # tightest uniform tol meeting K2 & K7 ceilings
    def tightest_uniform():
        for tol in tols:
            if configs_ge3(2, tol)[0] >= feasible[2] and configs_ge3(7, tol)[0] >= feasible[7]:
                return tol
        return None
    tu = tightest_uniform()
    print(f"\n  tightest UNIFORM tol s.t. K2 & K7 hit their ceiling: {tu}")
    if tu is not None:
        for K in Ks:
            ge3, cfg = configs_ge3(K, tu)
            short = sorted(k for k, c in cfg.items() if c < 3)
            band = f"[{K_VF_TARGET[K]-tu:.2f},{K_VF_TARGET[K]+tu:.2f}]"
            print(f"    K{K} band {band}: configs>=3 = {ge3}/16; <3 configs (d,bead): {short}")

    print("\n  tightest PER-K tol at each K's own ceiling:")
    for K in Ks:
        pick = next((tol for tol in tols if configs_ge3(K, tol)[0] >= feasible[K]), None)
        band = f"[{K_VF_TARGET[K]-pick:.2f},{K_VF_TARGET[K]+pick:.2f}]" if pick else 'n/a'
        print(f"    K{K}: tol={pick} band {band} (ceiling {feasible[K]}/16)")

    # true converged-rep gaps (cannot reach 3 even at widest tol)
    print("\n  true converged-rep gaps (<3 even at tol=%.2f, need generation):" % wide)
    for K in Ks:
        _, cfg = configs_ge3(K, wide)
        gaps = sorted((d, b, cfg.get((d, b), 0)) for d in DIAMS for b in BEADS
                      if cfg.get((d, b), 0) < 3)
        if gaps:
            print(f"    K{K}: " + ", ".join(f"d{d}/bead{b}={n}" for d, b, n in gaps))
        else:
            print(f"    K{K}: none")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--copy', action='store_true',
                    help='wipe & rebuild data_full/set1|set2|set3 with full folders')
    ap.add_argument('--add-highdisp', action='store_true',
                    help='incrementally add new K=2/4/7 high-dispersion substrates '
                         'to Set 1 & 3 (no rebuild); combine with --copy to execute')
    ap.add_argument('--band-scan', action='store_true',
                    help='report achieved-VF band-width trade-off for K=2/4/7 (read-only)')
    ap.add_argument('--resync-highdisp', action='store_true',
                    help='re-filter Set 1 & 3 K=2/4/7 to the in-band set (per K_VF_TOL); '
                         'combine with --copy to apply')
    args = ap.parse_args()

    if args.band_scan:
        band_scan()
        return

    if args.resync_highdisp:
        resync_highdisp(do_copy=args.copy)
        return

    if args.add_highdisp:
        add_highdisp(do_copy=args.copy)
        return

    result = classify()
    report(result)

    # always (re)write inventory + CSVs
    DATA_FULL.mkdir(parents=True, exist_ok=True)
    inv = {k: result[k] for k in ['set1', 'set2', 'set3']}
    with open(DATA_FULL / 'inventory.json', 'w') as f:
        json.dump(inv, f, indent=2)
    write_csv(DATA_FULL / 'set1_healthy.csv', result['set1'])
    write_csv(DATA_FULL / 'set2_axonal_loss.csv', result['set2'])
    write_csv(DATA_FULL / 'set3_beading.csv', result['set3'])
    write_csv(DATA_FULL / 'dropped_under500.csv', result['dropped_small'])
    write_csv(DATA_FULL / 'excluded_offgrid_diameter.csv', result['excluded_offgrid'])
    print(f"\nInventory + CSVs written to {DATA_FULL}")

    if args.copy:
        print("\nCopying full substrate folders (originals preserved)...")
        copy_full_folders(result)
    else:
        print("\n(dry run) re-run with --copy to rebuild data_full/ full folders")


if __name__ == '__main__':
    main()
