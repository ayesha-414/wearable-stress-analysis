#!/usr/bin/env python3
"""
scripts/diagnose_alignment.py
=============================
READ-ONLY diagnostic for time alignment of the PhysioNet Wearable Device
dataset (STRESS protocol, Empatica E4).

What it does
------------
* Reads raw CSV headers / tags / IBI for every STRESS recording folder
  (including split recordings such as f14_a / f14_b).
* Compares the project's stage->tag mapping (src/preprocessing.py) with the
  dataset authors' notebook (parsed from Wearable_Dataset.ipynb).
* Replicates the project's per-signal window selection and reports whether
  each sensor's window covers the same absolute time interval as the
  EDA-based tag boundaries.
* Investigates the unbounded Subtract window, Baseline boundaries and the
  f14_a / f14_b split recording.

What it does NOT do
-------------------
* It never modifies, deletes or overwrites any existing file.
* It never merges, corrects or resamples data for downstream use.
* It never trains a model.
* Its only write is ONE NEW report file (default:
  reports/diagnostics/alignment_diagnostic_<timestamp>.txt).

Uses only the Python standard library.

Usage (from the project root):
    python scripts\\diagnose_alignment.py
    python scripts\\diagnose_alignment.py --dataset "D:\\data\\Wearable_Dataset"
    python scripts\\diagnose_alignment.py --subjects S01 f01 f14
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ----------------------------------------------------------------------------
# Tolerances / expectations (edit here if you want different thresholds)
# ----------------------------------------------------------------------------
TOL_START_S = 0.5      # start-time difference vs EDA considered "an offset"
TOL_END_S = 2.0        # end-time difference vs EDA considered a mismatch
EXPECTED_FS = {"EDA": 4.0, "TEMP": 4.0, "HR": 1.0, "BVP": 64.0, "ACC": 32.0}
SIGNAL_NCOLS = {"EDA": 1, "TEMP": 1, "HR": 1, "BVP": 1, "ACC": 3}
SIGNAL_ORDER = ["EDA", "HR", "BVP", "TEMP", "ACC", "IBI"]
EXPECTED_TAGS_FALLBACK = {"V1": 12, "V2": 9}   # notebook markdown cell 1

# Project stage maps, COPIED from src/preprocessing.py
# (interpreted as indices into tags.csv, exactly as the project code does)
PROJECT_MAP_RAW = {
    "V1": {
        "Baseline": (0, 2), "Stroop": (3, 4), "First Rest": (2, 3),
        "TMCT": (5, 6), "Second Rest": (4, 5), "Real Opinion": (7, 8),
        "Opposite Opinion": (9, 10), "Subtract": (11, 12),
    },
    "V2": {
        "Baseline": (0, 1), "TMCT": (2, 3), "First Rest": (1, 2),
        "Real Opinion": (4, 5), "Opposite Opinion": (6, 7),
        "Second Rest": (3, 4), "Subtract": (7, 8),
    },
}

# Fallback copy of the notebook spans, in the NOTEBOOK's own indexing
# (notebook list = [session_start] + tags.csv rows). Used only if the notebook
# cannot be parsed. Source: Wearable_Dataset.ipynb, cell "graph_multiple".
NOTEBOOK_STAGE_ORDER = {
    "V1": ["Stroop", "TMCT", "Real Opinion", "Opposite Opinion", "Subtract"],
    "V2": ["TMCT", "Real Opinion", "Opposite Opinion", "Subtract"],
}
NOTEBOOK_T_FALLBACK = {
    "V1": [(3, 4), (5, 6), (7, 8), (9, 10), (11, 12)],
    "V2": [(2, 3), (4, 5), (6, 7), (8, 9)],
}


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
class Report:
    """Collects every line; prints to console; saved to a NEW file at the end."""

    def __init__(self):
        self.lines: list[str] = []
        self.flags: list[tuple[str, str, str]] = []   # (category, where, message)

    def out(self, text: str = "", console: bool = True):
        self.lines.append(text)
        if console:
            try:
                print(text)
            except UnicodeEncodeError:
                print(text.encode("ascii", "replace").decode("ascii"))

    def title(self, text: str):
        self.out("")
        self.out("=" * 78)
        self.out(text)
        self.out("=" * 78)

    def flag(self, category: str, where: str, message: str):
        self.flags.append((category, where, message))

    def table(self, headers, rows, console: bool = True):
        rows = [["" if c is None else str(c) for c in r] for r in rows]
        widths = [len(h) for h in headers]
        for r in rows:
            for i, c in enumerate(r):
                widths[i] = max(widths[i], len(c))
        line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
        self.out(line, console)
        self.out("  ".join("-" * w for w in widths), console)
        for r in rows:
            self.out("  ".join(c.ljust(widths[i]) for i, c in enumerate(r)), console)


def tonum(tok):
    try:
        v = float(tok)
    except (ValueError, TypeError):
        return None
    return v if math.isfinite(v) else None


def parse_ts(tok):
    """Epoch seconds from a float or 'YYYY-MM-DD HH:MM:SS[.f]' string
    (naive strings are treated as UTC). Returns None if unparsable."""
    if tok is None:
        return None
    tok = str(tok).strip().strip('"')
    if not tok:
        return None
    v = tonum(tok)
    if v is not None:
        return v
    s = tok.replace("T", " ").rstrip("Z")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            pass
    return None


def iso(ts):
    if ts is None:
        return "NA"
    try:
        d = datetime.fromtimestamp(ts, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return "BAD"
    s = d.strftime("%Y-%m-%d %H:%M:%S")
    frac = ts - math.floor(ts)
    if frac > 1e-6:
        s += ".%03d" % int(round(frac * 1000))
    return s


def fnum(x, nd=1, sign=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def cv_pct(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = statistics.mean(xs)
    return None if m == 0 else statistics.pstdev(xs) / m * 100.0


# ----------------------------------------------------------------------------
# Raw file readers (all read-only, all tolerant of bad data)
# ----------------------------------------------------------------------------
def read_regular(path: Path, ncols: int, keep_values: bool = False) -> dict:
    info = dict(status="missing", start=None, fs=None, n=0, bad=0, dur=None,
                end=None, hdr=None, issues=[], values=None)
    try:
        if not path.is_file():
            return info
        if path.stat().st_size == 0:
            info["status"] = "empty"
            return info
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            l1, l2 = fh.readline(), fh.readline()
            info["hdr"] = (l1.strip(), l2.strip())
            info["start"] = parse_ts(l1.split(",")[0])
            fs = tonum(l2.split(",")[0]) if l2.strip() else None
            info["fs"] = fs if (fs is not None and fs > 0) else None
            vals = [] if keep_values else None
            n = bad = 0
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                n += 1
                toks = s.split(",")
                v0 = tonum(toks[0]) if toks else None
                ok = len(toks) >= ncols and v0 is not None and all(
                    tonum(t) is not None for t in toks[1:ncols])
                if not ok:
                    bad += 1
                if vals is not None:
                    vals.append(v0 if (ok and v0 is not None) else float("nan"))
            info["n"], info["bad"], info["values"] = n, bad, vals
        if info["start"] is None:
            info["issues"].append("start timestamp not parsable")
        if info["fs"] is None:
            info["issues"].append("sampling rate missing/invalid")
        if n == 0:
            info["issues"].append("no data rows")
        if bad:
            info["issues"].append(f"{bad} malformed data rows")
        if info["start"] is None or info["fs"] is None:
            info["status"] = "bad_header"
        elif n == 0:
            info["status"] = "no_data"
        else:
            info["status"] = "ok"
            info["dur"] = n / info["fs"]
            info["end"] = info["start"] + info["dur"]
    except Exception as exc:  # never let one file kill the run
        info["status"] = "error"
        info["issues"].append(f"{type(exc).__name__}: {exc}")
    return info


def read_ibi(path: Path) -> dict:
    info = dict(status="missing", start=None, n=0, bad=0, first_off=None,
                last_off=None, nonmono=0, ibi_min=None, ibi_max=None,
                n_implausible=0, hdr=None, issues=[], end=None, dur=None)
    try:
        if not path.is_file():
            return info
        if path.stat().st_size == 0:
            info["status"] = "empty"
            return info
        offs, ibis = [], []
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            l1 = fh.readline()
            info["hdr"] = (l1.strip(), "")
            info["start"] = parse_ts(l1.split(",")[0])
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                toks = s.split(",")
                o = tonum(toks[0]) if len(toks) >= 1 else None
                b = tonum(toks[1]) if len(toks) >= 2 else None
                if o is None or b is None:
                    info["bad"] += 1
                    continue
                offs.append(o)
                ibis.append(b)
        info["n"] = len(offs)
        if offs:
            info["first_off"], info["last_off"] = offs[0], offs[-1]
            info["nonmono"] = sum(1 for i in range(1, len(offs)) if offs[i] < offs[i - 1])
            info["ibi_min"], info["ibi_max"] = min(ibis), max(ibis)
            info["n_implausible"] = sum(1 for b in ibis if b < 0.3 or b > 2.0)
        if info["start"] is None:
            info["issues"].append("start timestamp not parsable")
        if info["bad"]:
            info["issues"].append(f"{info['bad']} malformed rows")
        if info["nonmono"]:
            info["issues"].append(f"{info['nonmono']} non-increasing offsets")
        if info["n"] == 0:
            info["status"] = "empty"
        elif info["start"] is None:
            info["status"] = "bad_header"
        else:
            info["status"] = "ok"
            info["end"] = info["start"] + info["last_off"]
            info["dur"] = info["last_off"]
    except Exception as exc:
        info["status"] = "error"
        info["issues"].append(f"{type(exc).__name__}: {exc}")
    return info


def read_tags(path: Path) -> dict:
    info = dict(status="missing", abs=[], bad=0, n=0, issues=[])
    try:
        if not path.is_file():
            return info
        if path.stat().st_size == 0:
            info["status"] = "empty"
            return info
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                t = parse_ts(s.split(",")[0])
                if t is None:
                    info["bad"] += 1
                else:
                    info["abs"].append(t)
        info["n"] = len(info["abs"])
        if info["bad"]:
            info["issues"].append(f"{info['bad']} unparsable tag lines")
        info["status"] = "ok" if info["n"] else "empty"
    except Exception as exc:
        info["status"] = "error"
        info["issues"].append(f"{type(exc).__name__}: {exc}")
    return info


def read_recording(folder: Path) -> dict:
    rec = {"dir": folder, "sig": {}, "tags": None}
    for name, nc in SIGNAL_NCOLS.items():
        rec["sig"][name] = read_regular(folder / f"{name}.csv", nc,
                                        keep_values=name in ("EDA", "HR"))
    rec["sig"]["IBI"] = read_ibi(folder / "IBI.csv")
    rec["tags"] = read_tags(folder / "tags.csv")
    return rec


# ----------------------------------------------------------------------------
# Notebook parsing (so the mapping evidence comes from YOUR copy of the file)
# ----------------------------------------------------------------------------
def parse_notebook(path: Path) -> dict:
    res = dict(found=False, spans=None, insert_start=None, insert_arg=None,
               start_from_eda=None, tag_counts={}, notes=[])
    if not path.is_file():
        res["notes"].append(f"Notebook not found at {path}")
        return res
    try:
        with open(path, "r", encoding="utf-8") as fh:
            nb = json.load(fh)
    except Exception as exc:
        res["notes"].append(f"Notebook could not be parsed: {exc}")
        return res
    res["found"] = True
    pair_re = re.compile(
        r"""\[["']tags["']\]\[(\d+)\]\s*,\s*[^\n]*?\[["']tags["']\]\[(\d+)\]""")
    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        src = "".join(src) if isinstance(src, list) else str(src)
        if cell.get("cell_type") == "markdown":
            for ver in ("1", "2"):
                m = re.search(r"Stress\W{0,3}\s*version\s*%s:\s*(\d+)\s*tags" % ver, src, re.I)
                if m:
                    res["tag_counts"]["V" + ver] = int(m.group(1))
        elif cell.get("cell_type") == "code":
            if "axvspan" in src and res["spans"] is None:
                low = src.lower()
                k = low.find("second version")
                if k > 0:
                    v1 = [(int(a), int(b)) for a, b in pair_re.findall(src[:k])]
                    v2 = [(int(a), int(b)) for a, b in pair_re.findall(src[k:])]
                    if len(v1) == 5 and len(v2) == 4:
                        res["spans"] = {"V1": v1, "V2": v2}
                    else:
                        res["notes"].append(
                            f"axvspan pairs found: V1={len(v1)} V2={len(v2)} (expected 5/4)")
            m = re.search(r"np\.insert\(\s*tags_vector\s*,\s*0\s*,\s*([^)]*)\)", src)
            if m:
                res["insert_start"] = True
                res["insert_arg"] = m.group(1).strip()
            if "utc_start_dict" in src and "EDA.csv" in src and "columns" in src:
                res["start_from_eda"] = True
    if res["insert_start"] is None:
        res["insert_start"] = False
    return res


# ----------------------------------------------------------------------------
# Window logic
# ----------------------------------------------------------------------------
def project_windows(stage_map, tags_rel, end_rel):
    """Replicates src/preprocessing.segment_by_stage boundary logic."""
    n, out = len(tags_rel), {}
    for stage, (s, e) in stage_map.items():
        if s >= n or e > n:
            out[stage] = dict(status="SKIPPED", t0=None, t1=None, end=None, span=(s, e))
        elif e < n:
            t0, t1 = int(tags_rel[s]), int(tags_rel[e])
            out[stage] = dict(status="bounded", t0=t0, t1=t1, end=t1, span=(s, e))
        else:
            t0 = int(tags_rel[s])
            out[stage] = dict(status="UNBOUNDED", t0=t0, t1=None, end=end_rel, span=(s, e))
    return out


def notebook_windows(t_spans, stage_names, tags_rel):
    """Notebook indexing: T = [0.0 (EDA start)] + tags_rel."""
    T = [0.0] + list(tags_rel)
    out = {}
    for stage, (a, b) in zip(stage_names, t_spans):
        out[stage] = (T[a], T[b]) if (a < len(T) and b < len(T)) else None
    return out


def iou(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else None


def window_mean(info, a_abs, b_abs):
    vals, st, fs = info.get("values"), info.get("start"), info.get("fs")
    if not vals or st is None or not fs:
        return None
    i0 = max(0, int(math.ceil((a_abs - st) * fs)))
    i1 = min(len(vals), int(math.floor((b_abs - st) * fs)) + 1)
    seg = [v for v in vals[i0:i1] if not math.isnan(v)]
    return sum(seg) / len(seg) if len(seg) >= 10 else None


# ----------------------------------------------------------------------------
# Subject loading / derived values
# ----------------------------------------------------------------------------
def cohort_of(sid: str) -> str:
    if sid[:1] == "f":
        return "V2"
    if sid[:1] == "S":
        return "V1"
    return "?"


def load_subjects(stress_dir: Path, only):
    names = sorted(p.name for p in stress_dir.iterdir() if p.is_dir())
    groups = defaultdict(list)
    for n in names:
        m = re.fullmatch(r"([A-Za-z]\d+)_([A-Za-z0-9]+)", n)
        groups[m.group(1) if m else n].append(n)
    subs = {}
    for canon in sorted(groups):
        if only and canon not in only:
            continue
        folders = sorted(groups[canon])
        subs[canon] = dict(
            id=canon, cohort=cohort_of(canon), folders=folders,
            recs={f: read_recording(stress_dir / f) for f in folders},
            split=(canon not in folders) or len(folders) > 1)
    return subs


def eda_of(rec):
    e = rec["sig"]["EDA"]
    return e if e["status"] == "ok" else None


def tags_rel_of(rec):
    e = eda_of(rec)
    if e is None or rec["tags"]["n"] == 0:
        return None
    return [t - e["start"] for t in rec["tags"]["abs"]]


def analysable(sub):
    if sub["split"]:
        return False
    return tags_rel_of(sub["recs"][sub["folders"][0]]) is not None


# ----------------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------------
def section_header(rep, args, stress_dir):
    rep.title("WEARABLE ALIGNMENT DIAGNOSTIC (read-only)")
    rep.out(f"Generated (local time): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rep.out(f"Python: {sys.version.split()[0]}")
    rep.out(f"Project root : {PROJECT_ROOT}")
    rep.out(f"STRESS folder: {stress_dir}")
    rep.out("")
    rep.out("Assumptions:")
    rep.out("  A1. A file's header timestamp = time of its first data row.")
    rep.out("  A2. Naive datetime strings are treated as UTC (only differences matter).")
    rep.out("  A3. Project stage spans are indices into tags.csv (as src/preprocessing.py uses them).")
    rep.out("  A4. Stress_Level_v*.csv column order is assumed chronological (hint only).")
    rep.out(f"  Tolerances: start offset > {TOL_START_S}s, end mismatch > {TOL_END_S}s are flagged.")
    rep.out("Nothing is corrected, merged or modelled. Findings below are measurements.")


def section_notebook(rep, nb):
    rep.title("1. DATASET AUTHORS' NOTEBOOK vs PROJECT STAGE MAPPING")
    for n in nb["notes"]:
        rep.out(f"  note: {n}")
        rep.flag("NOTEBOOK", "notebook", n)
    if not nb["found"]:
        rep.out("  Notebook unavailable -> using hard-coded fallback copy of its spans.")
        return
    rep.out(f"  Notebook prepends a start time as tags[0] (np.insert(tags_vector,0,...)): "
            f"{nb['insert_start']}  (argument: {nb['insert_arg']})")
    rep.out(f"  That start time is read from EDA.csv header: {nb['start_from_eda']}")
    rep.out(f"  Tag counts stated in notebook markdown: {nb['tag_counts'] or 'not found'}")
    rep.out(f"  Task spans parsed from notebook (axvspan): {'yes' if nb['spans'] else 'NO (fallback used)'}")
    rep.out("  The notebook defines ONLY task spans; it defines no Baseline / rest spans.")
    rep.out("")
    shift = 1 if nb["insert_start"] else 0
    spans = nb["spans"] or NOTEBOOK_T_FALLBACK
    rows = []
    for coh in ("V1", "V2"):
        for stage, (a, b) in zip(NOTEBOOK_STAGE_ORDER[coh], spans[coh]):
            raw = (a - shift, b - shift)
            proj = PROJECT_MAP_RAW[coh][stage]
            same = raw == proj
            rows.append([coh, stage, f"T[{a}]->T[{b}]", f"{raw[0]}->{raw[1]}",
                         f"{proj[0]}->{proj[1]}", "same" if same else "DIFFERENT"])
            if not same:
                rep.flag("MAPPING_DIFF", coh, f"{stage}: notebook raw {raw} vs project {proj}")
    rep.table(["Cohort", "Stage", "Notebook idx", "= tags.csv idx", "Project idx", "Match?"], rows)
    rep.out("")
    rep.out("  Reading: both use the EDA header time as time zero, so this is an INDEX")
    rep.out("  difference, not a clock-origin difference. The notebook being the authors'")
    rep.out("  code is provenance, not proof; sections 4-7 gather measured evidence.")


def section_inventory(rep, subs):
    rep.title("2. FILE INVENTORY (per recording folder)")
    rows = []
    for sid, sub in subs.items():
        for f in sub["folders"]:
            rec = sub["recs"][f]
            cells = []
            for name in SIGNAL_ORDER:
                st = rec["sig"][name]["status"]
                cells.append("ok" if st == "ok" else st.upper())
                if st != "ok":
                    cat = "FILE_MISSING" if st == "missing" else "FILE_EMPTY" if st in ("empty", "no_data") else "FILE_MALFORMED"
                    rep.flag(cat, f, f"{name}.csv status={st} {rec['sig'][name]['issues']}")
                elif rec["sig"][name]["issues"]:
                    rep.flag("FILE_MALFORMED", f, f"{name}.csv {rec['sig'][name]['issues']}")
            ts = rec["tags"]["status"]
            cells.append("ok" if ts == "ok" else ts.upper())
            if ts != "ok":
                rep.flag("FILE_EMPTY" if ts == "empty" else "FILE_MISSING", f, f"tags.csv status={ts}")
            for iss in rec["tags"]["issues"]:
                rep.flag("FILE_MALFORMED", f, f"tags.csv {iss}")
            rows.append([f, sub["cohort"]] + cells)
    rep.table(["Recording", "Coh"] + SIGNAL_ORDER + ["tags"], rows)


def section_timing(rep, subs):
    rep.title("3. RAW HEADER TIMING")
    # Table 3a: start offsets
    rep.out("3a. Start time of each signal relative to EDA start (seconds; MISS = unusable)")
    rows = []
    for sid, sub in subs.items():
        for f in sub["folders"]:
            rec = sub["recs"][f]
            e = eda_of(rec)
            cells = []
            for name in SIGNAL_ORDER[1:]:
                s = rec["sig"][name]
                if e is None or s["status"] != "ok":
                    cells.append("MISS")
                    continue
                d = s["start"] - e["start"]
                cells.append(fnum(d, 1, True))
                if abs(d) > TOL_START_S:
                    rep.flag("START_OFFSET", f, f"{name} header starts {d:+.1f}s vs EDA")
            rows.append([f, iso(e["start"]) if e else "NA"] + cells)
    rep.table(["Recording", "EDA start (UTC)"] + [f"d{n}" for n in SIGNAL_ORDER[1:]], rows)

    rep.out("")
    rep.out("3b. Sampling rates found (flagged 'x' when different from the documented rate)")
    rows = []
    for sid, sub in subs.items():
        for f in sub["folders"]:
            rec = sub["recs"][f]
            cells = []
            for name in ("EDA", "HR", "BVP", "TEMP", "ACC"):
                s = rec["sig"][name]
                if s["fs"] is None:
                    cells.append("NA")
                    continue
                bad = abs(s["fs"] - EXPECTED_FS[name]) > 1e-6
                cells.append(f"{s['fs']:g}" + ("x" if bad else ""))
                if bad:
                    rep.flag("FS_MISMATCH", f, f"{name} fs={s['fs']:g} expected {EXPECTED_FS[name]:g}")
            rows.append([f] + cells)
    rep.table(["Recording", "EDA", "HR", "BVP", "TEMP", "ACC"], rows)

    rep.out("")
    rep.out("3c. Durations (s) = rows / fs; IBI = last offset. d_end = signal end - EDA end.")
    rows = []
    for sid, sub in subs.items():
        for f in sub["folders"]:
            rec = sub["recs"][f]
            e = eda_of(rec)
            cells = []
            for name in SIGNAL_ORDER:
                s = rec["sig"][name]
                if s["status"] != "ok":
                    cells.append("MISS")
                    continue
                txt = fnum(s["dur"], 0)
                if e is not None and name != "EDA":
                    de = s["end"] - e["end"]
                    txt += f" ({de:+.0f})"
                    if abs(de) > TOL_END_S:
                        rep.flag("DURATION_MISMATCH", f, f"{name} ends {de:+.1f}s vs EDA end")
                cells.append(txt)
            rows.append([f] + cells)
    rep.table(["Recording"] + SIGNAL_ORDER, rows)


def section_tags(rep, subs, nb):
    rep.title("4. TAGS (button presses), relative to EDA start")
    exp = dict(EXPECTED_TAGS_FALLBACK)
    exp.update(nb.get("tag_counts") or {})
    rows = []
    for sid, sub in subs.items():
        for f in sub["folders"]:
            rec = sub["recs"][f]
            e = eda_of(rec)
            n = rec["tags"]["n"]
            want = "split" if sub["split"] else exp.get(sub["cohort"], "?")
            if e is None or n == 0:
                rows.append([f, sub["cohort"], n, want, "n/a", "NA", "NA", "NA", "NA"])
                continue
            rel = [t - e["start"] for t in rec["tags"]["abs"]]
            note = "ok"
            if not sub["split"] and want != n:
                note = "COUNT!"
                rep.flag("TAG_COUNT", f, f"{n} tags, expected {want}")
            if any(rel[i] < rel[i - 1] for i in range(1, len(rel))):
                note += " ORDER!"
                rep.flag("TAG_ORDER", f, "tags not chronological")
            if rel[0] < 0 or rel[-1] > e["dur"] + TOL_END_S:
                note += " OUTSIDE!"
                rep.flag("TAG_OUTSIDE_EDA", f, f"tags span {rel[0]:.0f}..{rel[-1]:.0f}s, EDA is 0..{e['dur']:.0f}s")
            rows.append([f, sub["cohort"], n, want, note, fnum(rel[0], 0), fnum(rel[-1], 0),
                         fnum(e["dur"] - rel[-1], 0), fnum(e["dur"], 0)])
    rep.table(["Recording", "Coh", "n_tags", "expected", "check", "1st tag", "last tag",
               "post-last", "EDA dur"], rows)
    rep.out("  (1st tag = seconds from EDA start to first press; post-last = EDA end - last tag)")


def interval_key_label(key, N, mapping, nb_raw):
    """Stage names covering an interval key under a mapping."""
    proj, nbk = [], []
    if key == "pre":
        return "-", "-"
    for stage, (s, e) in mapping.items():
        if key == "last":
            if s <= N - 1 and e >= N:
                proj.append(stage)
        elif s <= key and key + 1 <= e:
            proj.append(stage)
    for stage, (s, e) in nb_raw.items():
        if key != "last" and s <= key and key + 1 <= e:
            nbk.append(stage)
    return ("/".join(proj) or "-"), ("/".join(nbk) or "-")


def section_mapping(rep, subs, nb):
    rep.title("5. MAPPING EVIDENCE: project vs notebook (measured, no verdict)")
    exp = dict(EXPECTED_TAGS_FALLBACK)
    exp.update(nb.get("tag_counts") or {})
    shift = 1 if nb["insert_start"] else 0
    spans = nb["spans"] or NOTEBOOK_T_FALLBACK

    # 5a project vs notebook windows, per cohort/stage
    rep.out("5a. Project window vs notebook window, stages the notebook defines")
    for coh in ("V1", "V2"):
        names, tsp = NOTEBOOK_STAGE_ORDER[coh], spans[coh]
        agg = defaultdict(lambda: dict(sd=[], ed=[], io=[], unb=0, skip=0, n=0))
        for sid, sub in subs.items():
            if sub["cohort"] != coh or not analysable(sub):
                continue
            rec = sub["recs"][sub["folders"][0]]
            rel, e = tags_rel_of(rec), eda_of(rec)
            pw = project_windows(PROJECT_MAP_RAW[coh], rel, e["dur"])
            nw = notebook_windows(tsp, names, rel)
            for stage in names:
                a, p, nbw = agg[stage], pw[stage], nw[stage]
                a["n"] += 1
                if p["status"] == "SKIPPED":
                    a["skip"] += 1
                    continue
                if p["status"] == "UNBOUNDED":
                    a["unb"] += 1
                if nbw is None:
                    continue
                a["sd"].append(p["t0"] - nbw[0])
                if p["t1"] is not None:
                    a["ed"].append(p["t1"] - nbw[1])
                a["io"].append(iou(p["t0"], p["end"], nbw[0], nbw[1]))
        rows = []
        for stage, (a, b) in zip(names, tsp):
            g = agg[stage]
            rows.append([stage, "%d->%d" % PROJECT_MAP_RAW[coh][stage],
                         "%d->%d" % (a - shift, b - shift), g["n"],
                         fnum(med(g["sd"]), 0, True), fnum(med(g["ed"]), 0, True),
                         g["unb"], g["skip"], fnum(med(g["io"]), 2)])
        rep.out(f"  Cohort {coh}")
        rep.table(["Stage", "project idx", "notebook idx", "n", "med start diff(s)",
                   "med end diff(s)", "#unbounded", "#skipped", "med IoU"], rows)
        rep.out("")

    # 5b chronology vs label-file column order
    rep.out("5b. Chronological order implied by the project map vs Stress_Level column order")
    for coh, fname in (("V1", "Stress_Level_v1.csv"), ("V2", "Stress_Level_v2.csv")):
        path = PROJECT_ROOT / fname
        cols = None
        try:
            with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as fh:
                hdr = next(csv.reader(fh))
                cols = [c.strip() for c in hdr[1:] if c.strip()]
        except Exception as exc:
            rep.out(f"  {coh}: cannot read {fname}: {exc}")
            continue
        by_start = [s for s, _ in sorted(PROJECT_MAP_RAW[coh].items(), key=lambda kv: kv[1])]
        same = by_start == cols
        rep.out(f"  {coh} project order (by tag index): {by_start}")
        rep.out(f"  {coh} {fname} columns             : {cols}")
        rep.out(f"  {coh} identical order: {same}   (assumption A4: column order is chronological)")
        if not same:
            rep.flag("MAPPING_DIFF", coh, "project stage order differs from Stress_Level column order")

    # 5c structural facts
    rep.out("")
    rep.out("5c. Structural facts")
    for coh in ("V1", "V2"):
        mx = max(e for _, e in PROJECT_MAP_RAW[coh].values())
        counts = Counter(s["recs"][s["folders"][0]]["tags"]["n"]
                         for s in subs.values() if s["cohort"] == coh and not s["split"])
        rep.out(f"  {coh}: highest tag index the project needs = {mx} (needs {mx + 1} tags to be bounded);"
                f" expected tags = {exp.get(coh)}; observed counts = {dict(counts)}")

    # 5d per-interval evidence
    rep.out("")
    rep.out("5d. Inter-tag intervals: duration spread and median EDA / HR (absolute-time slices)")
    rep.out("    'pre' = EDA start -> tag 0; 'last' = last tag -> EDA end.")
    rep.out("    Low spread (CV) is consistent with a fixed-length protocol step; it does not prove it.")
    for coh in ("V1", "V2"):
        N = exp.get(coh, 0)
        nb_raw = {st: (a - shift, b - shift) for st, (a, b) in zip(NOTEBOOK_STAGE_ORDER[coh], spans[coh])}
        data = defaultdict(lambda: dict(d=[], eda=[], hr=[]))
        for sid, sub in subs.items():
            if sub["cohort"] != coh or not analysable(sub):
                continue
            rec = sub["recs"][sub["folders"][0]]
            e, rel = eda_of(rec), tags_rel_of(rec)
            bounds = [("pre", 0.0, rel[0])]
            bounds += [(i, rel[i], rel[i + 1]) for i in range(len(rel) - 1)]
            bounds += [("last", rel[-1], e["dur"])]
            for key, a, b in bounds:
                g = data[key]
                g["d"].append(b - a)
                g["eda"].append(window_mean(rec["sig"]["EDA"], e["start"] + a, e["start"] + b))
                g["hr"].append(window_mean(rec["sig"]["HR"], e["start"] + a, e["start"] + b))
        keys = sorted([k for k in data if isinstance(k, int)])
        ordered = (["pre"] if "pre" in data else []) + keys + (["last"] if "last" in data else [])
        rows = []
        for k in ordered:
            g = data[k]
            pl, nl = interval_key_label(k, N, PROJECT_MAP_RAW[coh], nb_raw)
            rows.append([k if k in ("pre", "last") else f"{k}->{k + 1}", pl, nl, len(g["d"]),
                         fnum(med(g["d"]), 0), fnum(min(g["d"]), 0), fnum(max(g["d"]), 0),
                         fnum(cv_pct(g["d"]), 0), fnum(med(g["eda"]), 2), fnum(med(g["hr"]), 1)])
        rep.out(f"  Cohort {coh}")
        rep.table(["Interval", "project stage(s)", "notebook task", "n", "med dur", "min", "max",
                   "CV%", "med EDA", "med HR"], rows)
        rep.out("")


def section_alignment(rep, subs):
    rep.title("6. SENSOR WINDOW vs EDA-BASED TAG BOUNDARIES (project logic replicated)")
    rep.out("Project selection rule (from src/preprocessing.py):")
    rep.out("  EDA/HR/TEMP/BVP: index = int(t - THIS SIGNAL's own first timestamp);")
    rep.out("                   window = index in [int(tag_i_rel), int(tag_j_rel)]")
    rep.out("                   where tag_rel = tag_utc - EDA first timestamp.")
    rep.out("  IBI            : offset_s (relative to IBI's OWN header time) in [t0, t1).")
    rep.out("  ACC            : chunk counter (32 samples = 1 'second') from ACC's own start.")
    rep.out("So a signal whose header starts D seconds after EDA is read D seconds EARLY")
    rep.out("relative to the EDA-based boundaries:  shift = signal_start - EDA_start.")
    rep.out("A 'synthetic 10 s offset' therefore appears exactly when a header starts 10 s after EDA.")
    rep.out("")
    sigs = ["HR", "BVP", "TEMP", "ACC", "IBI"]
    counters = {s: Counter() for s in sigs}
    detail = []
    for sid, sub in subs.items():
        for f in sub["folders"]:
            rec = sub["recs"][f]
            e = eda_of(rec)
            if e is None:
                continue
            shifts = {}
            for s in sigs:
                si = rec["sig"][s]
                if si["status"] == "ok":
                    d = round(si["start"] - e["start"], 1)
                    counters[s][d] += 1
                    shifts[s] = d
            acc = rec["sig"]["ACC"]
            acc_ratio = (acc["fs"] / 32.0) if acc["status"] == "ok" else None
            worst, shortest = 0.0, None
            if not sub["split"] and tags_rel_of(rec):
                pw = project_windows(PROJECT_MAP_RAW.get(sub["cohort"], {}), tags_rel_of(rec), e["dur"])
                lens = [w["t1"] - w["t0"] for w in pw.values() if w["status"] == "bounded" and w["t1"] > w["t0"]]
                shortest = min(lens) if lens else None
            if shifts:
                worst = max(abs(v) for v in shifts.values())
            bad_acc = acc_ratio is not None and abs(acc_ratio - 1) > 1e-6
            if worst > TOL_START_S or bad_acc:
                pct = fnum(min(100.0, worst / shortest * 100), 0) if shortest else "NA"
                detail.append([f] + [fnum(shifts.get(s), 1, True) if s in shifts else "MISS" for s in sigs]
                              + [fnum(shortest, 0), pct, fnum(acc_ratio, 3)])
    rows = []
    for s in sigs:
        c = counters[s]
        nz = sum(v for k, v in c.items() if abs(k) > TOL_START_S)
        rows.append([s, sum(c.values()), nz,
                     ", ".join(f"{k:+g}s x{v}" for k, v in sorted(c.items())) or "none"])
    rep.out("6a. Distribution of (signal start - EDA start) across recordings")
    rep.table(["Signal", "n recordings", "n offset", "values (seconds x count)"], rows)
    rep.out("")
    rep.out("6b. Recordings with any offset > tolerance or ACC fs != 32 Hz")
    if detail:
        rep.table(["Recording"] + [f"shift {s}" for s in sigs]
                  + ["shortest window(s)", "worst % of it misplaced", "ACC fs/32"], detail)
        for d in detail:
            rep.flag("WINDOW_SHIFT", d[0], "signal windows shifted vs EDA boundaries (see 6b)")
    else:
        rep.out("  None: every parsed signal header starts within tolerance of EDA.")
    rep.out("  ('worst % misplaced' = max |shift| / shortest bounded project window, capped at 100.)")


def section_subtract_baseline(rep, subs, nb):
    rep.title("7. SUBTRACT WINDOW AND BASELINE BOUNDARIES")
    shift = 1 if nb["insert_start"] else 0
    spans = nb["spans"] or NOTEBOOK_T_FALLBACK
    rep.out("7a. Subtract window under the project logic")
    rep.out("  Rule: if the stage's end index == number of tags, t1 = None and the window runs")
    rep.out("  to the END OF THE SIGNAL.")
    rows = []
    for sid, sub in subs.items():
        if not analysable(sub):
            continue
        rec = sub["recs"][sub["folders"][0]]
        rel, e = tags_rel_of(rec), eda_of(rec)
        coh = sub["cohort"]
        pw = project_windows(PROJECT_MAP_RAW[coh], rel, e["dur"])
        w = pw["Subtract"]
        others = [x["t1"] - x["t0"] for k, x in pw.items() if k != "Subtract" and x["status"] == "bounded"]
        names, tsp = NOTEBOOK_STAGE_ORDER[coh], spans[coh]
        nw = notebook_windows(tsp, names, rel).get("Subtract")
        rows.append([sid, coh, len(rel), "%d->%d" % w["span"], w["status"],
                     fnum(w["t0"], 0) if w["t0"] is not None else "NA",
                     fnum(w["end"] - w["t0"], 0) if w["t0"] is not None else "NA",
                     fnum(nw[1] - nw[0], 0) if nw else "NA",
                     fnum(med(others), 0)])
        if w["status"] == "UNBOUNDED":
            rep.flag("UNBOUNDED_WINDOW", sid, f"Subtract runs {w['end'] - w['t0']:.0f}s to end of signal")
        if w["status"] == "SKIPPED":
            rep.flag("SKIPPED_WINDOW", sid, "Subtract skipped: not enough tags")
    rep.table(["Subject", "Coh", "n_tags", "project idx", "status", "t0(s)", "project len(s)",
               "notebook len(s)", "med other stages(s)"], rows)

    rep.out("")
    rep.out("7b. Baseline boundaries")
    exp = dict(EXPECTED_TAGS_FALLBACK)
    exp.update(nb.get("tag_counts") or {})
    for coh in ("V1", "V2"):
        pre, i0, i1 = [], [], []
        for sid, sub in subs.items():
            if sub["cohort"] != coh or not analysable(sub):
                continue
            rel = tags_rel_of(sub["recs"][sub["folders"][0]])
            pre.append(rel[0])
            if len(rel) > 1:
                i0.append(rel[1] - rel[0])
            if len(rel) > 2:
                i1.append(rel[2] - rel[1])
        s, e = PROJECT_MAP_RAW[coh]["Baseline"]
        rep.out(f"  {coh}: project Baseline = tags {s}->{e}. It starts at the first button press,")
        rep.out(f"      NOT at the session start; the pre-tag period (EDA start -> tag 0) is excluded.")
        rep.table(["Candidate interval", "n", "median(s)", "min(s)", "max(s)"],
                  [["EDA start -> tag 0", len(pre), fnum(med(pre), 0), fnum(min(pre), 0) if pre else "NA", fnum(max(pre), 0) if pre else "NA"],
                   ["tag 0 -> tag 1", len(i0), fnum(med(i0), 0), fnum(min(i0), 0) if i0 else "NA", fnum(max(i0), 0) if i0 else "NA"],
                   ["tag 1 -> tag 2", len(i1), fnum(med(i1), 0), fnum(min(i1), 0) if i1 else "NA", fnum(max(i1), 0) if i1 else "NA"]])
        rep.out(f"      Project Baseline spans {e - s} inter-tag interval(s) in {coh}.")
    rep.out("  The two cohorts use a different number of intervals for the stage called 'Baseline'.")
    rep.out("  The notebook gives no Baseline definition, so this cannot be checked against it.")


def section_split(rep, subs, nb):
    rep.title("8. SPLIT RECORDINGS (e.g. f14_a / f14_b): facts for a correct merge")
    split = [s for s in subs.values() if s["split"]]
    if not split:
        rep.out("  No split recordings found among the selected subjects.")
        return
    shift = 1 if nb["insert_start"] else 0
    spans = nb["spans"] or NOTEBOOK_T_FALLBACK
    exp = dict(EXPECTED_TAGS_FALLBACK)
    exp.update(nb.get("tag_counts") or {})
    for sub in split:
        sid, parts = sub["id"], sub["folders"]
        rep.out(f"--- {sid}: folders found = {parts}")
        if sid in parts:
            rep.out(f"  note: a combined folder '{sid}' ALSO exists; check which one is authoritative.")
            rep.flag("SPLIT_SESSION", sid, "both combined and part folders exist")
        else:
            rep.out(f"  No '{sid}' folder exists -> loaders that only iterate over existing folders")
            rep.out(f"  will never see this subject unless split parts are mapped to it explicitly.")
            rep.flag("SPLIT_SESSION", sid, "no combined folder; subject only exists as parts")
        rows, tagrows = [], []
        for p in parts:
            rec = sub["recs"][p]
            for name in SIGNAL_ORDER:
                s = rec["sig"][name]
                rows.append([p, name, s["status"], iso(s["start"]), iso(s["end"]),
                             fnum(s["dur"], 0), s["n"]])
            e = eda_of(rec)
            tg = rec["tags"]["abs"]
            tagrows.append([p, len(tg), ", ".join(iso(t) for t in tg) or "-",
                            ", ".join(fnum(t - e["start"], 0) for t in tg) if (e and tg) else "-"])
        rep.table(["Part", "Signal", "status", "start (UTC)", "end (UTC)", "dur(s)", "rows"], rows)
        rep.out("")
        rep.table(["Part", "n_tags", "tag times (UTC)", "rel. own EDA start (s)"], tagrows)
        # gap / overlap
        rep.out("")
        ordered = sorted(parts, key=lambda p: (eda_of(sub["recs"][p]) or {"start": 1e18})["start"])
        for a, b in zip(ordered, ordered[1:]):
            gap_rows = []
            for name in SIGNAL_ORDER:
                sa, sb = sub["recs"][a]["sig"][name], sub["recs"][b]["sig"][name]
                if sa["status"] == "ok" and sb["status"] == "ok":
                    g = sb["start"] - sa["end"]
                    gap_rows.append([name, fnum(g, 1, True),
                                     "GAP (unrecorded time)" if g > TOL_END_S else
                                     "OVERLAP" if g < -TOL_END_S else "contiguous"])
                else:
                    gap_rows.append([name, "NA", "missing in a part"])
            rep.out(f"  Gap = start({b}) - end({a}); positive = unrecorded time, negative = overlap")
            rep.table(["Signal", "gap (s)", "verdict"], gap_rows)
        # tag containment and merged tag list (DIAGNOSTIC ONLY)
        ea = eda_of(sub["recs"][ordered[0]])
        merged, where = [], []
        for p in ordered:
            rec = sub["recs"][p]
            e = eda_of(rec)
            for t in rec["tags"]["abs"]:
                inside = e is not None and e["start"] - TOL_START_S <= t <= e["end"] + TOL_END_S
                where.append((t, p, inside))
                merged.append(t)
        rep.out("")
        outside = [iso(t) for t, p, ins in where if not ins]
        rep.out(f"  Tags inside their own part's EDA interval: {sum(1 for _, _, i in where if i)}/{len(where)}")
        if outside:
            rep.out(f"  Tags OUTSIDE their part's EDA interval: {outside}")
            rep.flag("SPLIT_SESSION", sid, f"{len(outside)} tag(s) outside own part interval")
        total = len(merged)
        want = exp.get(sub["cohort"])
        rep.out(f"  Combined tag count = {total}; expected for {sub['cohort']} = {want} -> "
                f"{'consistent' if total == want else 'INCONSISTENT'}")
        if total != want:
            rep.flag("TAG_COUNT", sid, f"combined tags {total} vs expected {want}")
        mono = all(merged[i] <= merged[i + 1] for i in range(len(merged) - 1))
        rep.out(f"  Tags chronological across parts (in part order): {mono}")
        if ea is not None and merged:
            rel = [t - ea["start"] for t in merged]
            rep.out("  DIAGNOSTIC merged tag times relative to FIRST part's EDA start (s): "
                    + ", ".join(f"{x:.0f}" for x in rel))
            eb = eda_of(sub["recs"][ordered[-1]]) if len(ordered) > 1 else None
            gap0 = (eb["start"] - ea["end"]) if eb else 0.0
            ga, gb = ea["end"] - ea["start"], (eb["start"] - ea["start"]) if eb else 1e18
            rows = []
            cohort = sub["cohort"]
            end_rel = (eb["end"] - ea["start"]) if eb else ea["dur"]
            pw = project_windows(PROJECT_MAP_RAW.get(cohort, {}), rel, end_rel)
            for stage, w in pw.items():
                if w["status"] == "SKIPPED":
                    rows.append(["project", stage, "SKIPPED", "-", "-", "-"])
                    continue
                t0, t1 = w["t0"], w["end"]
                part0 = "A" if t0 < ga else ("GAP" if t0 < gb else "B")
                part1 = "A" if t1 <= ga else ("GAP" if t1 < gb else "B")
                touches = gap0 > TOL_END_S and (t0 < gb and t1 > ga)
                rows.append(["project", stage, f"{t0}..{t1}" + (" (open)" if w["status"] == "UNBOUNDED" else ""),
                             part0, part1, "YES" if touches else "no"])
                if touches:
                    rep.flag("SPLIT_SESSION", sid, f"project window {stage} overlaps the unrecorded gap")
            for stage, w in notebook_windows(spans.get(cohort, []), NOTEBOOK_STAGE_ORDER.get(cohort, []), rel).items():
                if w is None:
                    rows.append(["notebook", stage, "n/a", "-", "-", "-"])
                    continue
                t0, t1 = w
                part0 = "A" if t0 < ga else ("GAP" if t0 < gb else "B")
                part1 = "A" if t1 <= ga else ("GAP" if t1 < gb else "B")
                touches = gap0 > TOL_END_S and (t0 < gb and t1 > ga)
                rows.append(["notebook", stage, f"{t0:.0f}..{t1:.0f}", part0, part1, "YES" if touches else "no"])
                if touches:
                    rep.flag("SPLIT_SESSION", sid, f"notebook window {stage} overlaps the unrecorded gap")
            rep.out("")
            rep.table(["Mapping", "Stage", "window (s from part-A EDA start)", "t0 in", "t1 in",
                       "touches gap?"], rows)
        rep.out("")
        rep.out("  Requirements for a correct merge (derived from the facts above; NOT applied):")
        rep.out("   - Align parts on ABSOLUTE UTC time, never by appending rows by index.")
        rep.out("   - Leave any positive gap as missing data; do not interpolate across it.")
        rep.out("   - IBI offsets are relative to each part's own IBI header: convert to UTC per part first.")
        rep.out("   - Merge tags chronologically and measure them from ONE reference (part A's EDA start).")
        rep.out("   - Windows touching the gap need an explicit decision (drop or trim).")
        dc = PROJECT_ROOT / "data_constraints.txt"
        try:
            with open(dc, "r", encoding="utf-8-sig", errors="replace") as fh:
                lines = [ln.strip() for ln in fh if sid in ln]
            for ln in lines:
                rep.out(f"   data_constraints.txt: {ln}")
        except Exception:
            rep.out("   (data_constraints.txt not readable)")
        rep.out("   From code review (not measured here): src/loader.load_all() iterates only over existing")
        rep.out("   folders, so a subject that exists only as '_a'/'_b' parts is not loaded; and concatenated")
        rep.out("   IBI keeps per-part offsets, which restart near 0 in the second part.")


def section_labels(rep, subs):
    rep.title("9. LABEL FILES vs FOLDERS")
    have = set(subs)
    for coh, fname in (("V1", "Stress_Level_v1.csv"), ("V2", "Stress_Level_v2.csv")):
        try:
            with open(PROJECT_ROOT / fname, "r", encoding="utf-8-sig", errors="replace", newline="") as fh:
                ids = {row[0].strip() for i, row in enumerate(csv.reader(fh)) if i > 0 and row and row[0].strip()}
        except Exception as exc:
            rep.out(f"  {fname}: cannot read ({exc})")
            continue
        folders = {s for s, v in subs.items() if v["cohort"] == coh}
        rep.out(f"  {fname}: {len(ids)} subjects; folders found for this cohort: {len(folders)}")
        for s in sorted(ids - have):
            rep.out(f"    in labels but NO folder loaded: {s}")
            rep.flag("LABEL_MISMATCH", s, f"{fname} has subject but no STRESS folder analysed")
        for s in sorted(folders - ids):
            rep.out(f"    folder present but NO label row: {s}")
            rep.flag("LABEL_MISMATCH", s, f"STRESS folder without row in {fname}")


def section_summary(rep):
    rep.title("10. SUMMARY OF FLAGS")
    if not rep.flags:
        rep.out("  No inconsistencies flagged.")
    else:
        groups = defaultdict(list)
        for cat, where, msg in rep.flags:
            groups[cat].append((where, msg))
        rows = []
        for cat in sorted(groups):
            wheres = sorted({w for w, _ in groups[cat]})
            txt = ", ".join(wheres[:12]) + (" ..." if len(wheres) > 12 else "")
            rows.append([cat, len(groups[cat]), len(wheres), txt])
        rep.table(["Category", "n flags", "n places", "where (first 12)"], rows)
    rep.out("")
    rep.out("Flags are observations only. Nothing was changed. Share this report to decide fixes.")


def section_appendix(rep, subs):
    rep.out("", console=False)
    rep.out("=" * 78, console=False)
    rep.out("APPENDIX A (file only): raw header values and full tag lists", console=False)
    rep.out("=" * 78, console=False)
    for sid, sub in subs.items():
        for f in sub["folders"]:
            rec = sub["recs"][f]
            rep.out(f"[{f}]", console=False)
            for name in SIGNAL_ORDER:
                s = rec["sig"][name]
                rep.out(f"  {name:<5} status={s['status']:<10} hdr={s.get('hdr')} start={iso(s.get('start'))} "
                        f"fs={s.get('fs', '-')} rows={s['n']} issues={s['issues']}", console=False)
            e = eda_of(rec)
            tg = rec["tags"]["abs"]
            rep.out(f"  tags  n={len(tg)}", console=False)
            for i, t in enumerate(tg):
                rel = fnum(t - e["start"], 1) if e else "NA"
                rep.out(f"    tag[{i}] {iso(t)}  rel_EDA={rel}s", console=False)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def find_stress_dir(arg):
    if arg:
        p = Path(arg).expanduser()
        if p.is_dir() and p.name.upper() == "STRESS":
            return p
        if (p / "STRESS").is_dir():
            return p / "STRESS"
        return None
    p = PROJECT_ROOT / "Wearable_Dataset" / "STRESS"
    return p if p.is_dir() else None


def run_section(rep, name, fn, *a):
    try:
        fn(rep, *a)
    except Exception as exc:
        rep.out(f"[section '{name}' failed: {type(exc).__name__}: {exc}]")
        rep.out(traceback.format_exc(), console=False)
        rep.flag("SECTION_ERROR", name, f"{type(exc).__name__}: {exc}")


def unique_new_path(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(1, 1000):
        cand = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not cand.exists():
            return cand
    raise FileExistsError("could not find a free report filename")


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Read-only time-alignment diagnostic.")
    ap.add_argument("--dataset", help="Path to Wearable_Dataset (or its STRESS folder)")
    ap.add_argument("--output", help="Report path (must not already exist)")
    ap.add_argument("--subjects", nargs="*", help="Only these canonical ids, e.g. S01 f14")
    args = ap.parse_args(argv)

    rep = Report()
    stress_dir = find_stress_dir(args.dataset)
    code = 0
    try:
        section_header(rep, args, stress_dir or "NOT FOUND")
        if stress_dir is None:
            rep.out("")
            rep.out("ERROR: STRESS folder not found. Expected "
                    f"{PROJECT_ROOT / 'Wearable_Dataset' / 'STRESS'} or pass --dataset.")
            code = 2
        else:
            nb = parse_notebook(PROJECT_ROOT / "Wearable_Dataset.ipynb")
            run_section(rep, "notebook", section_notebook, nb)
            subs = load_subjects(stress_dir, set(args.subjects) if args.subjects else None)
            rep.out("")
            rep.out(f"Loaded {len(subs)} subject id(s) from {sum(len(s['folders']) for s in subs.values())} folder(s).")
            if not subs:
                rep.out("No subject folders found.")
                code = 2
            else:
                run_section(rep, "inventory", section_inventory, subs)
                run_section(rep, "timing", section_timing, subs)
                run_section(rep, "tags", section_tags, subs, nb)
                run_section(rep, "mapping", section_mapping, subs, nb)
                run_section(rep, "alignment", section_alignment, subs)
                run_section(rep, "subtract/baseline", section_subtract_baseline, subs, nb)
                run_section(rep, "split", section_split, subs, nb)
                run_section(rep, "labels", section_labels, subs)
                run_section(rep, "appendix", section_appendix, subs)
        section_summary(rep)
    finally:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(args.output) if args.output else PROJECT_ROOT / "reports" / "diagnostics" / f"alignment_diagnostic_{ts}.txt"
        try:
            out = unique_new_path(out)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "x", encoding="utf-8") as fh:
                fh.write("\n".join(rep.lines) + "\n")
            print(f"\nReport saved to: {out}")
        except Exception as exc:
            print(f"\nCould not save report ({type(exc).__name__}: {exc}). Console output above is complete.")
    return code


if __name__ == "__main__":
    sys.exit(main())
