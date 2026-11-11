#!/usr/bin/env python3
"""Auto-push new CBIR notebook versions (B1, B2, ...) to Kaggle and collect results.

Each run:
  1. For every kernel in state "running": check status; if complete, download
     outputs to results/<version>/ and mark "complete" (or "error").
  2. For every notebooks/b<N>_*.ipynb (N >= 1) that is new or changed (sha256),
     push it to Kaggle (GPU T4, internet on, private) and mark "running".
  3. Optionally poll running kernels for up to --wait minutes.
  4. Print a JSON summary (new results vs B0) on stdout's last line.

State: kaggle_runs.json. Needs KAGGLE_USERNAME / KAGGLE_KEY env vars.
"""
import argparse, csv, hashlib, json, os, re, shutil, subprocess, sys, tempfile, time
from datetime import datetime, timezone
from pathlib import Path

# Repo root (override with CBIR_ROOT). Notebooks live in <root>/notebooks/.
ROOT = Path(os.environ.get("CBIR_ROOT", Path(__file__).resolve().parents[1]))
NB_DIR = ROOT / "notebooks"
STATE = ROOT / "kaggle_runs.json"
RESULTS = ROOT / "results"
B0 = {"roxford5k": (65.15, 45.31), "rparis6k": (87.21, 73.33)}  # GeM ms+Lw M/H
# Kaggle datasets mounted into every run (ROxf/RPar images + gnd; avoids flaky robots.ox.ac.uk)
DATASETS = ["rofiqsamanhudi/rparis6k-and-roxford5k"]
NB_RE = re.compile(r"^b([1-9]\d*)_.*\.ipynb$", re.I)
USER = os.environ.get("KAGGLE_USERNAME", "")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sh(*args, check=True):
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} failed: {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout + r.stderr


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {"runs": {}}


def save_state(s):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, indent=2, ensure_ascii=False))
    tmp.replace(STATE)


def slugify(stem):
    return "cbir-" + re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")


def push(nb: Path, version: str):
    slug = slugify(nb.stem)
    title = slug.replace("-", " ")  # Kaggle derives the slug from the title
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(nb, Path(d) / f"{slug}.ipynb")
        meta = {
            "id": f"{USER}/{slug}", "title": title, "code_file": f"{slug}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
            "dataset_sources": DATASETS, "kernel_sources": [], "competition_sources": [],
            "model_sources": [], "machine_shape": "NvidiaTeslaT4",
        }
        (Path(d) / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        out = sh("kernels", "push", "-p", d, "--accelerator", "NvidiaTeslaT4")
    return f"{USER}/{slug}", out.strip().splitlines()[-1] if out.strip() else ""


def status(ref):
    out = sh("kernels", "status", ref, check=False)
    m = re.search(r"KernelWorkerStatus\.(\w+)", out)  # only trust the real status token;
    if not m:                                           # e.g. a 429 "Client Error" must not count as error
        return "unknown"
    k = m.group(1).lower()
    return {"cancel_acknowledged": "cancel", "cancel_requested": "running"}.get(k, k)


def summarize(csv_path: Path):
    """Return best GeM ms+Lw (fallback: any) M/H per dataset, plus delta vs B0."""
    rows = list(csv.DictReader(csv_path.open()))
    out = {}
    for ds in B0:
        cand = [r for r in rows if r.get("dataset") == ds]
        pref = [r for r in cand if r.get("scale") == "ms" and str(r.get("Lw")) == "True"] or cand
        if not pref:
            continue
        best = max(pref, key=lambda r: float(r.get("M_mAP") or 0))
        m, h = float(best["M_mAP"]), float(best["H_mAP"])
        out[ds] = {"head": best.get("head"), "M": m, "H": h,
                   "dM": round(m - B0[ds][0], 2), "dH": round(h - B0[ds][1], 2)}
    return out


LOG_RE = re.compile(r"(\S+)\s+(roxford5k|rparis6k)\s+(\w+)\s+(ss|ms\w*) Lw=([01]) \| mAP E\s*([\d.]+)\s+M\s*([\d.]+)\s+H\s*([\d.]+)")


def fetch_log(ref, dest):
    sh("kernels", "output", ref, "-p", str(dest), "-o", "-q", "--file-pattern", "zzz_no_output_match", check=False)
    logs = sorted(dest.glob("*.log"))
    if not logs:
        return ""
    t = logs[0].read_text(errors="replace")
    try:
        return "".join(x.get("data", "") for x in json.loads(t))
    except Exception:
        return t


def log_to_csv(ref, dest):
    rows = {}
    for m in LOG_RE.finditer(fetch_log(ref, dest)):
        st, ds, head, sc, lw, e, mm, hh = m.groups()
        rows[(st, ds, head, sc, lw)] = dict(stage=st, dataset=ds, head=head, scale=sc, Lw=str(lw == "1"),
                                            E_mAP=e, M_mAP=mm, H_mAP=hh)
    if not rows:
        return []
    out = dest / "results_from_log.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(rows.values())).keys()))
        w.writeheader(); w.writerows(rows.values())
    return [out]


def collect(ref, version, run):
    dest = RESULTS / version
    dest.mkdir(parents=True, exist_ok=True)
    sh("kernels", "output", ref, "-p", str(dest), "-o", "-q",
       "--file-pattern", r"(results\.(csv|json)|.*\.log)$", check=False)
    found = list(dest.rglob("results.csv"))
    if not found:
        # Fallback: output files are served from www.kaggleusercontent.com, which the sandbox may block.
        # The kernel log comes through the API, and score_all() prints every result row there.
        found = log_to_csv(ref, dest)
    if found:
        run["summary"] = summarize(found[0])
        run["results_path"] = str(found[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", type=int, default=0, help="minutes to poll running kernels")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not USER or not os.environ.get("KAGGLE_KEY"):
        sys.exit("KAGGLE_USERNAME/KAGGLE_KEY missing")

    s = load_state()
    events = []

    def check_running():
        for v, run in s["runs"].items():
            if run.get("state") != "running":
                continue
            st = status(run["ref"])
            if st in ("complete", "error", "cancel"):
                run["state"] = st
                run["finished_at"] = now()
                if st == "complete":
                    collect(run["ref"], v, run)
                else:
                    d = RESULTS / v / f"log_{now()[:19].replace(':', '')}"
                    d.mkdir(parents=True, exist_ok=True)
                    tail = fetch_log(run["ref"], d)[-3000:]
                    run["error_tail"] = tail[-800:]
                events.append({"version": v, "event": st, "ref": run["ref"],
                               "summary": run.get("summary")})
                save_state(s)

    check_running()

    for nb in sorted(NB_DIR.glob("*.ipynb")):
        m = NB_RE.match(nb.name)
        if not m:
            continue
        v = f"B{m.group(1)}"
        h = hashlib.sha256(nb.read_bytes()).hexdigest()[:16]
        run = s["runs"].get(v, {})
        if run.get("sha") == h and run.get("state") in ("running", "complete", "error", "cancel"):
            continue  # same notebook: never re-run a finished/failed version unless the file changed
        if a.dry_run:
            events.append({"version": v, "event": "would_push", "file": nb.name})
            continue
        try:
            ref, msg = push(nb, v)
            s["runs"][v] = {"file": nb.name, "sha": h, "ref": ref, "state": "running",
                            "pushed_at": now(), "push_msg": msg}
            events.append({"version": v, "event": "pushed", "ref": ref})
        except Exception as e:  # keep going with other notebooks
            s["runs"][v] = {**run, "file": nb.name, "sha": h, "state": "push_failed",
                            "error": str(e)[:500], "pushed_at": now()}
            events.append({"version": v, "event": "push_failed", "error": str(e)[:300]})
        save_state(s)

    deadline = time.time() + a.wait * 60
    while time.time() < deadline and any(r.get("state") == "running" for r in s["runs"].values()):
        time.sleep(120)
        check_running()

    running = [v for v, r in s["runs"].items() if r.get("state") == "running"]
    print(json.dumps({"events": events, "still_running": running}, ensure_ascii=False))


if __name__ == "__main__":
    main()
