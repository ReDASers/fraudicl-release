"""Matched 4-shot random-retrieval ablation.

Question: how much of the FRAUD-ICL gain over zero-shot comes from HAVING four labeled
demonstrations, versus SELECTING them by cosine similarity to the query?

This script runs the whole ablation. It prepares inputs, refuses to start
unless they match the cosine arm, waits for a free GPU, generates, verifies, and records
everything it did under Outputs/random-retrieval/provenance/.

    uv run python Scripts/run_random_retrieval.py                  # everything
    uv run python Scripts/run_random_retrieval.py --prepare-only   # no GPU work

Safe to re-run at any time: finished rows are kept and skipped, and --per-row-seed makes a
resumed row draw the same demonstrations as an uninterrupted run.

DESIGN -- one factor (retrieval) changes; the rest is the canonical cosine cell of
Outputs/full-run:

    retrieval   random: uniform draw of k=4 chunks from the 123,597-chunk knowledge base
    order       in_order
    definition  1
    format      new (FRAUD-ICL prompt)
    backbones   neural_chat, mistral_instruct, gemma, qwen3; 4-bit NF4, greedy, batch size 1
    queries     the released 500-example sub-sample, a strict subset of the 3,000 evaluation
                queries, so each row pairs with the cosine row for the same query
    seeding     --per-row-seed (base seed 99)
    grounding   off (backfill later with Scripts/backfill_grounding.py if wanted)
    zero-shot   off; already exists for these queries. Must be disabled explicitly, because
                main.py defaults --zero-shot to ON.

BASELINE -- the main grid (Outputs/full-run, pi=24, kappa=4). check_baseline() refuses to start
unless the code is configured as that run was. If the four cosine files are byte-identical to
the ones analyzed in the paper (pinned SHA-256) that is logged; otherwise a warning says the
comparison is against your own regeneration of the cosine arm.

pi and kappa are DEmO *ordering* settings. This arm is in_order, so they do not enter any of its
prompts; they identify which run it is paired with. The raw CSVs do not record them, so the check
covers the code constants and the file identity, not what the generating process had in memory.

TWO INPUTS THAT ARE NOT TAKEN FROM ./Saved AS THEY ARE:

  * The knowledge base is re-tagged from LABEL_MAPPING (chunks, order, labels and embeddings
    unchanged), so demonstration labels render exactly as in the main grid's prompts
    ("Label: non-deceptive") even if the local store was built by an older version.
  * A local quantized snapshot under Saved/ would be picked up silently by main.py. The main
    grid quantized from the Hub on the fly, so this script does too, by pointing
    --saved-path at a clean directory that holds no snapshots.

The clean directory and the model cache live under ~/.cache, outside the repository.

STAYING OUT OF THE WAY (--cpu-cap, --max-cpus, --min-free-ram-gb):

  * Processor affinity, inherited by every child, is a hard ceiling: at most
    min(max_cpus, floor(cpu_cap% * logical CPUs)) CPUs can ever be used. BELOW_NORMAL priority
    yields to interactive programs.
  * Generation stops, waits and resumes if free system RAM drops below --min-free-ram-gb.
  * CPU, RAM and GPU use are sampled every 30 s into provenance/resource_log.csv.
  * The GPU runs near 100% while generating; that is inherent.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
PY = Path(sys.executable)   # the interpreter running this script (uv run python ...)
OUT = ROOT / "Outputs" / "random-retrieval"
CONTROL_OUT = ROOT / "Outputs" / "random-retrieval-cosine-control"
PROV = OUT / "provenance"
SAMPLES = OUT / "samples.csv"
FULL_RUN = ROOT / "Outputs" / "full-run"
SRC_STORE = ROOT / "Saved" / "document_store"
SRC_DATASETS = ROOT / "Saved" / "datasets"

HF_HOME = Path.home() / ".cache" / "huggingface"
CLEAN_SAVED = Path.home() / ".cache" / "fraudicl" / "random-retrieval" / "Saved"

N_ROWS = 500
# key -> (output label, hub repo). Cheapest first, so results arrive early.
MODELS = {
    "neural_chat": ("Neural", "Intel/neural-chat-7b-v3-3"),
    "mistral_instruct": ("Mistral", "mistralai/Mistral-7B-Instruct-v0.2"),
    "gemma": ("Gemma", "google/gemma-2-9b-it"),
    "qwen3": ("Qwen", "Qwen/Qwen3-8B"),
}
# The paper's Def 1, FRAUD-ICL, cosine files: SHA-256 of the files analyzed in the paper
RERUN_PI, RERUN_KAPPA, RERUN_K = 24, 4, 4
BASELINE_SHA256 = {
    "Neural": "c60f7f36c3b1fb8d9e083b5ec0deea8c4dc2f62e35e4090d5a21e289824b41d4",
    "Mistral": "3e828d28aeb996773119fc7c4efb285fbd9d5e30c4462f7339c661c5ce5198a1",
    "Gemma": "8863245fb19ec40e63c281ebbc6a324c12c442840e267fe890f14bf90f79d2e7",
    "Qwen": "87452ad478fbb57dccfc5f33915803c2a6df36b4839c08457c9923770d45ec86",
}
EMBEDDER = "avsolatorio/GIST-small-Embedding-v0"
CODE_FILES = ["Scripts/run_random_retrieval.py", "Scripts/verify_random_retrieval.py", "main.py",
              "pyproject.toml", "uv.lock"]


def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def die(msg: str) -> None:
    log(f"STOP: {msg}")
    raise SystemExit(1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- resource limits

def apply_resource_limits(cpu_cap: float, max_cpus: int) -> list[int]:
    """Pin this process (and so every child) to a bounded CPU set, at below-normal priority."""
    proc = psutil.Process()
    total = psutil.cpu_count(logical=True)
    n = max(1, min(max_cpus, math.floor(total * cpu_cap / 100)))
    cpus = list(range(total - n, total))   # highest-numbered CPUs
    proc.cpu_affinity(cpus)
    proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    log(f"CPU limit: {n} of {total} logical CPUs ({100 * n / total:.0f}% ceiling, cap {cpu_cap:.0f}%), below-normal priority")
    return cpus


def keep_awake(on: bool) -> None:
    """Stop Windows sleeping mid-run (does not keep the display on)."""
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))


def child_env(offline: bool) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update({
        "HF_HOME": str(HF_HOME), "HF_HUB_CACHE": str(HF_HOME / "hub"), "HF_HUB_DISABLE_XET": "1",
        "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false",
        "HAYSTACK_TELEMETRY_ENABLED": "False",   # no outbound call per pipeline run
    })
    if offline:   # freeze model revisions to what was prefetched and recorded in the manifest
        env["HF_HUB_OFFLINE"] = "1"
    return env


def smi(query: str, kind: str = "gpu") -> list[list[str]]:
    r = subprocess.run(["nvidia-smi", f"--query-{kind}={query}", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    return [[c.strip() for c in line.split(",")] for line in r.stdout.splitlines() if line.strip()]


def family_pids() -> set[int]:
    me = psutil.Process()
    return {me.pid} | {c.pid for c in me.children(recursive=True)}


def other_gpu_python() -> list[int]:
    """PIDs of python processes using the GPU that are not ours."""
    ours = family_pids()
    return [int(p[0]) for p in smi("pid,process_name", "compute-apps")
            if len(p) >= 2 and "python" in p[1].lower() and int(p[0]) not in ours]


def wait_for_gpu(idle_minutes: float) -> None:
    """Return once no other python job has used the GPU for idle_minutes in a row, so a queue
    of jobs with short gaps between them is not mistaken for an idle GPU."""
    idle_since, announced = None, None
    while True:
        busy = other_gpu_python()
        if busy:
            idle_since = None
            if busy != announced:
                log(f"GPU in use by another python job (pid {busy}); waiting")
                announced = busy
        else:
            idle_since = idle_since or time.time()
            if time.time() - idle_since >= idle_minutes * 60:
                return
        time.sleep(30)


class Monitor(threading.Thread):
    """Samples resource use; stops the job if free RAM stays too low, or if its output file has
    not grown for stall_minutes (a hung process would otherwise block the run forever)."""

    def __init__(self, proc: subprocess.Popen, stage: str, min_free_gb: float, watch: Path, stall_minutes: float):
        super().__init__(daemon=True)
        self.proc, self.stage, self.min_free_gb = proc, stage, min_free_gb
        self.watch, self.stall_minutes = watch, stall_minutes
        self.low_ram = self.stalled = False
        self.done = threading.Event()

    def run(self) -> None:
        path = PROV / "resource_log.csv"
        new = not path.exists()
        ncpu, low, cache = psutil.cpu_count(), 0, {}
        last_size, last_change = -1, time.time()
        psutil.cpu_percent(None)
        while not self.done.wait(30):
            size = self.watch.stat().st_size if self.watch.exists() else -1
            if size != last_size:
                last_size, last_change = size, time.time()
            elif time.time() - last_change > self.stall_minutes * 60:
                self.stalled = True
                kill_tree(self.proc.pid)
                return
            try:
                root = psutil.Process(self.proc.pid)
                tree = [root] + root.children(recursive=True)
            except psutil.Error:
                return
            cpu = rss = 0.0
            for p in tree:
                try:
                    p = cache.setdefault(p.pid, p)
                    cpu += p.cpu_percent(None)
                    rss += p.memory_info().rss
                except psutil.Error:
                    pass
            vm = psutil.virtual_memory()
            gpu = (smi("utilization.gpu,memory.used,temperature.gpu") or [["", "", ""]])[0]
            with path.open("a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["time", "stage", "run_cpu_pct_of_machine", "system_cpu_pct",
                                "run_ram_gb", "free_ram_gb", "gpu_util_pct", "gpu_mem_mib",
                                "gpu_temp_c", "other_gpu_python_pids"])
                    new = False
                w.writerow([f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}", self.stage,
                            f"{cpu / ncpu:.1f}", f"{psutil.cpu_percent(None):.1f}",
                            f"{rss / 2**30:.2f}", f"{vm.available / 2**30:.1f}", *gpu,
                            " ".join(map(str, other_gpu_python()))])
            low = low + 1 if vm.available / 2**30 < self.min_free_gb else 0
            if low >= 3:
                self.low_ram = True
                kill_tree(self.proc.pid)
                return


def kill_tree(pid: int) -> None:
    try:
        root = psutil.Process(pid)
        procs = root.children(recursive=True) + [root]
    except psutil.Error:
        return
    for p in procs:
        try:
            p.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(procs, timeout=30)


# --------------------------------------------------------------------------- preparation

def acquire_lock() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / "run.lock"
    if lock.exists():
        try:
            pid = int(lock.read_text().split()[0])
            if psutil.pid_exists(pid) and "run_random_retrieval" in " ".join(psutil.Process(pid).cmdline()):
                die(f"already running (pid {pid}); delete {lock} only if that is wrong")
        except (ValueError, IndexError, psutil.Error):
            pass
    for p in psutil.process_iter(["pid", "cmdline"]):
        cmd = " ".join(p.info["cmdline"] or [])
        if "main.py" in cmd and "random-retrieval" in cmd:
            die(f"an orphaned generation process is still writing (pid {p.info['pid']}); stop it first")
    lock.write_text(f"{os.getpid()} {dt.datetime.now().isoformat()}")
    return lock


def archive_unmatched_outputs() -> None:
    """Move (never delete) rows produced without this script, i.e. with no manifest."""
    if (PROV / "manifest.json").exists() or not any(OUT.glob("*-Generation/*.csv")):
        return
    dest = ROOT / "Outputs" / f"random-retrieval-archived-{dt.datetime.now():%Y%m%d-%H%M%S}"
    dest.mkdir(parents=True)
    for p in OUT.iterdir():
        if p.name not in ("run.lock", "runner.log"):
            shutil.move(str(p), str(dest / p.name))
    log(f"moved existing rows with no manifest to {dest.name}")


def prepare_samples() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not SAMPLES.exists():
        die(f"{SAMPLES} is missing; it ships with the repository (the 500-example sub-sample)")
    import pandas as pd
    n = len(pd.read_csv(SAMPLES))
    if n != N_ROWS:
        die(f"expected {N_ROWS} samples, found {n}")
    log(f"samples: {n} rows, sha256 {sha256(SAMPLES)[:12]}")


def prepare_store() -> None:
    """Copy the knowledge base, replacing only `tags` with LABEL_MAPPING[label]."""
    import h5py
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from fraudicl.global_config import Database
    dst = CLEAN_SAVED / "document_store"
    CLEAN_SAVED.mkdir(parents=True, exist_ok=True)
    if not SRC_STORE.exists():
        die(f"knowledge base not found at {SRC_STORE}; run main.py once first (it builds it from the DiFrauD train split)")
    if not dst.exists():
        tmp = dst.with_suffix(".tmp")
        shutil.copy2(SRC_STORE, tmp)
        with h5py.File(tmp, "r+") as f:
            labels = [int(x) for x in f["labels"][:]]
            del f["tags"]
            f.create_dataset("tags", data=[Database.LABEL_MAPPING[x].encode() for x in labels])
        tmp.replace(dst)
    with h5py.File(SRC_STORE, "r") as a, h5py.File(dst, "r") as b:
        for k in ("contents", "domains", "labels", "embeddings"):
            if not np.array_equal(a[k][:], b[k][:]):
                die(f"derived store differs from source in {k}")
        bad = sum(t.decode() != Database.LABEL_MAPPING[int(x)] for t, x in zip(b["tags"][:], b["labels"][:]))
        if bad:
            die(f"{bad} tags in the derived store do not follow LABEL_MAPPING")
        log(f"knowledge base: {len(b['contents'])} chunks; contents, order, labels, embeddings identical to source; tags normalized")
    if SRC_DATASETS.exists() and not (CLEAN_SAVED / "datasets").exists():
        shutil.copytree(SRC_DATASETS, CLEAN_SAVED / "datasets")
    strays = [p.name for p in CLEAN_SAVED.iterdir() if "model" in p.name]
    if strays:
        die(f"unexpected model snapshots in {CLEAN_SAVED}: {strays}")


def prefetch_models(keys: list[str]) -> dict[str, str]:
    """Download weights (safetensors only) to the cache; return {repo: commit}."""
    os.environ.update({k: v for k, v in child_env(False).items() if k.startswith("HF_")})
    os.environ.pop("HF_HUB_OFFLINE", None)
    from dotenv import load_dotenv
    from huggingface_hub import snapshot_download
    load_dotenv(ROOT / ".env")
    commits = {}
    for repo in [MODELS[k][1] for k in keys] + [EMBEDDER]:
        patterns = ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"]
        kw = dict(allow_patterns=patterns, cache_dir=str(HF_HOME / "hub"), token=os.getenv("HF_TOKEN"), max_workers=2)
        for attempt in range(5):
            try:
                path = snapshot_download(repo, **kw)
                break
            except Exception as exc:
                log(f"download of {repo} failed ({exc.__class__.__name__}: {str(exc)[:160]}); retry {attempt + 1}/5")
                time.sleep(30)
        else:
            die(f"could not download {repo}; re-run to resume")
        commits[repo] = Path(path).name
        log(f"weights ready: {repo} @ {commits[repo][:10]}")
    return commits


def baseline_file(name: str) -> Path:
    return FULL_RUN / f"{name}-Generation" / "def_1_new_cosine_outputs.csv"


def check_baseline(keys: list[str]) -> None:
    """Gate: the code is configured like the main grid; report whether the cosine files are the paper's."""
    sys.path.insert(0, str(ROOT))
    from fraudicl.global_config import Order, Retrieval
    got = (Order.PI_SAMPLE, Order.TOP_K_ENTROPY, Retrieval.TOP_K)
    if got != (RERUN_PI, RERUN_KAPPA, RERUN_K):
        die(f"global_config has (pi, kappa, k) = {got}, not the paper's {(RERUN_PI, RERUN_KAPPA, RERUN_K)}")
    same = 0
    for key in keys:
        name = MODELS[key][0]
        if not baseline_file(name).exists():
            die(f"cosine arm missing: {baseline_file(name)}; generate the main grid first (Inference/run_experiment.sh)")
        if sha256(baseline_file(name)) == BASELINE_SHA256[name]:
            same += 1
        else:
            log(f"WARNING: {baseline_file(name).relative_to(ROOT).as_posix()} differs from the file analyzed in the paper; "
                "the comparison is against your own cosine arm")
    log(f"baseline gate: pi={RERUN_PI} kappa={RERUN_KAPPA} k={RERUN_K}; {same}/{len(keys)} cosine files identical to the paper's")


def check_matched(keys: list[str]) -> None:
    """Gate: for every cosine partner row, today's prompt code fed the same demonstrations from
    the derived store must reproduce the logged cosine prompt byte for byte. This is what makes
    random-vs-cosine differ in retrieval only."""
    import h5py
    import pandas as pd
    from types import SimpleNamespace as NS
    sys.path.insert(0, str(ROOT))
    from fraudicl.global_config import LLM
    from fraudicl.promptmanager import PromptManager
    with h5py.File(CLEAN_SAVED / "document_store", "r") as f:
        kb = {}
        for c, d, l, t in zip(f["contents"][:], f["domains"][:], f["labels"][:], f["tags"][:]):
            kb.setdefault(c.decode().strip(), set()).add((d.decode(), int(l), t.decode()))
    queries = set(pd.read_csv(SAMPLES).content)
    for key in keys:
        name = MODELS[key][0]
        c = pd.read_csv(baseline_file(name),
                        usecols=["order", "prompt", "raw_prompt", "retrieved_examples"], low_memory=False)
        c = c[(c.order == "in_order") & c.prompt.isin(queries)].drop_duplicates("prompt")
        if len(c) != N_ROWS:
            die(f"{name}: {len(c)} cosine partner rows, expected {N_ROWS}")
        pm = PromptManager(prompt_profile=LLM.get_model_profile(key)["prompt_profile"])
        bad_kb = bad_prompt = 0
        for _, r in c.iterrows():
            ex = json.loads(r.retrieved_examples)
            bad_kb += any((e["domain"], int(e["label"]), e["tag"]) not in kb.get(e["content"].strip(), ()) for e in ex)
            docs = [NS(content=e["content"], meta={"label": e["label"], "tag": e["tag"], "domain": e["domain"]}) for e in ex]
            bad_prompt += pm.run(1, False, docs, r.prompt.strip())["prompt"] != r.raw_prompt
        if bad_kb or bad_prompt:
            die(f"{name}: {bad_kb} rows with demos absent from the store (content/domain/label/tag), "
                f"{bad_prompt} prompts not reproduced; the arms would not be matched")
        log(f"matched-prompt gate: {name} {len(c)}/{N_ROWS} cosine prompts reproduced, all demos present in the store with the same tags")


def write_manifest(args: argparse.Namespace, commits: dict[str, str], cpus: list[int]) -> None:
    PROV.mkdir(parents=True, exist_ok=True)
    def git(*a: str) -> str:
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout.strip()
        except OSError:
            return ""
    versions = subprocess.run(
        [str(PY), "-c", "import json,torch,transformers,bitsandbytes,accelerate;"
         "print(json.dumps({m.__name__:m.__version__ for m in (torch,transformers,bitsandbytes,accelerate)}))"],
        capture_output=True, text=True, env=child_env(True)).stdout.strip().splitlines()
    code = CODE_FILES + [f"fraudicl/{p.name}" for p in sorted((ROOT / "fraudicl").glob("*.py"))]
    stamp = f"{dt.datetime.now():%Y%m%d-%H%M%S}"
    manifest = {
        "launched": dt.datetime.now().isoformat(), "host": platform.node(), "python": sys.version,
        "argv": sys.argv, "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
        "versions": json.loads(versions[-1]) if versions else None,
        "gpu": smi("name,driver_version,memory.total"), "cpu_affinity": cpus,
        "hub_commits": commits, "samples_sha256": sha256(SAMPLES),
        "source_store_sha256": sha256(SRC_STORE), "derived_store_sha256": sha256(CLEAN_SAVED / "document_store"),
        "baseline": {
            "run": "Outputs/full-run", "pi": RERUN_PI, "kappa": RERUN_KAPPA, "k": RERUN_K,
            "cosine_sha256": {MODELS[k][0]: sha256(baseline_file(MODELS[k][0])) for k in args.models},
            "zero_shot_sha256_at_launch": {
                MODELS[k][0]: sha256(baseline_file(MODELS[k][0]).with_name("def_-1_new_none_outputs.csv")) for k in args.models},
        },
        "code_sha256": {f: sha256(ROOT / f) for f in code},
        "commands": {k: random_command(k) for k in args.models},
    }
    text = json.dumps(manifest, indent=2)
    if (PROV / "manifest.json").exists():   # relaunch: keep the first, log this one beside it
        (PROV / f"manifest-relaunch-{stamp}.json").write_text(text, encoding="utf-8")
    else:
        (PROV / "manifest.json").write_text(text, encoding="utf-8")
    snap = PROV / f"code-{stamp}"
    for f in code:
        (snap / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, snap / f)
    (snap / "git-diff.patch").write_text(git("diff", "HEAD"), encoding="utf-8")
    log(f"provenance written to {PROV.relative_to(ROOT)}")


# --------------------------------------------------------------------------- generation

def main_command(key: str, retrieval: str, samples: Path, out: Path) -> list[str]:
    return [str(PY), "main.py",
            "--samples-path", str(samples), "--output-root", str(out),
            "--models", key, "--retrievals", retrieval, "--orders", "in_order",
            "--definitions", "1", "--formats", "new", "--batch-size", "1",
            "--no-hallucination-checker", "--no-zero-shot", "--per-row-seed",
            "--saved-path", str(CLEAN_SAVED), "--hf-cache-dir", str(HF_HOME),
            "--no-save-local-artifacts", "--device", "cuda:0", "--quantization", "4bit"]


def random_command(key: str) -> list[str]:
    return main_command(key, "random", SAMPLES, OUT)


def repair_and_count(path: Path, samples: Path) -> int:
    """Rows already logged. Drops a half-written final row (after a kill) and insists the rest
    are exactly the first n samples in order, which is what resume assumes."""
    import pandas as pd
    if not path.exists():
        return 0
    csv.field_size_limit(10**9)
    raw = path.read_bytes()
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:   # created but never written to: main.py will write the header itself
        return 0
    if not raw.endswith(b"\n"):
        shutil.copy2(path, path.with_suffix(f".partial-{dt.datetime.now():%H%M%S}"))
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows[:-1])
        log(f"dropped one half-written row from {path.name} (backup kept)")
        rows = rows[:-1]
    header, body = rows[0], rows[1:]
    if any(len(r) != len(header) for r in body):
        die(f"{path} has malformed rows")
    # dtype=str/keep_default_na: a sample whose text is e.g. "NA" must not become NaN here
    want = pd.read_csv(samples, dtype=str, keep_default_na=False).content.tolist()[:len(body)]
    if [r[header.index("prompt")] for r in body] != want:
        die(f"{path} rows are not the first {len(body)} samples in order")
    return len(body)


def generate(key: str, retrieval: str, samples: Path, out: Path, n: int, args: argparse.Namespace) -> None:
    name = MODELS[key][0]
    path = out / f"{name}-Generation" / f"def_1_new_{retrieval}_outputs.csv"
    failures = 0
    while True:
        have = repair_and_count(path, samples)
        if have >= n:
            log(f"{name}/{retrieval}: complete ({have}/{n})")
            return
        while psutil.virtual_memory().available / 2**30 < args.min_free_ram_gb + 4:
            log("waiting for free RAM")
            time.sleep(120)
        wait_for_gpu(0)
        log(f"{name}/{retrieval}: generating from row {have}/{n}")
        logfile = PROV / f"{name}-{retrieval}.log"
        with logfile.open("a", encoding="utf-8") as lf:
            proc = subprocess.Popen(main_command(key, retrieval, samples, out), cwd=ROOT,
                                    env=child_env(True), stdout=lf, stderr=subprocess.STDOUT,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            mon = Monitor(proc, f"{name}-{retrieval}", args.min_free_ram_gb, path, args.stall_minutes)
            mon.start()
            try:
                rc = proc.wait()
            finally:
                mon.done.set()
                kill_tree(proc.pid)
        if mon.low_ram:
            log(f"{name}: paused, free RAM under {args.min_free_ram_gb} GB; will resume")
            time.sleep(300)
            continue
        if mon.stalled:
            log(f"{name}: no output for {args.stall_minutes:g} min; process killed, resuming from the last logged row")
        now = repair_and_count(path, samples)
        if rc != 0 or now < n:
            failures = 0 if now > have else failures + 1
            log(f"{name}: exit code {rc} at row {now}/{n}; see {logfile.relative_to(ROOT)}")
            if failures >= 3:
                die(f"{name} failed 3 times without progress")
            time.sleep(60)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    ap.add_argument("--cpu-cap", type=float, default=85, help="hard CPU ceiling, %% of machine (default 85)")
    ap.add_argument("--max-cpus", type=int, default=8, help="logical CPUs to use, under the cap (default 8; the job needs ~1-2)")
    ap.add_argument("--min-free-ram-gb", type=float, default=8)
    ap.add_argument("--gpu-idle-minutes", type=float, default=5, help="GPU must be free of other python jobs this long before starting")
    ap.add_argument("--stall-minutes", type=float, default=30,
                    help="kill and resume a generation process whose output CSV has not grown for this long "
                         "(model load + quantization takes a few minutes; a row takes well under one)")
    ap.add_argument("--control-rows", type=int, default=50,
                    help="after the arm, regenerate this many COSINE rows per model on this machine, to measure how "
                         "far hardware alone moves the output (the cosine arm ran on other GPUs). 0 disables")
    ap.add_argument("--prepare-only", action="store_true", help="prepare and check inputs, then exit without the GPU")
    args = ap.parse_args()

    os.chdir(ROOT)
    lock = acquire_lock()
    try:
        cpus = apply_resource_limits(args.cpu_cap, args.max_cpus)
        keep_awake(True)
        archive_unmatched_outputs()
        check_baseline(args.models)
        prepare_samples()
        prepare_store()
        check_matched(args.models)
        commits = prefetch_models(args.models)
        write_manifest(args, commits, cpus)
        if args.prepare_only:
            log("prepared; not generating (--prepare-only)")
            return 0

        wait_for_gpu(args.gpu_idle_minutes)
        for key in args.models:
            generate(key, "random", SAMPLES, OUT, N_ROWS, args)

        if args.control_rows:
            import pandas as pd
            CONTROL_OUT.mkdir(parents=True, exist_ok=True)
            control = CONTROL_OUT / "samples.csv"
            # samples.csv is grouped by domain and label, so take evenly spaced rows, not the head
            s = pd.read_csv(SAMPLES)
            s.iloc[[i * N_ROWS // args.control_rows for i in range(args.control_rows)]].to_csv(control, index=False)
            for key in args.models:
                generate(key, "cosine", control, CONTROL_OUT, args.control_rows, args)

        names = [MODELS[k][0] for k in args.models]
        r = subprocess.run([str(PY), "Scripts/verify_random_retrieval.py", "--compare", "--models", *names],
                           cwd=ROOT, env=child_env(True), capture_output=True, text=True, encoding="utf-8")
        (PROV / "verify_report.txt").write_text(r.stdout + r.stderr, encoding="utf-8")
        print(r.stdout, flush=True)
        log(f"ALL DONE; verification {'passed' if r.returncode == 0 else 'FAILED'}: {PROV / 'verify_report.txt'}")
        return r.returncode
    finally:
        keep_awake(False)
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
