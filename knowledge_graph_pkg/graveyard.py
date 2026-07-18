"""
Model graveyard orchestration (ModelReduce Session 4).

The graveyard batch-probes a fleet of local models across many domains into
a single store -- the "eat a directory of models" workflow. It is built for
unattended runs on modest hardware:

* **Sequential** by default (one model in memory at a time) -- friendly to a
  fanless laptop; the caller controls concurrency.
* **Resume/checkpoint** -- every completed ``(model, domain)`` pair is
  recorded in ``store/graveyard_state.json`` so an interrupted run picks up
  where it left off instead of re-probing.
* **Fault-tolerant** -- a failure on one pair is recorded and the run
  continues; one bad model never aborts the batch.
* **Reportable** -- returns a :class:`GraveyardReport` with per-pair rows and
  a renderable progress table.

The actual probe+store step is injected as a ``prober`` callable
``(model, domain, store, **kw) -> n_facts`` so the orchestration is testable
without Ollama; the CLI supplies an Ollama-backed prober.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# Checkpoint key separator (NUL keeps it unambiguous vs. model/domain text).
_SEP = "\x00"


def _ckpt_path(store_dir: str) -> str:
    return os.path.join(store_dir, "graveyard_state.json")


def _load_completed(store_dir: str) -> set:
    path = _ckpt_path(store_dir)
    if os.path.isfile(path):
        try:
            data = json.loads(open(path, encoding="utf-8").read())
            return set(data.get("completed", []))
        except (json.JSONDecodeError, OSError):
            return set()
    return set()


def _save_completed(store_dir: str, completed: set) -> None:
    os.makedirs(store_dir, exist_ok=True)
    with open(_ckpt_path(store_dir), "w", encoding="utf-8") as fh:
        json.dump({"completed": sorted(completed)}, fh, ensure_ascii=False, indent=2)


@dataclass
class GraveyardReport:
    """Outcome of a graveyard run: counts plus per-pair rows."""

    probed: int = 0
    skipped: int = 0
    errors: int = 0
    total_facts: int = 0
    rows: List[Dict[str, Any]] = field(default_factory=list)

    def render(self) -> str:
        """Render a simple aligned progress table."""
        header = f"{'Model':<22} {'Domain':<14} {'Facts':>6}  Status"
        lines = [header, "-" * len(header)]
        for r in self.rows:
            lines.append(f"{str(r['model']):<22} {str(r['domain']):<14} "
                         f"{r['facts']:>6}  {r['status']}")
        lines.append("-" * len(header))
        lines.append(f"{'TOTAL':<22} {'':<14} {self.total_facts:>6}  "
                     f"probed={self.probed} skipped={self.skipped} errors={self.errors}")
        return "\n".join(lines)


def discover_ollama_models(lister: Optional[Callable[[], List[str]]] = None,
                           host: str = "http://localhost:11434",
                           exclude_embedding: bool = True) -> List[str]:
    """List local Ollama model names.

    ``lister`` can be injected for testing; by default it queries the Ollama
    client. Embedding-only models (e.g. ``mxbai-embed-large``) are excluded by
    default since they don't generate text facts.
    """
    if lister is None:
        def _default_lister() -> List[str]:
            import ollama
            client = ollama.Client(host=host)
            resp = client.list()
            models = resp.get("models", []) if isinstance(resp, dict) else getattr(resp, "models", [])
            names = []
            for m in models:
                name = m.get("model") or m.get("name") if isinstance(m, dict) else getattr(m, "model", None)
                if name:
                    names.append(name)
            return names
        lister = _default_lister

    names = lister()
    if exclude_embedding:
        names = [n for n in names if "embed" not in n.lower()]
    return names


def run_graveyard(models: List[str], domains: List[str], store_dir: str,
                  prober: Callable[..., int],
                  resume: bool = True, n_prompts: int = 10, seed: int = 42,
                  progress: bool = False, **prober_kwargs) -> GraveyardReport:
    """Probe every ``(model, domain)`` pair into the store at ``store_dir``.

    Args:
        models / domains: the grid to probe.
        store_dir: knowledge store directory (checkpoint lives here too).
        prober: callable ``(model, domain, store, n_prompts=, seed=, **kw)``
            returning the number of facts written. Injected for testability.
        resume: skip pairs recorded in the checkpoint (default True).
        progress: print each pair's outcome as it happens.

    Returns a :class:`GraveyardReport`. Failures are recorded per-pair and do
    not abort the run.
    """
    from .store import KnowledgeStore

    store = KnowledgeStore(store_dir)
    completed = _load_completed(store_dir) if resume else set()
    report = GraveyardReport()

    for model in models:
        for domain in domains:
            key = f"{model}{_SEP}{domain}"
            if resume and key in completed:
                report.skipped += 1
                report.rows.append({"model": model, "domain": domain,
                                    "facts": 0, "status": "skipped"})
                if progress:
                    print(f"skip  {model}/{domain} (already done)")
                continue
            try:
                n_facts = prober(model, domain, store,
                                 n_prompts=n_prompts, seed=seed, **prober_kwargs)
                report.probed += 1
                report.total_facts += int(n_facts or 0)
                report.rows.append({"model": model, "domain": domain,
                                    "facts": int(n_facts or 0), "status": "ok"})
                completed.add(key)
                _save_completed(store_dir, completed)  # checkpoint after each success
                if progress:
                    print(f"ok    {model}/{domain}: {n_facts} facts")
            except Exception as exc:  # noqa: BLE001 - one bad pair must not abort the batch
                report.errors += 1
                report.rows.append({"model": model, "domain": domain,
                                    "facts": 0, "status": f"error: {exc}"})
                if progress:
                    print(f"ERROR {model}/{domain}: {exc}")
    return report
