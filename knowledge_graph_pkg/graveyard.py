"""Model graveyard orchestration (ModelReduce Session 4)."""

from dataclasses import dataclass, field
import json
import os
from typing import Any, Callable, Dict, List, Optional

_SEP = "\x00"


def _ckpt_path(store_dir: str) -> str:
    return os.path.join(store_dir, "graveyard_state.json")


def _load_completed(store_dir: str) -> set:
    path = _ckpt_path(store_dir)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return set(data.get("completed", []))
        except (json.JSONDecodeError, OSError):
            return set()
    return set()


def _save_completed(store_dir: str, completed: set) -> None:
    os.makedirs(store_dir, exist_ok=True)
    path = _ckpt_path(store_dir)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump({"completed": sorted(completed)}, fh, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


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
            lines.append(
                f"{str(r['model']):<22} {str(r['domain']):<14} "
                f"{r['facts']:>6}  {r['status']}"
            )
        lines.append("-" * len(header))
        lines.append(
            f"{'TOTAL':<22} {'':<14} {self.total_facts:>6}  "
            f"probed={self.probed} skipped={self.skipped} errors={self.errors}"
        )
        return "\n".join(lines)


def discover_ollama_models(
    lister: Optional[Callable[[], List[str]]] = None,
    host: str = "http://localhost:11434",
    exclude_embedding: bool = True,
    timeout: float = 30.0,
) -> List[str]:
    """List local Ollama model names, excluding embedding models by default."""
    if lister is None:

        def _default_lister() -> List[str]:
            import ollama

            try:
                client = ollama.Client(host=host, timeout=timeout)
            except TypeError:
                client = ollama.Client(host=host)
            resp = client.list()
            models = (
                resp.get("models", [])
                if isinstance(resp, dict)
                else getattr(resp, "models", [])
            )
            names = []
            for m in models:
                name = (
                    m.get("model") or m.get("name")
                    if isinstance(m, dict)
                    else getattr(m, "model", None)
                )
                if name:
                    names.append(name)
            return names

        lister = _default_lister

    names = lister()
    if exclude_embedding:
        names = [n for n in names if "embed" not in n.lower()]
    return names


def run_graveyard(
    models: List[str],
    domains: List[str],
    store_dir: str,
    prober: Callable[..., int],
    resume: bool = True,
    n_prompts: int = 10,
    seed: int = 42,
    progress: bool = False,
    **prober_kwargs,
) -> GraveyardReport:
    """Probe every ``(model, domain)`` pair into ``store_dir`` with checkpoints."""
    from .store import KnowledgeStore

    store = KnowledgeStore(store_dir)
    persisted = _load_completed(store_dir)
    report = GraveyardReport()

    for model in models:
        for domain in domains:
            key = f"{model}{_SEP}{domain}"
            if resume and key in persisted:
                report.skipped += 1
                report.rows.append(
                    {"model": model, "domain": domain, "facts": 0, "status": "skipped"}
                )
                if progress:
                    print(f"skip  {model}/{domain} (already done)")
                continue
            try:
                n_facts = int(
                    prober(
                        model,
                        domain,
                        store,
                        n_prompts=n_prompts,
                        seed=seed,
                        **prober_kwargs,
                    )
                    or 0
                )
                report.probed += 1
                report.total_facts += n_facts
                status = "ok" if n_facts > 0 else "empty"
                report.rows.append(
                    {
                        "model": model,
                        "domain": domain,
                        "facts": n_facts,
                        "status": status,
                    }
                )
                if n_facts > 0:
                    persisted.add(key)
                    _save_completed(store_dir, persisted)
                if progress:
                    print(f"{status:<5} {model}/{domain}: {n_facts} facts")
            except Exception as exc:  # noqa: BLE001 - one bad pair must not abort the batch
                report.errors += 1
                report.rows.append(
                    {
                        "model": model,
                        "domain": domain,
                        "facts": 0,
                        "status": f"error: {exc}",
                    }
                )
                if progress:
                    print(f"ERROR {model}/{domain}: {exc}")
    return report
