#!/usr/bin/env python3
"""Dispatcher: orchestrates PSD export and Gemini layer naming with concurrency."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required. Install it via pip install pyyaml.") from exc

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

if load_dotenv:
    env_path = ROOT_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)

from process_psd import ExportResult, LayerRecord, export_psd  # type: ignore  # noqa: E402
from gemini_client import GeminiRenamer, load_prompt, render_prompt  # type: ignore  # noqa: E402
from cleanup import cleanup_psd_tmp  # type: ignore  # noqa: E402

LOGGER = logging.getLogger("dispatcher")
QUOTA_COOLDOWN = float(os.environ.get("GEMINI_QUOTA_COOLDOWN", 30.0))


def load_config(path: Path) -> Dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Invalid config file")
    return data


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def write_results(result: ExportResult, mappings: List[Dict], results_dir: Path) -> None:
    output = results_dir / f"{Path(result.psd).stem}.json"
    payload = {
        "psd": result.psd,
        "composite": result.composite_png,
        "layers": mappings,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Results saved to %s", output)


def resolve_api_key(raw_value: Optional[str]) -> str:
    if not raw_value:
        return os.environ.get("GEMINI_API_KEY", "")
    raw_value = raw_value.strip()
    if raw_value.startswith("${") and raw_value.endswith("}"):
        env_name = raw_value[2:-1]
        return os.environ.get(env_name, "")
    return raw_value


class RateLimiter:
    def __init__(self, rpm: Optional[int]) -> None:
        self.interval = 60.0 / rpm if rpm else 0.0
        self._next_time = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        if self.interval <= 0:
            return
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            if now < self._next_time:
                await asyncio.sleep(self._next_time - now)
                now = self._next_time
        self._next_time = now + self.interval


def _is_quota_error(exc: Exception) -> bool:
    text = str(exc)
    if "RESOURCE_EXHAUSTED" in text:
        return True
    if "429" in text:
        return True
    if "503" in text:
        return True
    return False


@dataclass
class LayerJob:
    psd_name: str
    prompt: str
    composite_path: Path
    layer_record: LayerRecord
    max_attempts: int
    backoff_seconds: float


async def worker_loop(
    worker_name: str,
    queue: asyncio.Queue,
    renamer: GeminiRenamer,
    limiter: RateLimiter,
    results: Dict[int, Dict],
) -> None:
    try:
        while True:
            job = await queue.get()
            if job is None:
                queue.task_done()
                return
            try:
                await limiter.wait()
                ai_output = None
                status = "error"
                attempts_used = 0
                job_requeued = False
                for attempt in range(1, job.max_attempts + 1):
                    attempts_used = attempt
                    try:
                        ai_output = await asyncio.to_thread(
                            renamer.rename_layer,
                            job.prompt,
                            job.composite_path,
                            Path(job.layer_record.layer_png),
                            job.psd_name,
                            job.layer_record.name,
                        )
                        status = "ok"
                        break
                    except Exception as exc:  # pragma: no cover
                        LOGGER.error(
                            "[%s] Gemini error on %s (attempt %d/%d): %s",
                            worker_name,
                            job.layer_record.name,
                            attempt,
                            job.max_attempts,
                            exc,
                        )
                        if attempt < job.max_attempts and job.backoff_seconds > 0:
                            await asyncio.sleep(job.backoff_seconds * attempt)
                        if _is_quota_error(exc):
                            LOGGER.warning(
                                "[%s] Quota error on %s, requeuing job and cooling down %.1fs",
                                worker_name,
                                job.layer_record.name,
                                QUOTA_COOLDOWN,
                            )
                            queue.put_nowait(job)
                            job_requeued = True
                            await asyncio.sleep(QUOTA_COOLDOWN)
                            break
                if job_requeued:
                    continue
                if ai_output is None:
                    ai_output = f"{job.psd_name}：{job.layer_record.name} -> ERROR"
                results[job.layer_record.index] = {
                    "index": job.layer_record.index,
                    "original": job.layer_record.name,
                    "group_path": job.layer_record.group_path,
                    "layer_png": job.layer_record.layer_png,
                    "output": ai_output,
                    "status": status,
                    "attempts": attempts_used,
                }
            finally:
                queue.task_done()
    except asyncio.CancelledError:  # pragma: no cover
        LOGGER.debug("Worker %s cancelled", worker_name)


async def run_layer_jobs(jobs: List[LayerJob], config: Dict) -> Dict[int, Dict]:
    queue: asyncio.Queue = asyncio.Queue()
    for job in jobs:
        queue.put_nowait(job)

    results: Dict[int, Dict] = {}
    gemini_cfg = config["gemini"]
    key_entries = config.get("api_keys") or []
    if not key_entries:
        key_entries = [
            {
                "name": "default",
                "key": None,
                "max_concurrency": 1,
                "rpm": 60,
            }
        ]

    workers = []
    custom_endpoint = os.environ.get("GEMINI_ENDPOINT")
    for key_entry in key_entries:
        api_key = resolve_api_key(key_entry.get("key"))
        if not api_key:
            LOGGER.warning("API key %s missing, skipping", key_entry.get("name", "unnamed"))
            continue
        limiter = RateLimiter(key_entry.get("rpm"))
        concurrency = max(1, int(key_entry.get("max_concurrency", 1)))
        for slot in range(concurrency):
            renamer = GeminiRenamer(
                model=gemini_cfg["model"],
                temperature=gemini_cfg["temperature"],
                top_p=gemini_cfg["top_p"],
                top_k=gemini_cfg["top_k"],
                max_output_tokens=gemini_cfg["max_output_tokens"],
                api_key=api_key,
                base_url=custom_endpoint,
            )
            worker_name = f"{key_entry.get('name', 'key')}-{slot + 1}"
            workers.append(
                asyncio.create_task(worker_loop(worker_name, queue, renamer, limiter, results))
            )

    if not workers:
        raise SystemExit("No valid API key configured. Check config.api_keys or GEMINI_API_KEY.")

    await queue.join()
    for _ in workers:
        queue.put_nowait(None)
    await asyncio.gather(*workers, return_exceptions=True)
    return results


def build_layer_jobs(
    result: ExportResult,
    prompt_template: str,
    retry_cfg: Dict,
) -> List[LayerJob]:
    jobs: List[LayerJob] = []
    max_attempts = max(1, int(retry_cfg.get("max_attempts", 1)))
    backoff_seconds = float(retry_cfg.get("backoff_seconds", 1.0))
    for layer in result.layers:
        metadata_json = json.dumps(
            {
                "bbox": {
                    "left": layer.left,
                    "top": layer.top,
                    "right": layer.right,
                    "bottom": layer.bottom,
                },
                "size": {
                    "width": layer.width,
                    "height": layer.height,
                },
                "visible": layer.visible,
                "opacity": layer.opacity,
            },
            ensure_ascii=False,
        )
        ctx = {
            "psd_name": result.psd,
            "original_name": layer.name,
            "group_path": layer.group_path,
            "layer_metadata": metadata_json,
            "merged_png": result.composite_png,
            "layer_png": layer.layer_png,
            "new_name": "NEW",
        }
        prompt_text = render_prompt(prompt_template, ctx)
        jobs.append(
            LayerJob(
                psd_name=result.psd,
                prompt=prompt_text,
                composite_path=Path(result.composite_png),
                layer_record=layer,
                max_attempts=max_attempts,
                backoff_seconds=backoff_seconds,
            )
        )
    return jobs


async def handle_psd(
    psd_path: Path,
    config: Dict,
    prompt_template: str,
    tmp_dir: Path,
    results_dir: Path,
) -> None:
    processing_cfg = config.get("processing", {})
    include_hidden = bool(processing_cfg.get("include_hidden_layers", False))
    result: ExportResult = await asyncio.to_thread(
        export_psd,
        psd_path,
        tmp_dir,
        include_hidden,
    )
    jobs = build_layer_jobs(result, prompt_template, processing_cfg.get("retry", {}))
    if not jobs:
        LOGGER.warning("No drawable layers detected in %s", psd_path)
        return

    results_map = await run_layer_jobs(jobs, config)
    ordered = [results_map[idx] for idx in sorted(results_map.keys())]
    await asyncio.to_thread(write_results, result, ordered, results_dir)
    await asyncio.to_thread(cleanup_psd_tmp, tmp_dir, Path(result.tmp_dir).name)


async def run_pipeline(psd_files: List[Path], config: Dict, prompt_template: str, tmp_dir: Path, results_dir: Path) -> None:
    max_psd = max(1, int(config.get("processing", {}).get("max_psd_concurrent", 1)))
    sem = asyncio.Semaphore(max_psd)

    async def runner(psd_path: Path) -> None:
        async with sem:
            LOGGER.info("Processing %s", psd_path.name)
            await handle_psd(psd_path, config, prompt_template, tmp_dir, results_dir)

    await asyncio.gather(*(runner(psd) for psd in psd_files))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT_DIR / "config" / "settings.yaml")
    parser.add_argument("--limit", type=int, help="最多处理多少个 PSD")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    config = load_config(args.config)

    paths_cfg = config.get("paths", {})
    psd_dir = (ROOT_DIR / paths_cfg.get("psd_dir", "psd")).resolve()
    tmp_dir = (ROOT_DIR / paths_cfg.get("tmp_dir", "tmp")).resolve()
    results_dir = (ROOT_DIR / paths_cfg.get("results_dir", "output/results")).resolve()
    ensure_dirs(psd_dir, tmp_dir, results_dir)

    prompt_template = load_prompt((ROOT_DIR / config["prompt"]["template_path"]).resolve())

    psd_files = sorted(psd_dir.glob("*.psd"))
    if args.limit:
        psd_files = psd_files[: args.limit]
    LOGGER.info("Found %d PSD files", len(psd_files))
    if not psd_files:
        return

    asyncio.run(run_pipeline(psd_files, config, prompt_template, tmp_dir, results_dir))


if __name__ == "__main__":
    main()
