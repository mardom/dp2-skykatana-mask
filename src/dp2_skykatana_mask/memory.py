"""Container-aware memory diagnostics and guards for 32 GB RSP Nublado pods.

On the Rubin Science Platform (Kubernetes / Nublado), standard host queries like
``os.sysconf`` or ``psutil.virtual_memory().total`` often report the underlying
Kubernetes node's RAM (256-512 GiB) rather than the user pod's 32 GiB cgroup
limit. Crossing the pod cgroup limit causes an immediate SIGKILL (OOMKilled).

This module inspects ``MEM_LIMIT`` (exported by Nublado) and Linux cgroup v1/v2
files first so batch sizes and worker threads stay safely inside 32 GiB RAM.
"""

from __future__ import annotations

import gc
import logging
import os
import resource
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

GIB = float(1024**3)

CGROUP_V2_MAX = Path("/sys/fs/cgroup/memory.max")
CGROUP_V2_CURRENT = Path("/sys/fs/cgroup/memory.current")
CGROUP_V1_LIMIT = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
CGROUP_V1_USAGE = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
PROC_SELF_STATUS = Path("/proc/self/status")

# Ignore kernel "unlimited" sentinel values (~2^63)
_UNLIMITED_THRESHOLD_BYTES = 1 << 50


@dataclass(frozen=True)
class MemorySnapshot:
    """Point-in-time process and container memory statistics in GiB."""

    process_rss_gib: float
    cgroup_current_gib: float | None
    effective_limit_gib: float
    limit_source: str

    @property
    def used_gib(self) -> float:
        """Conservative memory in use (max of process RSS and pod cgroup usage)."""
        if self.cgroup_current_gib is not None:
            return max(self.process_rss_gib, self.cgroup_current_gib)
        return self.process_rss_gib

    @property
    def utilization_fraction(self) -> float:
        if self.effective_limit_gib <= 0:
            return 0.0
        return self.used_gib / self.effective_limit_gib

    def format_short(self) -> str:
        cg = (
            f", pod={self.cgroup_current_gib:.2f} GiB"
            if self.cgroup_current_gib is not None
            else ""
        )
        return (
            f"RSS={self.process_rss_gib:.2f} GiB{cg} / "
            f"limit={self.effective_limit_gib:.1f} GiB ({self.limit_source}, "
            f"{100.0 * self.utilization_fraction:.1f}%)"
        )


def _read_int_file(path: Path) -> int | None:
    try:
        if not path.exists():
            return None
        raw = path.read_text(encoding="utf-8").strip()
        if not raw or raw == "max":
            return None
        val = int(raw)
        if val <= 0 or val >= _UNLIMITED_THRESHOLD_BYTES:
            return None
        return val
    except OSError:
        return None
    except ValueError:
        return None


def detect_effective_memory_limit_gib(configured_limit_gib: float = 32.0) -> tuple[float, str]:
    """Detect the tightest applicable memory ceiling (Nublado env, cgroup, or config).

    Parameters
    ----------
    configured_limit_gib : float
        Target memory ceiling from configuration (default 32.0 GiB for RSP).

    Returns
    -------
    limit_gib : float
        Effective memory limit in GiB.
    source : str
        Human-readable source of the active limit.
    """
    candidates: list[tuple[float, str]] = []

    if configured_limit_gib > 0:
        candidates.append((float(configured_limit_gib), "config"))

    # Nublado sets MEM_LIMIT in bytes (or occasionally float/int string)
    mem_env = os.environ.get("MEM_LIMIT", "").strip()
    if mem_env:
        try:
            env_bytes = int(float(mem_env))
            if 0 < env_bytes < _UNLIMITED_THRESHOLD_BYTES:
                candidates.append((env_bytes / GIB, "RSP MEM_LIMIT"))
        except ValueError:
            pass

    cg2 = _read_int_file(CGROUP_V2_MAX)
    if cg2 is not None:
        candidates.append((cg2 / GIB, "cgroup-v2"))

    cg1 = _read_int_file(CGROUP_V1_LIMIT)
    if cg1 is not None:
        candidates.append((cg1 / GIB, "cgroup-v1"))

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            host_bytes = pages * page_size
            if 0 < host_bytes < _UNLIMITED_THRESHOLD_BYTES:
                candidates.append((host_bytes / GIB, "host-ram"))
    except (ValueError, OSError, AttributeError):
        pass

    if not candidates:
        return 32.0, "default-32GiB"

    return min(candidates, key=lambda item: item[0])


def read_process_rss_gib() -> float:
    """Return current resident set size (VmRSS) of this Python process in GiB."""
    try:
        if PROC_SELF_STATUS.exists():
            for line in PROC_SELF_STATUS.read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        kb = int(parts[1])
                        return (kb * 1024.0) / GIB
    except (OSError, ValueError):
        pass

    # Fallback: maxrss (in KiB on Linux)
    ru_maxrss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (ru_maxrss_kb * 1024.0) / GIB


def read_cgroup_current_gib() -> float | None:
    """Return total memory currently charged to this container cgroup in GiB."""
    cg2 = _read_int_file(CGROUP_V2_CURRENT)
    if cg2 is not None:
        return cg2 / GIB
    cg1 = _read_int_file(CGROUP_V1_USAGE)
    if cg1 is not None:
        return cg1 / GIB
    return None


class MemoryGuard:
    """Tracks live memory against a 32 GiB RSP container ceiling and enforces thread caps."""

    def __init__(
        self,
        max_ram_gib: float = 32.0,
        soft_warning_fraction: float = 0.75,
        hard_abort_fraction: float = 0.90,
    ) -> None:
        self.limit_gib, self.limit_source = detect_effective_memory_limit_gib(max_ram_gib)
        self.soft_warning_fraction = float(soft_warning_fraction)
        self.hard_abort_fraction = float(hard_abort_fraction)
        self.peak_used_gib: float = 0.0

    def snapshot(self) -> MemorySnapshot:
        snap = MemorySnapshot(
            process_rss_gib=read_process_rss_gib(),
            cgroup_current_gib=read_cgroup_current_gib(),
            effective_limit_gib=self.limit_gib,
            limit_source=self.limit_source,
        )
        if snap.used_gib > self.peak_used_gib:
            self.peak_used_gib = snap.used_gib
        return snap

    def check(self, label: str, *, collect_garbage: bool = False) -> MemorySnapshot:
        """Log memory status and raise MemoryError cleanly before Kubernetes SIGKILLs the pod."""
        if collect_garbage:
            gc.collect()
        snap = self.snapshot()
        frac = snap.utilization_fraction
        if frac >= self.hard_abort_fraction:
            gc.collect()
            snap = self.snapshot()
            frac = snap.utilization_fraction
            if frac >= self.hard_abort_fraction:
                raise MemoryError(
                    f"[{label}] Memory usage reached {snap.format_short()}, exceeding the "
                    f"{100.0 * self.hard_abort_fraction:.0f}% safety threshold before RSP pod OOMKill. "
                    "Lower `cov_batch_size`, `circle_chunk_size`, or `order_sparse` in your YAML config."
                )
        if frac >= self.soft_warning_fraction:
            logger.warning("[%s] High memory watermark: %s", label, snap.format_short())
        else:
            logger.info("[%s] %s", label, snap.format_short())
        return snap


def configure_rsp_thread_env(max_threads: int = 2) -> None:
    """Prevent BLAS/OpenMP/NumExpr from spawning 64+ threads on shared RSP nodes."""
    t_str = str(max(1, int(max_threads)))
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, t_str)


def estimate_working_set_gib(
    area_deg2: float,
    order_prop: int,
    order_star: int,
    order_out: int,
    order_cov: int = 5,
    cov_batch_size: int = 64,
    circle_chunk_size: int = 100_000,
    n_threads: int = 2,
) -> dict[str, float]:
    """Analytic upper-bound memory estimate (in GiB) for each pipeline stage."""
    sphere_deg2 = 4.0 * 3.141592653589793 * (180.0 / 3.141592653589793) ** 2
    nside_cov = 1 << order_cov
    cov_pix_area = sphere_deg2 / (12 * nside_cov * nside_cov)
    n_cov_touched = max(1.0, area_deg2 / cov_pix_area * 1.25)

    def _bitpacked_stage_gib(order_s: int) -> float:
        nfine = 1 << (2 * (order_s - order_cov))
        bytes_per_cov = max(8.0, nfine / 8.0)
        return (n_cov_touched * bytes_per_cov) / GIB

    def _unpacked_bool_gib(order_s: int) -> float:
        nfine = 1 << (2 * (order_s - order_cov))
        return (n_cov_touched * float(nfine)) / GIB

    # Stage 1: one float64 map batch + valid_pixels + pass_count/req_ok buffers
    nfine_prop = 1 << (2 * (order_prop - order_cov))
    prop_batch_gib = (cov_batch_size * nfine_prop * (8.0 + 8.0 + 3.0)) / GIB
    prop_stage_gib = _bitpacked_stage_gib(order_prop)

    # Stage 2: SkyMaskPipe.build_circ_mask allocates unpacked bool map before bit-packing
    star_unpacked_gib = _unpacked_bool_gib(order_star)
    star_chunk_buffers_gib = (circle_chunk_size * n_threads * 128.0) / GIB
    star_stage_gib = _bitpacked_stage_gib(order_star)

    # Stage 3: SkyMaskPipe.combine row-by-row
    combine_gib = prop_stage_gib + star_stage_gib + _bitpacked_stage_gib(order_out) + 0.25

    python_butler_overhead_gib = 1.2
    peak_gib = python_butler_overhead_gib + max(
        prop_batch_gib + prop_stage_gib,
        prop_stage_gib + star_unpacked_gib + star_chunk_buffers_gib,
        combine_gib,
    )

    return {
        "python_and_butler_base_gib": round(python_butler_overhead_gib, 3),
        "stage1_prop_working_gib": round(prop_batch_gib + prop_stage_gib, 3),
        "stage1_propmask_bitpacked_gib": round(prop_stage_gib, 3),
        "stage2_star_working_gib": round(star_unpacked_gib + star_chunk_buffers_gib, 3),
        "stage2_starmask_bitpacked_gib": round(star_stage_gib, 3),
        "stage3_combine_working_gib": round(combine_gib, 3),
        "estimated_peak_ram_gib": round(peak_gib, 3),
    }
