"""Stage 1: Memory-bounded multi-band DP2 survey property map masking (`sp_maps`).

Processes ``HealSparseMap`` survey property maps in small batches of coverage pixels
so peak RAM stays < 1.5 GiB on 32 GiB RSP Nublado containers even across the full
DP2 footprint at ``order_sparse=14`` (12.9") or ``order_sparse=15`` (6.4"), while
supporting:
  - Per-band minimal number of exposures (from ``exposure_time_consolidated_map_sum``),
  - Per-band minimal 5-sigma PSF limiting magnitudes (``psf_maglim``),
  - Per-band maximal PSF size / seeing, sky noise, sky background, or custom maps,
  - Pixel-by-pixel ``min_filters`` (at least K out of N bands passing) and ``required_bands``,
  - Seamless registration as bit-packed ``SkyMaskPipe`` stages.
"""

from __future__ import annotations

import gc
import logging
import math
from dataclasses import dataclass
from typing import Any, Sequence

import healsparse as hsp
import hpgeom as hpg
import numpy as np
from skykatana import SkyMaskPipe

from .butler_io import get_sp_map_chunk, get_sp_map_coverage, list_consolidated_maps
from .config import BandCriteria, MaskPipelineConfig, RegionSelector
from .memory import MemoryGuard

logger = logging.getLogger(__name__)

_COMPARISON_OPS = {
    "gt": lambda v, t: v > t,
    "ge": lambda v, t: v >= t,
    "lt": lambda v, t: v < t,
    "le": lambda v, t: v <= t,
}


@dataclass
class PropertyMaskStats:
    """Summary statistics from building the DP2 property-map footprint stage."""

    order_sparse: int
    order_cov: int
    nside_sparse: int
    nside_cov: int
    bands_evaluated: list[str]
    min_filters: int
    required_bands: list[str]
    n_cov_pixels_scanned: int
    n_valid_pixels: int
    area_deg2: float
    per_band_area_deg2: dict[str, float]


def _angular_separation_deg(
    ra1_deg: np.ndarray | float,
    dec1_deg: np.ndarray | float,
    ra2_deg: float,
    dec2_deg: float,
) -> np.ndarray:
    ra1 = np.deg2rad(ra1_deg)
    dec1 = np.deg2rad(dec1_deg)
    ra2 = math.radians(ra2_deg)
    dec2 = math.radians(dec2_deg)
    d_ra = ra1 - ra2
    d_dec = dec1 - dec2
    a = np.sin(0.5 * d_dec) ** 2 + np.cos(dec1) * math.cos(dec2) * np.sin(0.5 * d_ra) ** 2
    return np.rad2deg(2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0)))


def _filter_cov_pixels_by_region(
    cov_pixels: np.ndarray,
    nside_cov: int,
    region: RegionSelector,
) -> np.ndarray:
    """Pre-filter coarse coverage pixels intersecting ``region`` (with safety margin)."""
    if region.mode == "full" or cov_pixels.size == 0:
        return cov_pixels

    cov_ra, cov_dec = hpg.pixel_to_angle(nside_cov, cov_pixels, nest=True, degrees=True)
    pix_margin_deg = float(hpg.max_pixel_radius(nside_cov, degrees=True)) * 1.15

    if region.mode == "cone":
        sep = _angular_separation_deg(cov_ra, cov_dec, region.ra_deg, region.dec_deg)
        return cov_pixels[sep <= (region.radius_deg + pix_margin_deg)]

    # mode == "box"
    dec_ok = (cov_dec >= region.dec_min_deg - pix_margin_deg) & (
        cov_dec <= region.dec_max_deg + pix_margin_deg
    )
    ra_margin = pix_margin_deg / max(0.05, math.cos(math.radians(0.5 * (region.dec_min_deg + region.dec_max_deg))))
    ra_lo = (region.ra_min_deg - ra_margin) % 360.0
    ra_hi = (region.ra_max_deg + ra_margin) % 360.0
    if region.ra_max_deg - region.ra_min_deg >= 360.0 - 2.0 * ra_margin:
        ra_ok = np.ones_like(dec_ok)
    elif ra_lo <= ra_hi:
        ra_ok = (cov_ra >= ra_lo) & (cov_ra <= ra_hi)
    else:
        ra_ok = (cov_ra >= ra_lo) | (cov_ra <= ra_hi)
    return cov_pixels[dec_ok & ra_ok]


def _filter_fine_pixels_by_region(
    fine_pixels: np.ndarray,
    nside_sparse: int,
    region: RegionSelector,
) -> np.ndarray:
    """Apply exact fine-pixel boundary cut when ``region.mode`` is ``'cone'`` or ``'box'``."""
    if region.mode == "full" or fine_pixels.size == 0:
        return fine_pixels

    ra, dec = hpg.pixel_to_angle(nside_sparse, fine_pixels, nest=True, degrees=True)
    if region.mode == "cone":
        sep = _angular_separation_deg(ra, dec, region.ra_deg, region.dec_deg)
        return fine_pixels[sep <= region.radius_deg]

    dec_ok = (dec >= region.dec_min_deg) & (dec <= region.dec_max_deg)
    if region.ra_min_deg <= region.ra_max_deg:
        ra_ok = (ra >= region.ra_min_deg) & (ra <= region.ra_max_deg)
    else:
        ra_ok = (ra >= region.ra_min_deg) | (ra <= region.ra_max_deg)
    return fine_pixels[dec_ok & ra_ok]


def _conditions_for_band(
    cfg: MaskPipelineConfig,
    band: str,
) -> list[tuple[str, str, float, bool]]:
    """Build ordered list of ``(dataset_type, op, threshold, require_positive)`` for ``band``."""
    pm_cfg = cfg.property_maps
    crit = pm_cfg.per_band.get(band, BandCriteria())
    conds: list[tuple[str, str, float, bool]] = []

    min_exptime = crit.effective_min_exptime_sec()
    if min_exptime is not None:
        conds.append((pm_cfg.exptime_dataset, "ge", float(min_exptime), True))

    if crit.min_maglim is not None:
        conds.append((pm_cfg.maglim_dataset, "ge", float(crit.min_maglim), True))
    if crit.max_maglim is not None:
        conds.append((pm_cfg.maglim_dataset, "le", float(crit.max_maglim), True))

    max_psf_pix = crit.effective_max_psf_pixels()
    if max_psf_pix is not None:
        conds.append((pm_cfg.psf_size_dataset, "le", float(max_psf_pix), True))

    if crit.max_sky_noise_njy is not None:
        conds.append((pm_cfg.sky_noise_dataset, "le", float(crit.max_sky_noise_njy), True))

    if crit.max_sky_background_njy is not None:
        conds.append((pm_cfg.sky_bg_dataset, "le", float(crit.max_sky_background_njy), False))

    for extra in crit.extra_conditions:
        conds.append((extra.dataset_type, extra.comparison, float(extra.threshold), False))

    # If a band is listed in `bands` with no explicit thresholds, any valid (>0) depth pixel counts
    if not conds:
        conds.append((pm_cfg.maglim_dataset, "gt", 0.0, True))

    return conds


def attach_stage_to_pipe(
    pipe: SkyMaskPipe,
    mask: hsp.HealSparseMap,
    stage_name: str,
    extra_meta: dict[str, Any] | None = None,
) -> hsp.HealSparseMap:
    """Attach a ``HealSparseMap`` onto ``SkyMaskPipe`` using its native stage registry."""
    if hasattr(pipe, "_finalize_stage"):
        return pipe._finalize_stage(
            mask,
            default_name=stage_name,
            output_stage=stage_name,
            extra_meta=extra_meta or {},
        )
    setattr(pipe, stage_name, mask)
    if hasattr(pipe, "_params") and extra_meta:
        pipe._params[stage_name] = dict(extra_meta)
    return mask


def build_dp2_property_mask(
    pipe: SkyMaskPipe,
    butler: Any,
    cfg: MaskPipelineConfig,
    guard: MemoryGuard,
    output_stage: str = "propmask",
) -> tuple[hsp.HealSparseMap, PropertyMaskStats]:
    """Stream DP2 consolidated property maps from Butler into a bit-packed ``SkyMaskPipe`` stage."""
    pm_cfg = cfg.property_maps
    bands = list(pm_cfg.bands)
    required_set = set(pm_cfg.required_bands)
    order_sparse = int(pm_cfg.order_sparse)
    nside_sparse = 1 << order_sparse

    band_conditions = {b: _conditions_for_band(cfg, b) for b in bands}
    needed_datasets = {ds for conds in band_conditions.values() for (ds, _, _, _) in conds}

    available_maps = set(list_consolidated_maps(butler))
    if available_maps:
        missing = sorted(needed_datasets - available_maps)
        if missing:
            raise RuntimeError(
                f"Requested DP2 property map dataset(s) {missing} not found in Butler. "
                f"Available consolidated maps: {sorted(available_maps)}"
            )

    # 1. Inspect lightweight `.coverage` masks (~12 KB each) to find active sky tiles
    band_cov_sets: dict[str, np.ndarray] = {}
    native_nside_cov: int | None = None

    for band in bands:
        anchor_ds = band_conditions[band][0][0]
        cov_obj = get_sp_map_coverage(butler, anchor_ds, band=band, skymap=pm_cfg.skymap)
        if hasattr(cov_obj, "coverage_mask"):
            if native_nside_cov is None:
                native_nside_cov = int(cov_obj.nside_coverage)
            raw_cov = np.flatnonzero(cov_obj.coverage_mask).astype(np.int64)
        else:
            cov_arr = np.asarray(cov_obj)
            if native_nside_cov is None:
                native_nside_cov = int(round(math.sqrt(cov_arr.size / 12.0)))
            raw_cov = np.flatnonzero(cov_arr > 0).astype(np.int64)
        band_cov_sets[band] = _filter_cov_pixels_by_region(raw_cov, native_nside_cov, cfg.region)
        logger.info(
            "Band %s coverage (%s): %d coarse pixels (nside_cov=%d)",
            band,
            anchor_ds,
            len(band_cov_sets[band]),
            native_nside_cov,
        )

    assert native_nside_cov is not None
    native_order_cov = int(round(math.log2(native_nside_cov)))
    if order_sparse <= native_order_cov:
        raise ValueError(
            f"order_sparse ({order_sparse}) must exceed native coverage order ({native_order_cov})"
        )

    # Candidate coverage pixels: union across bands, intersected with mandatory required_bands
    all_cov_arrays = [arr for arr in band_cov_sets.values() if arr.size > 0]
    if not all_cov_arrays:
        raise RuntimeError("Zero DP2 coverage pixels found for the requested bands and region.")

    candidate_cov = np.unique(np.concatenate(all_cov_arrays))
    for req_band in pm_cfg.required_bands:
        candidate_cov = np.intersect1d(candidate_cov, band_cov_sets[req_band], assume_unique=True)

    # Count how many bands even cover each coarse tile: a tile covered by < min_filters bands can never pass!
    n_total_cov = 12 * native_nside_cov * native_nside_cov
    band_has_cov: dict[str, np.ndarray] = {}
    cov_band_multiplicity = np.zeros(n_total_cov, dtype=np.uint8)
    for band in bands:
        mask_arr = np.zeros(n_total_cov, dtype=bool)
        mask_arr[band_cov_sets[band]] = True
        band_has_cov[band] = mask_arr
        cov_band_multiplicity += mask_arr.astype(np.uint8)

    candidate_cov = candidate_cov[cov_band_multiplicity[candidate_cov] >= pm_cfg.min_filters]
    if candidate_cov.size == 0:
        raise RuntimeError(
            f"Zero coverage tiles have >= {pm_cfg.min_filters} bands (with required_bands={pm_cfg.required_bands})."
        )

    logger.info(
        "Evaluating %d candidate coverage tiles in batches of %d (order_sparse=%d, nside=%d)",
        candidate_cov.size,
        cfg.memory.cov_batch_size,
        order_sparse,
        nside_sparse,
    )

    # 2. Prepare bit-packed output map(s)
    propmask = hsp.HealSparseMap.make_empty(
        native_nside_cov,
        nside_sparse,
        dtype=np.bool_,
        bit_packed=True,
    )
    per_band_masks: dict[str, hsp.HealSparseMap] = {}
    if pm_cfg.keep_per_band_stages:
        for band in bands:
            per_band_masks[band] = hsp.HealSparseMap.make_empty(
                native_nside_cov,
                nside_sparse,
                dtype=np.bool_,
                bit_packed=True,
            )

    per_band_pixel_counts: dict[str, int] = {b: 0 for b in bands}
    shift = 2 * (order_sparse - native_order_cov)
    n_fine = 1 << shift
    fine_mask_bits = n_fine - 1

    cov_to_slot = np.full(n_total_cov, -1, dtype=np.int32)
    batch_size = max(1, int(cfg.memory.cov_batch_size))
    n_batches = int(math.ceil(candidate_cov.size / batch_size))

    # 3. Stream coverage batches strictly within 32 GiB RAM
    for batch_idx in range(n_batches):
        cov_batch = candidate_cov[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        m_tiles = int(cov_batch.size)
        cov_to_slot[cov_batch] = np.arange(m_tiles, dtype=np.int32)

        pass_count = np.zeros(m_tiles * n_fine, dtype=np.uint8)
        req_ok = np.ones(m_tiles * n_fine, dtype=bool)

        try:
            for band in bands:
                b_cov = cov_batch[band_has_cov[band][cov_batch]]
                if b_cov.size == 0:
                    if band in required_set:
                        req_ok.fill(False)
                        break
                    continue

                conds = band_conditions[band]
                band_pass: np.ndarray | None = None

                for cond_idx, (ds_name, op_name, thresh, require_pos) in enumerate(conds):
                    op_fn = _COMPARISON_OPS[op_name]
                    hsp_chunk = get_sp_map_chunk(
                        butler,
                        ds_name,
                        band=band,
                        skymap=pm_cfg.skymap,
                        cov_pixels=b_cov,
                        nside_sparse=nside_sparse,
                    )
                    vpix = np.asarray(hsp_chunk.valid_pixels, dtype=np.int64)
                    if vpix.size == 0:
                        del hsp_chunk
                        band_pass = None
                        break

                    vals = np.asarray(hsp_chunk.get_values_pix(vpix), dtype=np.float64)
                    del hsp_chunk

                    valid_val = np.isfinite(vals)
                    if require_pos:
                        valid_val &= vals > 0.0
                    keep = valid_val & op_fn(vals, thresh)
                    sel_pix = vpix[keep]
                    del vpix, vals, valid_val, keep

                    if sel_pix.size == 0:
                        band_pass = None
                        break

                    cov_ids = sel_pix >> shift
                    slots = cov_to_slot[cov_ids]
                    in_batch = slots >= 0
                    if not np.all(in_batch):
                        sel_pix = sel_pix[in_batch]
                        slots = slots[in_batch]

                    flat_idx = slots.astype(np.int64) * n_fine + (sel_pix & fine_mask_bits)
                    del sel_pix, cov_ids, slots

                    if cond_idx == 0:
                        band_pass = np.zeros(m_tiles * n_fine, dtype=bool)
                        band_pass[flat_idx] = True
                    else:
                        assert band_pass is not None
                        next_pass = np.zeros(m_tiles * n_fine, dtype=bool)
                        next_pass[flat_idx] = band_pass[flat_idx]
                        band_pass = next_pass
                    del flat_idx

                    if not np.any(band_pass):
                        band_pass = None
                        break

                if band_pass is not None:
                    b_flat = np.flatnonzero(band_pass)
                    if b_flat.size > 0:
                        pass_count[b_flat] += 1
                        if cfg.region.mode == "full":
                            per_band_pixel_counts[band] += int(b_flat.size)
                        if pm_cfg.keep_per_band_stages or cfg.region.mode != "full":
                            b_slots = b_flat // n_fine
                            b_offs = b_flat & fine_mask_bits
                            b_gpix = (cov_batch[b_slots] << shift) | b_offs
                            b_gpix = _filter_fine_pixels_by_region(b_gpix, nside_sparse, cfg.region)
                            if cfg.region.mode != "full":
                                per_band_pixel_counts[band] += int(b_gpix.size)
                            if pm_cfg.keep_per_band_stages and b_gpix.size > 0:
                                per_band_masks[band].update_values_pix(b_gpix, True)
                    if band in required_set:
                        req_ok &= band_pass
                    del band_pass
                elif band in required_set:
                    req_ok.fill(False)
                    break

            surviving_flat = np.flatnonzero(req_ok & (pass_count >= pm_cfg.min_filters))
            if surviving_flat.size > 0:
                slots = surviving_flat // n_fine
                offsets = surviving_flat & fine_mask_bits
                global_pix = (cov_batch[slots] << shift) | offsets
                global_pix = _filter_fine_pixels_by_region(global_pix, nside_sparse, cfg.region)
                if global_pix.size > 0:
                    propmask.update_values_pix(global_pix, True)
                del global_pix
        finally:
            cov_to_slot[cov_batch] = -1
            del pass_count, req_ok

        if (batch_idx + 1) % max(1, n_batches // 5) == 0 or (batch_idx + 1) == n_batches:
            guard.check(f"propmask batch {batch_idx + 1}/{n_batches}", collect_garbage=True)

    target_order_cov = int(pm_cfg.order_cov)
    if target_order_cov != native_order_cov:
        logger.info("Aligning propmask order_cov: %d -> %d", native_order_cov, target_order_cov)
        propmask = pipe.change_cov_order(propmask, order=target_order_cov, inplace=False, verbose=False)
        for b in list(per_band_masks):
            per_band_masks[b] = pipe.change_cov_order(
                per_band_masks[b], order=target_order_cov, inplace=False, verbose=False
            )

    pix_area_deg2 = 4.0 * math.pi * (180.0 / math.pi) ** 2 / (12.0 * nside_sparse * nside_sparse)
    per_band_area = {b: round(cnt * pix_area_deg2, 4) for b, cnt in per_band_pixel_counts.items()}

    if pm_cfg.keep_per_band_stages:
        for band, b_mask in per_band_masks.items():
            attach_stage_to_pipe(
                pipe,
                b_mask,
                stage_name=f"prop_{band}_mask",
                extra_meta={
                    "band": band,
                    "pixels": int(b_mask.n_valid),
                    "area_deg2": float(b_mask.get_valid_area(degrees=True)),
                },
            )

    area_deg2 = float(propmask.get_valid_area(degrees=True))
    n_valid = int(propmask.n_valid)
    if n_valid == 0:
        raise RuntimeError(
            "Property-map mask contains 0 valid pixels after applying exposure, filter, and depth cuts. "
            "Relax `min_exposures`, `min_filters`, or `min_maglim` in your configuration."
        )

    extra_meta = {
        "skymap": pm_cfg.skymap,
        "bands": bands,
        "min_filters": pm_cfg.min_filters,
        "required_bands": list(pm_cfg.required_bands),
        "pixels": n_valid,
        "area_deg2": area_deg2,
        "per_band_area_deg2": per_band_area,
        "bit_packed": True,
    }
    attach_stage_to_pipe(pipe, propmask, stage_name=output_stage, extra_meta=extra_meta)
    gc.collect()
    guard.check("propmask finalized", collect_garbage=False)

    stats = PropertyMaskStats(
        order_sparse=order_sparse,
        order_cov=target_order_cov,
        nside_sparse=nside_sparse,
        nside_cov=1 << target_order_cov,
        bands_evaluated=bands,
        min_filters=pm_cfg.min_filters,
        required_bands=list(pm_cfg.required_bands),
        n_cov_pixels_scanned=int(candidate_cov.size),
        n_valid_pixels=n_valid,
        area_deg2=area_deg2,
        per_band_area_deg2=per_band_area,
    )
    return propmask, stats


def build_prop_mask_from_loaded_maps(
    pipe: SkyMaskPipe,
    prop_maps: Sequence[hsp.HealSparseMap | str],
    thresholds: Sequence[float],
    comparisons: Sequence[str],
    order_sparse: int = 14,
    order_cov: int = 5,
    output_stage: str = "propmask",
) -> hsp.HealSparseMap:
    """Direct wrapper around ``SkyMaskPipe.build_prop_mask`` for pre-loaded local cutouts."""
    return pipe.build_prop_mask(
        prop_maps=list(prop_maps),
        thresholds=list(thresholds),
        comparisons=list(comparisons),
        order_sparse=order_sparse,
        order_cov=order_cov,
        bit_packed=True,
        output_stage=output_stage,
    )
