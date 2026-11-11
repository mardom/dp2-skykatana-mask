"""Stage 2: Bright-star halo & bleed-trail masking from DP2 `the_monster` catalog.

Follows Tutorial 204.2 (`https://dp2.lsst.io/tutorials/notebook/204/notebook-204-2.html`):
  1. Dynamically resolves the latest ``the_monster_YYYYMMDD`` dataset in Butler.
  2. Identifies only the ``htm7`` trixels overlapping the valid survey footprint plus a
     safety buffer equal to ``max_radius_arcsec`` (via ``lsst.sphgeom.HtmPixelization(7)``,
     with a pure-Python fallback for local offline tests).
  3. Streams ``htm7`` shards one at a time, converts ``coord_ra``/``coord_dec`` from
     radians to degrees, converts synthetic ``monster_*_flux`` (``nJy``) to AB magnitudes,
     applies magnitude & buffered-footprint cuts immediately, and flushes ``(ra, dec, radius)``
     rows to a compact Parquet file so raw ``SimpleCatalog`` shards never accumulate in RAM.
  4. Pixelizes circular halos (and optional saturated bleed boxes) into bit-packed
     ``SkyMaskPipe`` stages (`starmask`, `boxmask`) with 32 GiB-safe chunking.
"""

from __future__ import annotations

import gc
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import astropy.units as u
from astropy.table import Table
import healsparse as hsp
import hpgeom as hpg
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from skykatana import SkyMaskPipe

from .butler_io import (
    discover_monster_dataset,
    query_monster_refs_for_htm7,
    resolve_monster_flux_columns,
)
from .config import MaskPipelineConfig, MonsterStarConfig
from .memory import MemoryGuard

logger = logging.getLogger(__name__)

# Exact AB zero-point for nanojansky fluxes: -2.5 * log10(1e-9 / 3631) = 31.40006562228223
AB_MAG_ZEROPOINT_NJY = float((1.0 * u.nJy).to_value(u.ABmag))


@dataclass
class MonsterStarMaskStats:
    """Summary statistics from building the Monster bright-star mask."""

    dataset_type: str
    flux_columns_used: list[str]
    n_htm7_trixels_queried: int
    n_htm7_shards_loaded: int
    n_stars_selected: int
    n_bleed_boxes: int
    starmask_valid_pixels: int
    starmask_area_deg2: float
    boxmask_area_deg2: float
    circles_parquet_path: str | None


def flux_njy_to_abmag(flux_njy: np.ndarray) -> np.ndarray:
    """Convert nanojansky flux array to AB magnitudes (`NaN` where flux <= 0 or non-finite)."""
    arr = np.asarray(flux_njy, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    pos = np.isfinite(arr) & (arr > 0.0)
    if np.any(pos):
        out[pos] = -2.5 * np.log10(arr[pos]) + AB_MAG_ZEROPOINT_NJY
    return out


def compute_star_halo_radius_arcsec(mags: np.ndarray, cfg: MonsterStarConfig) -> np.ndarray:
    """Compute magnitude-dependent circular exclusion radius (in arcseconds)."""
    m = np.asarray(mags, dtype=np.float64)

    if cfg.radius_model == "csv_fit":
        assert cfg.fit_coefficients_csv is not None
        coef_df = pd.read_csv(cfg.fit_coefficients_csv)
        if "profile_behavior" in coef_df.columns:
            sub = coef_df[coef_df["profile_behavior"] == cfg.fit_behavior_branch]
            row = sub.iloc[0] if len(sub) > 0 else coef_df.iloc[0]
        else:
            row = coef_df.iloc[0]
        a_arcsec = float(row["A_arcsec"])
        b_slope = float(row["B_per_mag"])
        rad = a_arcsec * np.exp(b_slope * m)
    elif cfg.radius_model == "exponential":
        rad = float(cfg.A_arcsec) * np.exp(float(cfg.B_per_mag) * m)
        if cfg.mag_knee is not None:
            rad = np.where(m > float(cfg.mag_knee), float(cfg.floor_arcsec), rad)
    elif cfg.radius_model == "powerlaw10":
        rad = float(cfg.r0_arcsec) * (10.0 ** (float(cfg.slope_dex_per_mag) * (float(cfg.m0) - m)))
    else:
        raise ValueError(f"Unsupported radius_model: {cfg.radius_model!r}")

    return np.clip(rad, float(cfg.min_radius_arcsec), float(cfg.max_radius_arcsec))


def _coarse_footprint_centers_and_mask(
    footprint_mask: hsp.HealSparseMap,
    buffer_deg: float,
    coarse_order: int = 6,  # NSIDE=64 (~0.92 deg pixels)
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Extract unique coarse HEALPix centers and a dilated boolean lookup array."""
    nside_sparse = int(footprint_mask.nside_sparse)
    order_sparse = int(round(math.log2(nside_sparse)))
    eff_coarse_order = min(coarse_order, order_sparse)
    coarse_nside = 1 << eff_coarse_order
    shift = 2 * (order_sparse - eff_coarse_order)

    coarse_chunks: list[np.ndarray] = []
    for pix_chunk in footprint_mask.iter_valid_pixels_by_covpix():
        arr = np.asarray(pix_chunk, dtype=np.int64)
        if arr.size > 0:
            coarse_chunks.append(np.unique(arr >> shift))

    if not coarse_chunks:
        raise RuntimeError("Cannot query Monster stars over an empty footprint mask.")

    coarse_pix = np.unique(np.concatenate(coarse_chunks))
    ra_c, dec_c = hpg.pixel_to_angle(coarse_nside, coarse_pix, nest=True, degrees=True)
    coarse_radius_deg = float(hpg.max_pixel_radius(coarse_nside, degrees=True))
    search_radius_deg = coarse_radius_deg + float(buffer_deg)

    # Build a dilated nside=128 boolean mask so stars inside buffered footprint are kept in O(1)
    filter_nside = 128
    dilated = np.zeros(12 * filter_nside * filter_nside, dtype=bool)
    disc_rad = coarse_radius_deg + float(buffer_deg) + float(
        hpg.max_pixel_radius(filter_nside, degrees=True)
    )
    for r0, d0 in zip(ra_c, dec_c):
        qpix = hpg.query_circle(
            filter_nside, float(r0), float(d0), disc_rad, nest=True, degrees=True, inclusive=True
        )
        dilated[qpix] = True

    return ra_c, dec_c, search_radius_deg, dilated


def compute_overlapping_htm7_ids(
    ra_centers_deg: np.ndarray,
    dec_centers_deg: np.ndarray,
    search_radius_deg: float,
) -> list[int] | None:
    """Compute exact superset of ``htm7`` trixel IDs via ``lsst.sphgeom.HtmPixelization(7)``.

    Returns ``None`` if ``lsst.sphgeom`` is not installed (e.g. local non-RSP unit tests),
    allowing ``query_monster_refs_for_htm7`` to query available shards directly.
    """
    try:
        from lsst.sphgeom import Angle, Circle, HtmPixelization, LonLat, RangeSet, UnitVector3d
    except ImportError:
        logger.info("lsst.sphgeom not installed; delegating Monster shard selection to Butler")
        return None

    htm = HtmPixelization(7)
    combined = RangeSet()
    opening_angle = Angle.fromDegrees(float(search_radius_deg))
    for ra_val, dec_val in zip(ra_centers_deg, dec_centers_deg):
        uv = UnitVector3d(LonLat.fromDegrees(float(ra_val), float(dec_val)))
        combined |= htm.envelope(Circle(uv, opening_angle))

    htm7_ids: list[int] = []
    for begin, end in combined.ranges():
        htm7_ids.extend(range(int(begin), int(end)))
    return sorted(set(htm7_ids))


def _extract_columns_from_shard(
    shard: Any,
    flux_cols_cache: list[str] | None,
    cfg: MonsterStarConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Extract ``(ra_deg, dec_deg, mag_ab, flux_cols)`` from a ``SimpleCatalog`` or Table."""
    # Determine available column names without copying whole catalog when possible
    if hasattr(shard, "schema") and hasattr(shard.schema, "getNames"):
        col_names = list(shard.schema.getNames())
        is_afw = True
    elif hasattr(shard, "colnames"):
        col_names = list(shard.colnames)
        is_afw = False
    elif hasattr(shard, "columns"):
        col_names = list(shard.columns)
        is_afw = False
    else:
        tbl = shard.asAstropy()
        col_names = list(tbl.colnames)
        shard = tbl
        is_afw = False

    flux_cols = flux_cols_cache or resolve_monster_flux_columns(
        col_names,
        cfg.reference_bands,
        cfg.explicit_flux_column,
    )

    ra_col = "coord_ra" if "coord_ra" in col_names else "ra"
    dec_col = "coord_dec" if "coord_dec" in col_names else "dec"

    if is_afw:
        # LSST afw SimpleCatalog stores coord_ra / coord_dec in radians and fluxes in nJy
        ra_deg = np.rad2deg(np.asarray(shard[ra_col], dtype=np.float64)) % 360.0
        dec_deg = np.rad2deg(np.asarray(shard[dec_col], dtype=np.float64))
    else:
        ra_raw = shard[ra_col]
        dec_raw = shard[dec_col]
        ra_unit = getattr(ra_raw, "unit", None)
        dec_unit = getattr(dec_raw, "unit", None)
        if ra_unit is not None:
            ra_deg = np.asarray(u.Quantity(ra_raw).to_value(u.deg), dtype=np.float64) % 360.0
        else:
            arr_ra = np.asarray(ra_raw, dtype=np.float64)
            # Detect radians if all coordinates lie in [0, 2*pi] while Monster spans full RA
            if arr_ra.size > 0 and np.nanmax(np.abs(arr_ra)) <= (2.0 * math.pi + 1e-6) and getattr(
                shard, "meta", {}
            ).get("COORD_UNIT") == "rad":
                ra_deg = np.rad2deg(arr_ra) % 360.0
            else:
                ra_deg = arr_ra % 360.0

        if dec_unit is not None:
            dec_deg = np.asarray(u.Quantity(dec_raw).to_value(u.deg), dtype=np.float64)
        else:
            arr_dec = np.asarray(dec_raw, dtype=np.float64)
            if getattr(shard, "meta", {}).get("COORD_UNIT") == "rad":
                dec_deg = np.rad2deg(arr_dec)
            else:
                dec_deg = arr_dec

    # Compute effective (brightest) AB magnitude across requested Monster reference bands
    best_mag = np.full(ra_deg.shape, np.inf, dtype=np.float64)
    for fc in flux_cols:
        f_raw = shard[fc]
        f_unit = getattr(f_raw, "unit", None)
        if f_unit is not None and f_unit != u.dimensionless_unscaled:
            f_njy = np.asarray(u.Quantity(f_raw).to_value(u.nJy), dtype=np.float64)
        else:
            f_njy = np.asarray(f_raw, dtype=np.float64)
        m_band = flux_njy_to_abmag(f_njy)
        valid_m = np.isfinite(m_band)
        best_mag[valid_m] = np.minimum(best_mag[valid_m], m_band[valid_m])

    best_mag[~np.isfinite(best_mag)] = np.nan
    return ra_deg, dec_deg, best_mag, flux_cols


def build_monster_star_mask(
    pipe: SkyMaskPipe,
    butler: Any,
    footprint_mask: hsp.HealSparseMap,
    cfg: MaskPipelineConfig,
    guard: MemoryGuard,
    work_dir: Path,
    star_stage: str = "starmask",
    box_stage: str = "boxmask",
) -> MonsterStarMaskStats:
    """Query `the_monster` over the footprint and build bit-packed `SkyMaskPipe` star stage(s)."""
    ms_cfg = cfg.monster_stars
    work_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = work_dir / "monster_bright_stars_circles.parquet"

    monster_ds = discover_monster_dataset(butler, ms_cfg.dataset_type)
    buffer_deg = float(ms_cfg.max_radius_arcsec) / 3600.0

    ra_c, dec_c, search_rad_deg, dilated_128 = _coarse_footprint_centers_and_mask(
        footprint_mask, buffer_deg=buffer_deg
    )
    htm7_ids = compute_overlapping_htm7_ids(ra_c, dec_c, search_rad_deg)
    refs = query_monster_refs_for_htm7(
        butler,
        monster_ds,
        htm7_ids=htm7_ids,
        collections=ms_cfg.collections,
        batch_size=cfg.memory.htm7_query_batch_size,
    )
    logger.info(
        "Monster star query: dataset=%s, htm7_candidates=%s, shards_found=%d",
        monster_ds,
        len(htm7_ids) if htm7_ids is not None else "all-available",
        len(refs),
    )

    arrow_schema = pa.schema(
        [
            ("ra", pa.float64()),
            ("dec", pa.float64()),
            ("mag", pa.float32()),
            ("radius", pa.float32()),  # degrees (as expected by SkyMaskPipe.build_circ_mask)
        ]
    )

    writer = pq.ParquetWriter(parquet_path, arrow_schema, compression="snappy")
    buf_ra: list[np.ndarray] = []
    buf_dec: list[np.ndarray] = []
    buf_mag: list[np.ndarray] = []
    buf_rad_deg: list[np.ndarray] = []
    buffered_rows = 0
    total_stars = 0
    flux_cols_used: list[str] | None = None

    def _flush_buffer() -> None:
        nonlocal buffered_rows
        if buffered_rows == 0:
            return
        tbl = pa.Table.from_arrays(
            [
                pa.array(np.concatenate(buf_ra), type=pa.float64()),
                pa.array(np.concatenate(buf_dec), type=pa.float64()),
                pa.array(np.concatenate(buf_mag).astype(np.float32), type=pa.float32()),
                pa.array(np.concatenate(buf_rad_deg).astype(np.float32), type=pa.float32()),
            ],
            schema=arrow_schema,
        )
        writer.write_table(tbl)
        buf_ra.clear()
        buf_dec.clear()
        buf_mag.clear()
        buf_rad_deg.clear()
        buffered_rows = 0

    try:
        for idx, ref in enumerate(refs):
            shard = butler.get(ref)
            ra_deg, dec_deg, mag_ab, flux_cols_used = _extract_columns_from_shard(
                shard, flux_cols_used, ms_cfg
            )
            del shard

            # Immediate magnitude & spatial filter before keeping any arrays
            keep = (
                np.isfinite(ra_deg)
                & np.isfinite(dec_deg)
                & np.isfinite(mag_ab)
                & (mag_ab >= float(ms_cfg.mag_min))
                & (mag_ab <= float(ms_cfg.mag_max))
            )
            if np.any(keep):
                ra_k = ra_deg[keep]
                dec_k = dec_deg[keep]
                mag_k = mag_ab[keep]
                pix_128 = hpg.angle_to_pixel(128, ra_k, dec_k, nest=True, degrees=True)
                in_foot = dilated_128[pix_128]
                if np.any(in_foot):
                    ra_f = ra_k[in_foot]
                    dec_f = dec_k[in_foot]
                    mag_f = mag_k[in_foot]
                    rad_arcsec = compute_star_halo_radius_arcsec(mag_f, ms_cfg)
                    rad_deg = rad_arcsec / 3600.0

                    buf_ra.append(ra_f)
                    buf_dec.append(dec_f)
                    buf_mag.append(mag_f)
                    buf_rad_deg.append(rad_deg)
                    buffered_rows += int(ra_f.size)
                    total_stars += int(ra_f.size)

            del ra_deg, dec_deg, mag_ab, keep

            if buffered_rows >= int(cfg.memory.htm7_stream_flush_rows):
                _flush_buffer()
                gc.collect()

            if (idx + 1) % 250 == 0 or (idx + 1) == len(refs):
                guard.check(f"monster shard {idx + 1}/{len(refs)} (stars={total_stars:,})")

        _flush_buffer()
    finally:
        writer.close()

    gc.collect()
    guard.check("monster catalog streaming complete", collect_garbage=False)

    order_cov = int(cfg.property_maps.order_cov)
    order_star = int(ms_cfg.order_sparse)

    if total_stars == 0:
        logger.warning("Zero Monster stars matched the magnitude/footprint selection; creating empty starmask.")
        empty_map = hsp.HealSparseMap.make_empty(
            1 << order_cov, 1 << order_star, dtype=np.bool_, bit_packed=True
        )
        if hasattr(pipe, "_finalize_stage"):
            pipe._finalize_stage(
                empty_map,
                default_name=star_stage,
                output_stage=star_stage,
                extra_meta={"pixels": 0, "area_deg2": 0.0, "n_stars": 0},
            )
        else:
            setattr(pipe, star_stage, empty_map)
        if not ms_cfg.save_star_circles_parquet and parquet_path.exists():
            parquet_path.unlink()
        return MonsterStarMaskStats(
            dataset_type=monster_ds,
            flux_columns_used=flux_cols_used or [],
            n_htm7_trixels_queried=len(htm7_ids) if htm7_ids is not None else 0,
            n_htm7_shards_loaded=len(refs),
            n_stars_selected=0,
            n_bleed_boxes=0,
            starmask_valid_pixels=0,
            starmask_area_deg2=0.0,
            boxmask_area_deg2=0.0,
            circles_parquet_path=str(parquet_path) if ms_cfg.save_star_circles_parquet else None,
        )

    # Load compact 3-column circles DataFrame (~24 B/star -> ~48 MB for 2M stars) for SkyMaskPipe
    circles_df = pd.read_parquet(parquet_path, columns=["ra", "dec", "radius"])
    starmask = pipe.build_circ_mask(
        data=circles_df,
        order_sparse=order_star,
        order_cov=order_cov,
        columns=["ra", "dec", "radius"],
        bit_packed=True,
        n_threads=int(cfg.memory.skykatana_threads),
        chunk_size=int(cfg.memory.circle_chunk_size),
        output_stage=star_stage,
    )
    del circles_df
    gc.collect()
    guard.check("starmask pixelized", collect_garbage=False)

    n_bleed_boxes = 0
    boxmask_area_deg2 = 0.0
    if ms_cfg.enable_bleed_boxes:
        bright_df = pd.read_parquet(parquet_path, columns=["ra", "dec", "mag", "radius"])
        bright_df = bright_df[bright_df["mag"] <= float(ms_cfg.bleed_mag_max)].reset_index(drop=True)
        n_bleed_boxes = len(bright_df)
        if n_bleed_boxes > 0:
            # Note: SkyMaskPipe.pixelate_boxes accesses `table[coldec].value`, which requires an
            # Astropy Table (via file path) rather than a pandas Series.
            boxes_fits_path = work_dir / "monster_bleed_boxes.fits"
            boxes_tbl = Table(
                {
                    "ra_c": bright_df["ra"].to_numpy(dtype=np.float64),
                    "dec_c": bright_df["dec"].to_numpy(dtype=np.float64),
                    "width": bright_df["radius"].to_numpy(dtype=np.float64)
                    * float(ms_cfg.bleed_width_factor),
                    "height": bright_df["radius"].to_numpy(dtype=np.float64)
                    * float(ms_cfg.bleed_height_factor),
                }
            )
            del bright_df
            boxes_tbl.write(boxes_fits_path, format="fits", overwrite=True)
            del boxes_tbl
            boxmask = pipe.build_box_mask(
                data=boxes_fits_path,
                order_sparse=order_star,
                order_cov=order_cov,
                fmt="fits",
                columns=["ra_c", "dec_c", "width", "height"],
                bit_packed=True,
                n_threads=int(cfg.memory.skykatana_threads),
                output_stage=box_stage,
            )
            boxmask_area_deg2 = float(boxmask.get_valid_area(degrees=True))
            if not ms_cfg.save_star_circles_parquet and boxes_fits_path.exists():
                boxes_fits_path.unlink()
            gc.collect()
        else:
            del bright_df

    saved_parquet_str: str | None = str(parquet_path)
    if not ms_cfg.save_star_circles_parquet and parquet_path.exists():
        parquet_path.unlink()
        saved_parquet_str = None

    return MonsterStarMaskStats(
        dataset_type=monster_ds,
        flux_columns_used=flux_cols_used or [],
        n_htm7_trixels_queried=len(htm7_ids) if htm7_ids is not None else 0,
        n_htm7_shards_loaded=len(refs),
        n_stars_selected=total_stars,
        n_bleed_boxes=n_bleed_boxes,
        starmask_valid_pixels=int(starmask.n_valid),
        starmask_area_deg2=float(starmask.get_valid_area(degrees=True)),
        boxmask_area_deg2=boxmask_area_deg2,
        circles_parquet_path=saved_parquet_str,
    )
