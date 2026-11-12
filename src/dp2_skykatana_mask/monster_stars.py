"""Stage 2: Bright-star halo & bleed-trail masking from DP2 `the_monster` catalog.

Follows:
  - DP2 Tutorial 204.2 (`https://dp2.lsst.io/tutorials/notebook/204/notebook-204-2.html`) for
    querying ``the_monster_YYYYMMDD`` (`htm7` shards, radians coordinates, ``nJy`` synthetic
    LSST fluxes -> AB magnitudes with zero-point ``31.4000656``).
  - SkyKatana's native ``radfunction(df: pd.DataFrame, **kwargs)`` architecture
    (``functools.partial(skykatana_lsst_radfunction, band=..., completeness=...)``), which
    computes each star's exclusion halo ``'radius'`` in **degrees** as a function of its
    observed synthetic LSST magnitude and streams circles into ``SkyMaskPipe`` (`starmask`,
    plus optional saturated bleed-trail `boxmask`).

Note on Milky Way dust extinction (Tutorial 311.1):
  Instrumental diffraction spikes, optical ghosts, and scattered-light halos on LSSTCam CCDs
  depend on the *observed* photon flux reaching the telescope (not the star's intrinsic
  unreddened brightness behind Galactic dust). Furthermore, 2D SFD98 $E(B-V)$ maps integrate
  dust to infinity and over-correct foreground Milky Way stars. Therefore, ``starmask`` uses
  *observed* synthetic LSST magnitudes from ``the_monster``.
"""

from __future__ import annotations

from functools import partial
import gc
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

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

# Calibrated SkyKatana empirical Rubin halo profile parameters (arcseconds vs magnitude):
# R_arcsec(m) = A * exp(B * m), derived from angular cross-correlation completeness profiles
# around bright stars in Rubin imaging (SkyKatana Rubin bright-star mask presets).
# Keyed by band -> (A_arcsec_at_c85, B_per_mag, knee_mag, faint_floor_arcsec).
SKYKATANA_RUBIN_BAND_PARS_C85: dict[str, tuple[float, float, float, float]] = {
    "u": (980.000, -0.2420, 17.00, 9.5),
    "g": (1143.921, -0.2410, 17.50, 10.5),
    "r": (1068.4259556870957, -0.23979264155393915, 17.19, 10.2),
    "i": (1056.128, -0.2450, 17.20, 10.0),
    "z": (965.000, -0.2480, 16.80, 9.2),
    "y": (860.000, -0.2520, 16.20, 8.5),
}

# Explicit completeness presets calibrated on Rubin ECDFS/WFD fields (0.80, 0.85, 0.90)
SKYKATANA_R_COMPLETENESS_PRESETS: dict[int, tuple[float, float, float, float]] = {
    80: (945.0, -0.244, 17.0, 9.8),
    85: (1068.4259556870957, -0.23979264155393915, 17.19, 10.2),
    90: (1222.8, -0.240, 19.0, 13.0),
}


@dataclass
class MonsterStarMaskStats:
    """Summary statistics from building the Monster bright-star mask."""

    dataset_type: str
    flux_columns_used: list[str]
    radius_model: str
    profile_band: str
    completeness: float
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


def _completeness_scale_factor(completeness: float) -> float:
    """Smooth multiplicative scale relative to the reference 85% completeness profile."""
    c = float(completeness)
    if abs(c - 0.85) < 1e-4:
        return 1.0
    if abs(c - 0.90) < 1e-3:
        return 1222.8 / 1068.4259556870957
    if abs(c - 0.80) < 1e-3:
        return 945.0 / 1068.4259556870957
    # Log-odds interpolation across [0.50, 0.99] anchored at c=0.85 -> 1.0, c=0.90 -> 1.1445
    c_clamped = min(max(c, 0.50), 0.98)
    d_logit = math.log(c_clamped / (1.0 - c_clamped)) - math.log(0.85 / 0.15)
    return math.exp(0.293 * d_logit)


def _skykatana_single_band_arcsec(
    mags: np.ndarray,
    band: str = "r",
    completeness: float = 0.85,
    use_faint_floor: bool = False,
) -> np.ndarray:
    """Evaluate SkyKatana's empirical Rubin stellar halo radius [arcsec] for one LSST band."""
    m = np.asarray(mags, dtype=np.float64)
    c_scale = _completeness_scale_factor(completeness)

    if band == "r_halo":
        # SkyKatana 3-piece extended scattered-light + outer halo + core fit in Rubin r-band
        a1, b1 = 11960.396409077423, -0.387584032932514
        a2, b2 = 91530.48016015181, -0.5364916773790975
        a3, b3 = 1068.4259556870816, -0.23979264155393853
        th1 = 13.666666666666666
        th2 = 15.0
        rad = np.select(
            [m < th1, (m >= th1) & (m < th2), m >= th2],
            [
                a1 * np.exp(b1 * m),
                a2 * np.exp(b2 * m),
                a3 * np.exp(b3 * m),
            ],
            default=np.nan,
        )
        rad = rad * c_scale
        if use_faint_floor:
            rad = np.where(m > 17.5, 10.2 * c_scale, rad)
        return rad

    c_pct = int(round(100.0 * float(completeness)))
    if band == "r" and c_pct in SKYKATANA_R_COMPLETENESS_PRESETS:
        a_val, b_val, knee_mag, floor_val = SKYKATANA_R_COMPLETENESS_PRESETS[c_pct]
        rad = a_val * np.exp(b_val * m)
        if use_faint_floor:
            rad = np.where(m > knee_mag, floor_val, rad)
        return rad

    if band not in SKYKATANA_RUBIN_BAND_PARS_C85:
        raise ValueError(
            f"Unsupported SkyKatana Rubin profile band {band!r}; "
            f"expected one of {sorted([*SKYKATANA_RUBIN_BAND_PARS_C85, 'r_halo', 'multi'])}"
        )

    a_c85, b_slope, knee_mag, floor_c85 = SKYKATANA_RUBIN_BAND_PARS_C85[band]
    rad = (a_c85 * c_scale) * np.exp(b_slope * m)
    if use_faint_floor:
        rad = np.where(m > knee_mag, floor_c85 * c_scale, rad)
    return rad


def skykatana_lsst_radfunction(
    df: pd.DataFrame,
    **kwargs: Any,
) -> pd.DataFrame:
    """SkyKatana-compatible ``radfunction(df, **kwargs)`` computing stellar mask ``'radius'`` [deg]
    as a function of synthetic **LSST AB magnitude**.

    Follows the exact ``SkyMaskPipe.build_star_mask_online`` ``radfunction`` convention:
    mutates ``df['radius']`` in-place (in **degrees**, i.e., ``arcsec / 3600.0``) and returns
    ``df``, so it can be wrapped with ``functools.partial`` inside ``starq['radfunction']`` or
    called directly on ``the_monster`` chunks.

    Parameters
    ----------
    df : pandas.DataFrame
        Input star table containing an LSST magnitude column (default ``'mag'``, or
        ``'lsst_r_mag'`` / ``'ComCam_r_mag'``, or per-band ``'lsst_{b}_mag'`` columns).
    mag_col : str, default ``'mag'``
        Primary LSST magnitude column name in ``df``. If not present, automatically checks
        ``f'lsst_{band}_mag'``, ``f'ComCam_{band}_mag'``, and ``'phot_g_mean_mag'``.
    band : str, default ``'r'``
        Target LSST profile band: ``'u'``, ``'g'``, ``'r'``, ``'i'``, ``'z'``, ``'y'``,
        ``'r_halo'`` (3-piece extended scattered-light halo), or ``'multi'`` (maximum
        exclusion radius across all ``lsst_{b}_mag`` columns present in ``df``).
    completeness : float, default ``0.85``
        Target galaxy-density completeness threshold around bright stars (e.g. ``0.80``,
        ``0.85``, ``0.90``).
    radius_model : str, default ``'skykatana_rubin'``
        ``'skykatana_rubin'`` (default calibrated Rubin profiles), ``'exponential'``,
        ``'powerlaw10'``, or ``'csv_fit'``.
    use_faint_floor : bool, default ``False``
        If True, sets a constant faint-star radius plateau above the empirical knee magnitude.
    min_radius_arcsec, max_radius_arcsec : float
        Lower and upper safety bounds in arcseconds (defaults ``5.0`` and ``600.0``).
    """
    radius_model = str(kwargs.get("radius_model", "skykatana_rubin"))
    band = str(kwargs.get("band", "r"))
    completeness = float(kwargs.get("completeness", 0.85))
    use_faint_floor = bool(kwargs.get("use_faint_floor", False))
    min_rad = float(kwargs.get("min_radius_arcsec", 5.0))
    max_rad = float(kwargs.get("max_radius_arcsec", 600.0))

    # Resolve magnitude column gracefully across Monster / LSST / Gaia naming conventions
    requested_col = str(kwargs.get("mag_col", "mag"))
    base_band = "r" if band in ("r_halo", "multi") else band
    col_candidates = [
        requested_col,
        f"lsst_{base_band}_mag",
        f"ComCam_{base_band}_mag",
        "mag",
        "lsst_r_mag",
        "ComCam_r_mag",
        "phot_g_mean_mag",
    ]
    mag_col = next((c for c in col_candidates if c in df.columns), None)
    if mag_col is None and not (radius_model == "skykatana_rubin" and band == "multi"):
        raise KeyError(
            f"Could not locate LSST magnitude column in DataFrame columns {list(df.columns)}; "
            f"checked {col_candidates}"
        )

    if radius_model == "skykatana_rubin":
        if band == "multi":
            ref_bands = kwargs.get("reference_bands", ("u", "g", "r", "i", "z", "y"))
            per_band_rads: list[np.ndarray] = []
            for b in ref_bands:
                for bcol in (f"lsst_{b}_mag", f"ComCam_{b}_mag"):
                    if bcol in df.columns:
                        r_b = _skykatana_single_band_arcsec(
                            df[bcol].to_numpy(dtype=np.float64),
                            band=str(b),
                            completeness=completeness,
                            use_faint_floor=use_faint_floor,
                        )
                        per_band_rads.append(np.where(np.isfinite(r_b), r_b, -np.inf))
                        break
            if per_band_rads:
                stacked = np.vstack(per_band_rads)
                rad_arcsec = np.max(stacked, axis=0)
                rad_arcsec[~np.isfinite(rad_arcsec)] = min_rad
            else:
                assert mag_col is not None
                rad_arcsec = _skykatana_single_band_arcsec(
                    df[mag_col].to_numpy(dtype=np.float64),
                    band="r",
                    completeness=completeness,
                    use_faint_floor=use_faint_floor,
                )
        else:
            assert mag_col is not None
            rad_arcsec = _skykatana_single_band_arcsec(
                df[mag_col].to_numpy(dtype=np.float64),
                band=band,
                completeness=completeness,
                use_faint_floor=use_faint_floor,
            )
    else:
        assert mag_col is not None
        m = df[mag_col].to_numpy(dtype=np.float64)
        if radius_model == "csv_fit":
            csv_path = kwargs.get("fit_coefficients_csv")
            if not csv_path:
                raise ValueError("fit_coefficients_csv is required when radius_model='csv_fit'")
            branch = str(kwargs.get("fit_behavior_branch", "deficit"))
            coef_df = pd.read_csv(csv_path)
            if "profile_behavior" in coef_df.columns:
                sub = coef_df[coef_df["profile_behavior"] == branch]
                row = sub.iloc[0] if len(sub) > 0 else coef_df.iloc[0]
            else:
                row = coef_df.iloc[0]
            rad_arcsec = float(row["A_arcsec"]) * np.exp(float(row["B_per_mag"]) * m)
        elif radius_model == "exponential":
            a_arc = float(kwargs.get("A_arcsec", 1068.426))
            b_slp = float(kwargs.get("B_per_mag", -0.2398))
            rad_arcsec = a_arc * np.exp(b_slp * m)
            mag_knee = kwargs.get("mag_knee")
            if mag_knee is not None:
                floor_arc = float(kwargs.get("floor_arcsec", 8.0))
                rad_arcsec = np.where(m > float(mag_knee), floor_arc, rad_arcsec)
        elif radius_model == "powerlaw10":
            r0 = float(kwargs.get("r0_arcsec", 150.0))
            m0 = float(kwargs.get("m0", 10.0))
            slope = float(kwargs.get("slope_dex_per_mag", 0.16))
            rad_arcsec = r0 * (10.0 ** (slope * (m0 - m)))
        else:
            raise ValueError(f"Unsupported radius_model: {radius_model!r}")

    rad_arcsec = np.clip(rad_arcsec, min_rad, max_rad)
    # SkyKatana convention: `df['radius']` is stored in degrees!
    df["radius"] = rad_arcsec / 3600.0
    return df


def make_skykatana_radfunction(
    cfg: MonsterStarConfig,
    mag_col: str = "mag",
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Construct a ``functools.partial`` SkyKatana ``radfunction`` from ``MonsterStarConfig``."""
    return partial(
        skykatana_lsst_radfunction,
        mag_col=mag_col,
        band=cfg.profile_band,
        completeness=cfg.completeness,
        radius_model=cfg.radius_model,
        use_faint_floor=cfg.use_faint_floor,
        reference_bands=tuple(cfg.reference_bands),
        A_arcsec=cfg.A_arcsec,
        B_per_mag=cfg.B_per_mag,
        mag_knee=cfg.mag_knee,
        floor_arcsec=cfg.floor_arcsec,
        r0_arcsec=cfg.r0_arcsec,
        m0=cfg.m0,
        slope_dex_per_mag=cfg.slope_dex_per_mag,
        fit_coefficients_csv=cfg.fit_coefficients_csv,
        fit_behavior_branch=cfg.fit_behavior_branch,
        min_radius_arcsec=cfg.min_radius_arcsec,
        max_radius_arcsec=cfg.max_radius_arcsec,
    )


def compute_star_halo_radius_arcsec(mags: np.ndarray, cfg: MonsterStarConfig) -> np.ndarray:
    """Evaluate exclusion halo radius [arcsec] for an array of LSST magnitudes via ``radfunction``."""
    tmp_df = pd.DataFrame({"mag": np.asarray(mags, dtype=np.float64)})
    rad_fn = make_skykatana_radfunction(cfg, mag_col="mag")
    rad_fn(tmp_df)
    return tmp_df["radius"].to_numpy(dtype=np.float64) * 3600.0


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


def _infer_band_from_flux_col(col_name: str) -> str | None:
    parts = col_name.split("_")
    # e.g. monster_ComCam_r_flux -> 'r'
    if len(parts) >= 3 and parts[-1] == "flux" and parts[-2] in ("u", "g", "r", "i", "z", "y"):
        return parts[-2]
    return None


def _extract_columns_from_shard(
    shard: Any,
    flux_cols_cache: list[str] | None,
    cfg: MonsterStarConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], list[str]]:
    """Extract ``(ra_deg, dec_deg, primary_lsst_mag, band_mags, flux_cols)`` from a shard."""
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

    # Ensure profile_band's flux column is included when resolving single-band LSST magnitudes
    target_band = "r" if cfg.profile_band in ("r_halo", "multi") else cfg.profile_band
    bands_to_resolve = list(cfg.reference_bands)
    if target_band not in bands_to_resolve and target_band in ("u", "g", "r", "i", "z", "y"):
        bands_to_resolve = [target_band, *bands_to_resolve]

    flux_cols = flux_cols_cache or resolve_monster_flux_columns(
        col_names,
        bands_to_resolve,
        cfg.explicit_flux_column,
    )

    ra_col = "coord_ra" if "coord_ra" in col_names else "ra"
    dec_col = "coord_dec" if "coord_dec" in col_names else "dec"

    if is_afw:
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

    band_mags: dict[str, np.ndarray] = {}
    best_mag = np.full(ra_deg.shape, np.inf, dtype=np.float64)
    for fc in flux_cols:
        f_raw = shard[fc]
        f_unit = getattr(f_raw, "unit", None)
        if f_unit is not None and f_unit != u.dimensionless_unscaled:
            f_njy = np.asarray(u.Quantity(f_raw).to_value(u.nJy), dtype=np.float64)
        else:
            f_njy = np.asarray(f_raw, dtype=np.float64)
        m_band = flux_njy_to_abmag(f_njy)
        b_tag = _infer_band_from_flux_col(fc)
        if b_tag is not None:
            band_mags[f"lsst_{b_tag}_mag"] = m_band
        valid_m = np.isfinite(m_band)
        best_mag[valid_m] = np.minimum(best_mag[valid_m], m_band[valid_m])

    best_mag[~np.isfinite(best_mag)] = np.nan

    # Primary LSST magnitude: use target profile band (`lsst_r_mag` etc.) where valid,
    # falling back to brightest valid reference band for stars saturated/missing in one band!
    target_col = f"lsst_{target_band}_mag"
    if cfg.profile_band != "multi" and target_col in band_mags:
        primary_mag = np.where(np.isfinite(band_mags[target_col]), band_mags[target_col], best_mag)
    else:
        primary_mag = best_mag

    return ra_deg, dec_deg, primary_mag, band_mags, flux_cols


def build_monster_star_mask(
    pipe: SkyMaskPipe,
    butler: Any,
    footprint_mask: hsp.HealSparseMap,
    cfg: MaskPipelineConfig,
    guard: MemoryGuard,
    work_dir: Path,
    star_stage: str = "starmask",
    box_stage: str = "boxmask",
    radfunction: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> MonsterStarMaskStats:
    """Query `the_monster` over the footprint, evaluate SkyKatana ``radfunction`` on LSST
    magnitudes, and build bit-packed ``SkyMaskPipe`` star stage(s).
    """
    ms_cfg = cfg.monster_stars
    work_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = work_dir / "monster_bright_stars_circles.parquet"

    active_radfn = radfunction if radfunction is not None else make_skykatana_radfunction(ms_cfg)

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
        "Monster star query: dataset=%s, htm7_candidates=%s, shards_found=%d, "
        "radius_model=%s (band=%s, completeness=%.2f)",
        monster_ds,
        len(htm7_ids) if htm7_ids is not None else "all-available",
        len(refs),
        ms_cfg.radius_model,
        ms_cfg.profile_band,
        ms_cfg.completeness,
    )

    arrow_schema = pa.schema(
        [
            ("ra", pa.float64()),
            ("dec", pa.float64()),
            ("mag", pa.float32()),
            ("radius", pa.float32()),  # degrees (computed by SkyKatana radfunction!)
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
            ra_deg, dec_deg, mag_ab, band_mags, flux_cols_used = _extract_columns_from_shard(
                shard, flux_cols_used, ms_cfg
            )
            del shard

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

                    # Build chunk DataFrame with synthetic LSST magnitudes and invoke SkyKatana's
                    # `radfunction(df)` directly to populate `df['radius']` in degrees!
                    df_cols: dict[str, np.ndarray] = {
                        "ra": ra_f,
                        "dec": dec_f,
                        "mag": mag_f,
                    }
                    for bm_name, bm_arr in band_mags.items():
                        df_cols[bm_name] = bm_arr[keep][in_foot]
                    chunk_df = pd.DataFrame(df_cols)
                    active_radfn(chunk_df)
                    rad_deg = chunk_df["radius"].to_numpy(dtype=np.float64)
                    del chunk_df, df_cols

                    buf_ra.append(ra_f)
                    buf_dec.append(dec_f)
                    buf_mag.append(mag_f)
                    buf_rad_deg.append(rad_deg)
                    buffered_rows += int(ra_f.size)
                    total_stars += int(ra_f.size)

            del ra_deg, dec_deg, mag_ab, band_mags, keep

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
            radius_model=ms_cfg.radius_model,
            profile_band=ms_cfg.profile_band,
            completeness=float(ms_cfg.completeness),
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
        radius_model=ms_cfg.radius_model,
        profile_band=ms_cfg.profile_band,
        completeness=float(ms_cfg.completeness),
        n_htm7_trixels_queried=len(htm7_ids) if htm7_ids is not None else 0,
        n_htm7_shards_loaded=len(refs),
        n_stars_selected=total_stars,
        n_bleed_boxes=n_bleed_boxes,
        starmask_valid_pixels=int(starmask.n_valid),
        starmask_area_deg2=float(starmask.get_valid_area(degrees=True)),
        boxmask_area_deg2=boxmask_area_deg2,
        circles_parquet_path=saved_parquet_str,
    )


def build_monster_hats_star_mask_online(
    pipe: SkyMaskPipe,
    starq: dict[str, Any],
    order_sparse: int = 15,
    order_cov: int | None = None,
    bit_packed: bool = True,
    n_threads: int = 2,
    chunk_size: int = 100_000,
    output_stage: str = "starmask",
) -> hsp.HealSparseMap:
    """Drop-in ``SkyMaskPipe.build_star_mask_online`` adapter for ``the_monster`` HATS catalogs.

    Upstream ``SkyMaskPipe.build_star_mask_online`` filters on Gaia ``phot_g_mean_mag``, whereas
    ``the_monster`` (DMTN-277) stores synthetic LSST fluxes (e.g. ``monster_ComCam_r_flux``) in
    ``nJy``. This helper converts ``lsst_mag_lims`` ``[m_min, m_max]`` into ``nJy`` flux bounds,
    streams MOC chunks via ``lsdb``, computes ``lsst_{band}_mag``, calls ``starq['radfunction']``,
    and pixelizes directly into ``pipe`` via ``pipe.pixelate_circles``.
    """
    import lsdb
    from lsdb.core.search.region_search import MOCSearch
    from mocpy import MOC
    from skykatana.skykatana import getarea_moc, split_moc_into_chunks

    ord_cov = order_cov if order_cov is not None else pipe.order_cov
    nside_sparse = 1 << int(order_sparse)
    nside_cov = 1 << int(ord_cov)
    mask = hsp.HealSparseMap.make_empty(nside_cov, nside_sparse, dtype=np.bool_, bit_packed=True)

    search_stage = starq["search_stage"]
    order_search = int(round(math.log2(int(search_stage.nside_sparse))))
    max_depth = 8
    moc = MOC.from_healpix_cells(
        ipix=search_stage.valid_pixels, depth=order_search, max_depth=max_depth
    )

    if starq.get("avoid_mw", False):
        moc_pb = SkyMaskPipe.gal_plane_bulge_moc(
            max_depth=max_depth,
            b0_deg=float(starq.get("b0_deg", 10.0)),
            bulge_a_deg=float(starq.get("bulge_a_deg", 15.0)),
            bulge_b_deg=float(starq.get("bulge_b_deg", 10.0)),
        )
        moc = moc.difference(moc_pb)

    band = str(starq.get("lsst_band", "r"))
    flux_col = str(starq.get("flux_col", f"monster_ComCam_{band}_flux"))
    mag_col = f"lsst_{band}_mag"
    m_min, m_max = starq.get("lsst_mag_lims", (-5.0, 18.0))
    flux_min = 10.0 ** ((AB_MAG_ZEROPOINT_NJY - float(m_max)) / 2.5)
    flux_max = 10.0 ** ((AB_MAG_ZEROPOINT_NJY - float(m_min)) / 2.5)

    ra_col, dec_col = starq.get("coord_columns", ("ra", "dec"))
    cat = lsdb.open_catalog(
        starq["cat"],
        columns=[ra_col, dec_col, flux_col],
        search_filter=MOCSearch(moc),
        filters=[[flux_col, ">=", flux_min], [flux_col, "<=", flux_max]],
    )

    max_area_single = float(starq.get("max_area_single", 400.0))
    target_chunk_area = float(starq.get("target_chunk_area", 300.0))
    coarse_order_bfs = int(starq.get("coarse_order_bfs", 5))

    if getarea_moc(moc) < max_area_single:
        chunk_mocs = [moc.add_neighbours()]
    else:
        chunk_mocs = split_moc_into_chunks(
            moc, target_deg2=target_chunk_area, coarse_order=coarse_order_bfs
        )
        chunk_mocs = [c.intersection(moc).add_neighbours() for c in chunk_mocs]

    rad_fn = starq.get(
        "radfunction",
        partial(skykatana_lsst_radfunction, mag_col=mag_col, band=band, completeness=0.85),
    )

    for mi in chunk_mocs:
        s = cat.moc_search(moc=mi).compute()
        if len(s) == 0:
            continue
        s = s.rename(columns={ra_col: "ra", dec_col: "dec"})
        s[mag_col] = flux_njy_to_abmag(s[flux_col].to_numpy(dtype=np.float64))
        s["mag"] = s[mag_col]
        rad_fn(s)
        pipe.pixelate_circles(
            s,
            mask,
            order=int(order_sparse),
            columns=["ra", "dec", "radius"],
            n_threads=int(n_threads),
            chunk_size=int(chunk_size),
        )
        del s
        gc.collect()

    if bit_packed:
        mask = mask.as_bit_packed_map()

    extra_meta = {
        "flux_col": flux_col,
        "lsst_band": band,
        "lsst_mag_lims": [float(m_min), float(m_max)],
        "pixels": int(mask.n_valid),
        "area_deg2": float(mask.get_valid_area(degrees=True)),
        "bit_packed": bool(bit_packed),
    }
    return pipe._finalize_stage(
        mask, default_name="starmask", output_stage=output_stage, extra_meta=extra_meta
    )
