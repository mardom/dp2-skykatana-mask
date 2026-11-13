"""Multi-scale, 32 GiB RSP-safe SkyKatana visualizations for complete Rubin DP2 masks.

Why multi-scale visualization is essential for complete DP2 survey masks
------------------------------------------------------------------------
A full Rubin DP2 mask at HEALPix order 15 ($0.64''$ pixels) spans **five orders of
magnitude** in angular scale:

- **Macro survey geometry** ($1^\\circ$–$360^\\circ$): Wide-Fast-Deep strips crossing the
  $\\mathrm{RA} = 0^\\circ / 360^\\circ$ wrap, Deep Drilling Fields, Galactic plane/bulge
  vetoes (``mwmask``), and high-$E(B-V)$ cirrus vetoes (``dustmask``).
- **Micro star-halo & bleed geometry** ($5''$–$150''$, i.e. $0.0014^\\circ$–$0.04^\\circ$):
  Individual ``the_monster`` bright-star halos sized by ``skykatana_lsst_radfunction``
  and vertical CCD bleed trails (``boxmask``).

This module wraps SkyKatana's native visualization methods (``SkyMaskPipe.plot_moc``,
``SkyMaskPipe.plot_fracmap``, ``SkyMaskPipe.plot``, ``SkyMaskPipe.pix_in_zone``,
``SkyMaskPipe.plot_srcs``, and ``SkyMaskPipe.gal_plane_bulge_moc``) with:

1. **Streaming wrap-aware sky framing** (no ``.valid_pixels`` materialization).
2. **Distortion-free equatorial projections**: tangent ``SIN`` with exact 2D ``[fov_w, fov_h]``
   for regional footprints, and equatorial ``CAR`` (``CRVAL2 = 0``) cropped to the occupied
   declination band for wide surveys so $\\mathrm{RA} = 0^\\circ / 360^\\circ$ never splits the
   footprint and parallels/meridians remain straight.
3. **Dense interior cutout extraction** via ``SkyMaskPipe.pix_in_zone`` so zoomed
   ``SkyMaskPipe.plot`` panels always render 45,000 crisp random points alongside
   analytical red ``R(m_LSST)`` star circles and CCD bleed boxes, even on 6,500 deg$^2$ masks.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import astropy.units as u
import healpy as hp
import healsparse as hsp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import Angle, SkyCoord
from mocpy import WCS as MocWCS
from skykatana import SkyMaskPipe


def compute_sky_extent(
    pipe: SkyMaskPipe,
    stage: str = "mask",
    n_probe: int = 40_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute wrap-aware RA/Dec bounds from a streamed ``makerans`` sample (<20 ms, <5 MB RAM).

    Uses the largest empty gap on $S^1$ (`[0, 360)`) so footprints crossing
    ``RA = 0 deg / 360 deg`` yield a tight continuous ``(ra_start, ra_span, ra_c)`` arc
    rather than artificially spanning ``[0, 360]``.
    """
    hsp_map = getattr(pipe, stage, None)
    if hsp_map is None or int(hsp_map.n_valid) == 0:
        return {
            "ra_c": 0.0,
            "dec_c": -25.0,
            "ra_start": 0.0,
            "ra_span": 180.0,
            "dec_min": -45.0,
            "dec_max": 10.0,
            "dec_span": 55.0,
            "wraps_zero": False,
            "wide_survey": True,
        }

    nr = min(int(n_probe), max(4_000, int(hsp_map.n_valid)))
    pts = pipe.makerans(stage=stage, nr=nr, rng=np.random.RandomState(seed))
    ra = np.sort(pts["ra"].to_numpy(dtype=np.float64) % 360.0)
    dec = pts["dec"].to_numpy(dtype=np.float64)

    dec_min = float(dec.min())
    dec_max = float(dec.max())
    dec_c = 0.5 * (dec_min + dec_max)
    dec_span = max(dec_max - dec_min, 0.25)

    gaps = np.diff(np.concatenate([ra, [ra[0] + 360.0]]))
    i_max = int(np.argmax(gaps))
    ra_start = float(ra[(i_max + 1) % ra.size])
    ra_span = max(360.0 - float(gaps[i_max]), 0.25)
    ra_c = float((ra_start + 0.5 * ra_span) % 360.0)
    wraps_zero = bool((ra_start + ra_span) > 360.0)
    wide_survey = bool((ra_span > 55.0) or (dec_span > 40.0))

    return {
        "ra_c": ra_c,
        "dec_c": dec_c,
        "ra_start": ra_start,
        "ra_span": ra_span,
        "dec_min": dec_min,
        "dec_max": dec_max,
        "dec_span": dec_span,
        "wraps_zero": wraps_zero,
        "wide_survey": wide_survey,
    }


def _build_wcs_framing(ext: dict[str, Any]) -> tuple[str, SkyCoord, list[Angle], tuple[float, float]]:
    """Construct `(projection, center_skycoord, [fov_lon, fov_lat], figsize)` for `mocpy.WCS`."""
    if ext["wide_survey"]:
        proj = "CAR"
        fov_lon = min(360.0, float(ext["ra_span"]) * 1.10)
        max_abs_dec = max(abs(float(ext["dec_min"])), abs(float(ext["dec_max"])), 20.0)
        fov_lat = min(176.0, 2.0 * (max_abs_dec + 8.0))
        # Equatorial reference (CRVAL2 = 0.0) prevents oblique pole rotation in cylindrical WCS!
        center_sc = SkyCoord(ra=float(ext["ra_c"]) * u.deg, dec=0.0 * u.deg, frame="icrs")
        fov_seq = [Angle(fov_lon, "deg"), Angle(fov_lat, "deg")]
        vis_dec_span = max(float(ext["dec_span"]) + 12.0, 26.0)
        aspect = float(np.clip(fov_lon / vis_dec_span, 1.75, 2.9))
        figsize = (11.0, round(11.0 / aspect + 0.75, 2))
    else:
        proj = "SIN"
        cos_d = max(float(np.cos(np.radians(ext["dec_c"]))), 0.15)
        sky_w = max(float(ext["ra_span"]) * cos_d * 1.16, 0.35)
        sky_h = max(float(ext["dec_span"]) * 1.16, 0.35)
        center_sc = SkyCoord(
            ra=float(ext["ra_c"]) * u.deg,
            dec=float(ext["dec_c"]) * u.deg,
            frame="icrs",
        )
        fov_seq = [Angle(sky_w, "deg"), Angle(sky_h, "deg")]
        figsize = (7.8, 7.2)
    return proj, center_sc, fov_seq, figsize


def _crop_car_declination_band(
    ax: plt.Axes,
    wcs: Any,
    ra_c: float,
    dec_min: float,
    dec_max: float,
    pad_deg: float = 6.0,
) -> None:
    """Crop vertical pixel limits of an equatorial ``CAR`` WCS axes to ``[dec_min-pad, dec_max+pad]``."""
    d_lo = max(-88.0, float(dec_min) - pad_deg)
    d_hi = min(88.0, float(dec_max) + pad_deg)
    _, y_pix = wcs.world_to_pixel(
        SkyCoord(ra=[ra_c, ra_c] * u.deg, dec=[d_lo, d_hi] * u.deg, frame="icrs")
    )
    y0, y1 = float(min(y_pix)), float(max(y_pix))
    if np.isfinite(y0) and np.isfinite(y1) and (y1 - y0) > 5.0:
        ax.set_ylim(y0, y1)


def select_dense_zoom_box(
    pipe: SkyMaskPipe,
    circles_path: str | Path | None = None,
    half_width_deg: float = 0.25,
    zoom_center: tuple[float, float] | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Choose a well-populated interior ``(clip_ra, clip_dec)`` window for star-halo inspection.

    If ``zoom_center=(ra_deg, dec_deg)`` is supplied explicitly, centers there. Otherwise
    selects a high-occupancy interior coarse coverage tile of ``pipe.mask`` and centers near
    a prominent bright Monster star inside that interior region so both circular
    ``R(m_LSST)`` halos and CCD bleed-trail boxes can be appreciated clearly.
    """
    if zoom_center is not None:
        z_ra0 = float(zoom_center[0]) % 360.0
        z_dec0 = float(np.clip(zoom_center[1], -89.0, 89.0))
    else:
        mask_map = pipe.mask
        star_map = getattr(pipe, "starmask", None)
        cov_indices = np.where(mask_map.coverage_mask)[0]
        if cov_indices.size == 0:
            return (0.0, 0.5), (-28.3, -27.8)

        step = max(1, cov_indices.size // 96)
        sampled_cov = [int(c) for c in cov_indices[::step]]
        counts = {c: int(mask_map.valid_pixels_single_covpix(c).size) for c in sampled_cov}
        max_count = max(counts.values()) if counts else 0

        picked_from_catalog = False
        if circles_path is not None and Path(circles_path).is_file() and max_count > 0:
            st_df = pd.read_parquet(circles_path, columns=["ra", "dec", "radius"])
            if len(st_df) > 0:
                # Inspect up to top 250 largest-radius stars and choose the first sitting inside
                # a dense interior coarse tile (>= 55% of peak tile occupancy) away from RA=0/360 seam.
                top_st = st_df.nlargest(min(250, len(st_df)), "radius")
                cand_cov = hp.ang2pix(
                    mask_map.nside_coverage,
                    top_st["ra"].to_numpy(dtype=np.float64),
                    top_st["dec"].to_numpy(dtype=np.float64),
                    nest=True,
                    lonlat=True,
                )
                for row_idx, c_val in enumerate(cand_cov):
                    c_int = int(c_val)
                    if not mask_map.coverage_mask[c_int]:
                        continue
                    n_here = counts.get(c_int)
                    if n_here is None:
                        n_here = int(mask_map.valid_pixels_single_covpix(c_int).size)
                        counts[c_int] = n_here
                    r_cand = float(top_st["ra"].iloc[row_idx]) % 360.0
                    d_cand = float(top_st["dec"].iloc[row_idx])
                    if n_here >= 0.55 * max_count and 1.0 < r_cand < 359.0:
                        # Slight ~0.04 deg offset frames the bright star + neighbors naturally
                        z_ra0 = (r_cand + 0.04) % 360.0
                        z_dec0 = float(np.clip(d_cand + 0.03, -89.0, 89.0))
                        picked_from_catalog = True
                        break

        if not picked_from_catalog:
            best_cov = sampled_cov[0]
            best_score = -1.0
            for c_int, n_m in counts.items():
                if n_m == 0:
                    continue
                n_s = (
                    int(star_map.valid_pixels_single_covpix(c_int).size)
                    if star_map is not None
                    else 0
                )
                score = n_m * (1.35 if n_s > 0 else 1.0)
                if score > best_score:
                    best_score = score
                    best_cov = c_int

            pix = mask_map.valid_pixels_single_covpix(best_cov)
            ra_p, dec_p = hp.pix2ang(mask_map.nside_sparse, pix, nest=True, lonlat=True)
            ang_rad = np.radians(ra_p)
            z_ra0 = float(
                np.degrees(np.arctan2(np.median(np.sin(ang_rad)), np.median(np.cos(ang_rad))))
                % 360.0
            )
            z_dec0 = float(np.median(dec_p))

    cos_d = max(float(np.cos(np.radians(z_dec0))), 0.15)
    hw_ra = float(half_width_deg) / cos_d
    if hw_ra < 20.0 and (z_ra0 < hw_ra + 0.02 or z_ra0 > 360.0 - hw_ra - 0.02):
        z_ra0 = float(np.clip(z_ra0, hw_ra + 0.05, 360.0 - hw_ra - 0.05))
    return (z_ra0 - hw_ra, z_ra0 + hw_ra), (z_dec0 - half_width_deg, z_dec0 + half_width_deg)


def extract_zoom_subpipe(
    pipe: SkyMaskPipe,
    clip_ra: tuple[float, float],
    clip_dec: tuple[float, float],
    stage: str = "mask",
) -> SkyMaskPipe:
    """Extract a lightweight single-window ``SkyMaskPipe`` (<1 MB RAM) via ``SkyMaskPipe.pix_in_zone``.

    Calling ``sub_pipe.plot(stage='mask', nr=45_000, ...)`` then concentrates all 45,000
    random points inside the zoomed window in ~0.15 s, regardless of whether the parent
    survey mask covers 4 deg$^2$ or 6,500 deg$^2$.
    """
    src_map = getattr(pipe, stage)
    pad_ra = 0.06 * (clip_ra[1] - clip_ra[0])
    pad_dec = 0.06 * (clip_dec[1] - clip_dec[0])
    ra_lo = max(0.0, float(clip_ra[0]) - pad_ra)
    ra_hi = min(360.0, float(clip_ra[1]) + pad_ra)
    dec_lo = max(-90.0, float(clip_dec[0]) - pad_dec)
    dec_hi = min(90.0, float(clip_dec[1]) + pad_dec)

    sub_pix = SkyMaskPipe.pix_in_zone(src_map, ra_lo, ra_hi, dec_lo, dec_hi, return_mode="array")
    order_sparse = int(round(np.log2(src_map.nside_sparse)))
    order_cov = int(round(np.log2(src_map.nside_coverage)))

    sub_pipe = SkyMaskPipe(order_out=order_sparse, order_cov=order_cov)
    sub_map = hsp.HealSparseMap.make_empty(
        src_map.nside_coverage,
        src_map.nside_sparse,
        dtype=np.bool_,
        bit_packed=True,
    )
    if sub_pix.size > 0:
        sub_map.update_values_pix(sub_pix, True)
    setattr(sub_pipe, stage, sub_map)
    return sub_pipe


def plot_complete_mask_multiscale(
    pipe: SkyMaskPipe,
    circles_path: str | Path | None = None,
    boxes_path: str | Path | None = None,
    zoom_half_width_deg: float = 0.25,
    zoom_center: tuple[float, float] | None = None,
    nr_zoom: int = 45_000,
    show: bool = True,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Render a 2-panel complete-mask overview + star-hole cutout using SkyKatana tools.

    - **Left panel (Macro footprint & vetoes)**: ``SkyMaskPipe.plot_moc`` on a tight
      wrap-aware WCS projection (``SIN`` for regional masks, equatorial ``CAR`` for wide
      surveys), layering ``propmask``, ``dustmask``, ``mwmask``, ``starmask`` (regional),
      and the surviving combined ``mask``.
    - **Right panel (Micro star halos & bleed trails)**: Extracts the zoomed cutout via
      ``SkyMaskPipe.pix_in_zone`` and calls ``SkyMaskPipe.plot`` with analytical Monster
      ``R(m_LSST)`` circles (``plot_circles``) and CCD bleed boxes (``plot_boxes``).
    """
    ext = compute_sky_extent(pipe, "mask")
    wide = bool(ext["wide_survey"])
    proj, center_sc, fov_seq, _ = _build_wcs_framing(ext)
    full_area = float(pipe.mask.get_valid_area(degrees=True))

    clip_ra_z, clip_dec_z = select_dense_zoom_box(
        pipe,
        circles_path=circles_path,
        half_width_deg=zoom_half_width_deg,
        zoom_center=zoom_center,
    )
    zoom_pipe = extract_zoom_subpipe(pipe, clip_ra_z, clip_dec_z, stage="mask")

    circ_dict: dict[str, Any] | bool = (
        {"data": str(circles_path), "fmt": "parquet", "columns": ["ra", "dec", "radius"]}
        if (circles_path is not None and Path(circles_path).is_file())
        else False
    )
    box_dict: dict[str, Any] | bool = (
        {"data": str(boxes_path), "fmt": "fits", "columns": ["ra_c", "dec_c", "width", "height"]}
        if (boxes_path is not None and Path(boxes_path).is_file())
        else False
    )

    fig = plt.figure(figsize=(14.2, 5.8) if wide else (13.6, 6.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0] if wide else [1.05, 1.0], wspace=0.22)

    with MocWCS(
        fig,
        fov=fov_seq,
        center=center_sc,
        coordsys="icrs",
        projection=proj,
        rotation=Angle(0, u.deg),
    ) as wcs_ov:
        ax_ov = fig.add_subplot(gs[0, 0], projection=wcs_ov)
        ax_ov.coords[0].set_format_unit(u.deg, decimal=True, show_decimal_unit=True)
        ax_ov.coords[1].set_format_unit(u.deg, decimal=True, show_decimal_unit=True)
        ax_ov.set_xlabel("RA")
        ax_ov.set_ylabel("Dec")
        ax_ov.grid(color="black", linestyle="dotted", alpha=0.5)

    if getattr(pipe, "propmask", None) is not None and int(pipe.propmask.n_valid) > 0:
        pipe.plot_moc(
            "propmask",
            ax=ax_ov,
            wcs=wcs_ov,
            color="#94a3b8",
            alpha=0.38,
            linewidth=0.0,
            label="propmask (initial depth)",
            show=False,
        )
    if getattr(pipe, "dustmask", None) is not None and int(pipe.dustmask.n_valid) > 0:
        pipe.plot_moc(
            "dustmask",
            ax=ax_ov,
            wcs=wcs_ov,
            color="#f97316",
            alpha=0.70,
            linewidth=0.0,
            label="dustmask (high E(B-V) veto)",
            show=False,
        )
    if getattr(pipe, "mwmask", None) is not None and int(pipe.mwmask.n_valid) > 0 and wide:
        pipe.plot_moc(
            "mwmask",
            ax=ax_ov,
            wcs=wcs_ov,
            color="#eab308",
            alpha=0.20,
            linewidth=0.4,
            label="mwmask (Galactic plane/bulge)",
            show=False,
        )
    pipe.plot_moc(
        "mask",
        ax=ax_ov,
        wcs=wcs_ov,
        color="#0f766e",
        alpha=0.88,
        linewidth=0.0,
        label=f"Final mask ({full_area:,.2f} deg²)",
        show=False,
    )
    if getattr(pipe, "starmask", None) is not None and int(pipe.starmask.n_valid) > 0 and not wide:
        pipe.plot_moc(
            "starmask",
            ax=ax_ov,
            wcs=wcs_ov,
            color="#dc2626",
            alpha=0.75,
            linewidth=0.0,
            label="starmask (Monster halos)",
            show=False,
        )

    if wide:
        _crop_car_declination_band(
            ax_ov, wcs_ov, ext["ra_c"], ext["dec_min"], ext["dec_max"], pad_deg=6.0
        )

    box_ra = [clip_ra_z[0], clip_ra_z[1], clip_ra_z[1], clip_ra_z[0], clip_ra_z[0]]
    box_dec = [clip_dec_z[0], clip_dec_z[0], clip_dec_z[1], clip_dec_z[1], clip_dec_z[0]]
    px_z, py_z = wcs_ov.world_to_pixel(
        SkyCoord(ra=box_ra * u.deg, dec=box_dec * u.deg, frame="icrs")
    )
    ax_ov.plot(
        px_z,
        py_z,
        color="#dc2626",
        linewidth=2.0,
        zorder=10,
        label="Zoom cutout" if wide else None,
    )
    if wide:
        cx_z, cy_z = float(np.mean(px_z[:4])), float(np.mean(py_z[:4]))
        ax_ov.scatter(
            [cx_z],
            [cy_z],
            s=95,
            facecolors="none",
            edgecolors="#dc2626",
            linewidths=2.0,
            zorder=11,
        )

    ax_ov.set_title(f"Complete Mask Footprint — SkyMaskPipe.plot_moc ({proj})", pad=10)
    ax_ov.legend(loc="upper right", fontsize=8.0, framealpha=0.92)

    ax_zm = fig.add_subplot(gs[0, 1])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*tight_layout.*", category=UserWarning)
        zoom_pipe.plot(
            stage="mask",
            nr=int(nr_zoom),
            s=0.75,
            clipra=clip_ra_z,
            clipdec=clip_dec_z,
            plot_circles=circ_dict,
            plot_boxes=box_dict,
            ax=ax_zm,
            color="#0f766e",
            alpha=0.68,
        )
    ax_zm.set_xlim(clip_ra_z[1], clip_ra_z[0])  # Astronomical East-left
    ax_zm.set_ylim(clip_dec_z[0], clip_dec_z[1])
    ax_zm.set_xlabel("RA [deg]")
    ax_zm.set_ylabel("Dec [deg]")
    span_deg = clip_dec_z[1] - clip_dec_z[0]
    ax_zm.set_title(
        f"Interior Cutout ({span_deg:.2f}° × {span_deg:.2f}°) — Monster Halos & Bleed Trails",
        pad=10,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, (ax_ov, ax_zm)


def plot_complete_mask_fracmap(
    pipe: SkyMaskPipe,
    order_frac: int | None = None,
    cmap: str = "Spectral_r",
    avg_edges: bool = False,
    show: bool = True,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes, Any]:
    """Render high-contrast fractional area coverage $f \\in [0, 1]$ via ``SkyMaskPipe.plot_fracmap``."""
    ext = compute_sky_extent(pipe, "mask")
    wide = bool(ext["wide_survey"])
    proj, center_sc, fov_seq, figsize_wcs = _build_wcs_framing(ext)

    if order_frac is None:
        eff_order_frac = 8 if wide else min(11, max(6, int(pipe.order_out) - 1))
    else:
        eff_order_frac = int(order_frac)

    fig, ax, wcs, im, _ = pipe.plot_fracmap(
        "mask",
        order_frac=eff_order_frac,
        center=center_sc,
        fov=fov_seq,
        projection=proj,
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
        avg_edges=bool(avg_edges),
        thresholds=[0.5, 0.9] if not wide else None,
        contour_kwargs={"colors": "black", "linewidths": 0.5, "alpha": 0.55},
        figsize=(figsize_wcs[0] + 0.8, figsize_wcs[1]),
    )
    ax.set_title(
        f"Fractional Coverage f ∈ [0, 1] — plot_fracmap (order={eff_order_frac})",
        pad=12,
    )

    if wide:
        _crop_car_declination_band(
            ax, wcs, ext["ra_c"], ext["dec_min"], ext["dec_max"], pad_deg=6.0
        )
        # Align colorbar height with the vertically cropped CAR WCS panel
        if getattr(im, "colorbar", None) is not None:
            fig.canvas.draw()
            pos = ax.get_position()
            cbar_ax = im.colorbar.ax
            cpos = cbar_ax.get_position()
            cbar_ax.set_position([pos.x1 + 0.015, pos.y0, cpos.width, pos.height])

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax, wcs


def plot_complete_mask_allsky(
    pipe: SkyMaskPipe,
    show: bool = True,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes, Any]:
    """Render full $4\\pi$ equatorial Mollweide celestial context via ``SkyMaskPipe.plot_moc``."""
    ext = compute_sky_extent(pipe, "mask")
    wide = bool(ext["wide_survey"])
    full_area = float(pipe.mask.get_valid_area(degrees=True))

    allsky_center = SkyCoord(ra=float(ext["ra_c"]) * u.deg, dec=0.0 * u.deg, frame="icrs")
    allsky_fov = [Angle(360.0, "deg"), Angle(180.0, "deg")]

    fig, ax, wcs = pipe.plot_moc(
        "mask",
        center=allsky_center,
        fov=allsky_fov,
        projection="MOL",
        color="#0f766e",
        alpha=0.90,
        linewidth=0.6,
        label=f"DP2 surviving mask ({full_area:,.2f} deg²)",
        figsize=(10.2, 5.2),
        show=False,
    )

    if getattr(pipe, "mwmask", None) is not None and int(pipe.mwmask.n_valid) > 0:
        pipe.plot_moc(
            "mwmask",
            ax=ax,
            wcs=wcs,
            color="#eab308",
            alpha=0.22,
            linewidth=0.4,
            label="mwmask (Galactic plane + bulge)",
            show=False,
        )
    else:
        mw_ref = SkyMaskPipe.gal_plane_bulge_moc(max_depth=6, b0_deg=10.0)
        mw_ref.fill(
            ax=ax,
            wcs=wcs,
            alpha=0.16,
            fill=True,
            color="#eab308",
            linewidth=0,
            label="Galactic plane |b| < 10° (ref)",
        )

    if not wide:
        pipe.plot_srcs(
            [ext["ra_c"]],
            [ext["dec_c"]],
            ax=ax,
            wcs=wcs,
            marker="o",
            s=115,
            color="none",
            edgecolor="#dc2626",
            alpha=1.0,
            linewidths=1.9,
            label=f"Field location ({ext['ra_c']:.1f}°, {ext['dec_c']:.1f}°)",
            show=False,
        )

    # Annotate celestial equator meridians & central parallels inside the Mollweide globe
    world_tr = ax.get_transform("world")
    for ra_lbl in (0, 60, 120, 180, 240, 300):
        # Skip the seam edge at ra_c +/- 180 where projection wraps
        d_edge = abs(((ra_lbl - float(ext["ra_c"]) + 180.0) % 360.0) - 180.0)
        if d_edge < 168.0:
            ax.text(
                float(ra_lbl),
                2.5,
                f"{ra_lbl}°",
                transform=world_tr,
                fontsize=7.5,
                color="#334155",
                ha="center",
                va="bottom",
                alpha=0.85,
            )
    dec_label_ra = float((float(ext["ra_c"]) + 22.0) % 360.0)
    for dec_lbl in (-60, -30, 30, 60):
        ax.text(
            dec_label_ra,
            float(dec_lbl),
            f"{dec_lbl:+d}°",
            transform=world_tr,
            fontsize=7.5,
            color="#334155",
            ha="left",
            va="bottom",
            alpha=0.85,
        )

    ax.set_title(
        "All-Sky Celestial Context — SkyMaskPipe.plot_moc (Equatorial Mollweide)",
        pad=12,
    )
    ax.legend(loc="upper right", fontsize=8.2, framealpha=0.92)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax, wcs
