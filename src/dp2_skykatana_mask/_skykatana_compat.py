"""Runtime safety patches for upstream SkyKatana on NumPy 2.x and 32 GiB RSP containers.

Addresses three upstream issues in ``skykatana.skykatana``:
1. ``_write_stage_fits_bitpack`` builds its variable-length ``PB()`` FITS column via
   ``np.array(batch_packed, dtype=object)``. Whenever a stage (such as ``dustmask`` or
   ``boxmask`` on a single-pointing pilot) occupies 1 coverage tile — or multiple coverage
   tiles whose packed byte arrays happen to have identical length — ``np.array(..., dtype=object)``
   collapses the list into a **2D ``(N, M)`` array of Python ``int`` objects** instead of a
   **1D ``(N,)`` object array of ``uint8`` ndarrays**, causing ``fitsio``'s C extension to
   dereference a ``PyLongObject*`` as a ``PyArrayObject*`` and segfault (``SIGSEGV``, exit 139).
2. ``SkyMaskPipe.read`` defaults ``max_workers`` to ``os.cpu_count() // 2``. On 64-core RSP
   Kubernetes host nodes, spawning 32 concurrent ``fitsio`` reader threads allocates
   ``32 * 96 MB = 3 GiB`` of scratch buffers and can trigger non-reentrant ``cfitsio`` races.
   Defaulting ``max_workers=1`` when unspecified keeps stage loading deterministic and <100 MB.
3. ``SkyMaskPipe.plot_moc`` without ``clipra``/``clipdec`` calls ``stage.valid_pixels`` before
   computing ``order_display = self.get_plot_order(wcs)``. On a full 6,500 deg² order-15 DP2
   mask (~20 billion valid pixels), materializing ``stage.valid_pixels`` allocates ~160 GiB
   before immediately coarsening to display order 6–8. Streaming via ``SkyMaskPipe.moc_from_stage``
   produces the identical display MOC in <20 MB RAM.
"""

from __future__ import annotations

import functools
import os
from typing import Any

from astropy.coordinates import Angle
import astropy.units as u
import fitsio
import matplotlib.pyplot as plt
from mocpy import WCS
import numpy as np
import skykatana.skykatana as _sk_mod
from skykatana import SkyMaskPipe

_PATCHED = False


def _safe_write_stage_fits_bitpack(
    stage_path: str | os.PathLike[str],
    hspmap: Any,
    rows_per_batch: int = 32768,
) -> None:
    """Drop-in replacement for ``skykatana.skykatana._write_stage_fits_bitpack`` safe for 1-row/equal-length batches."""
    if not np.issubdtype(hspmap.dtype, np.bool_):
        raise TypeError(f"Expected boolean HealSparseMap; got dtype={hspmap.dtype!r}")

    nside_cov = int(hspmap.nside_coverage)
    nside_spa = int(hspmap.nside_sparse)
    if nside_spa % nside_cov != 0:
        raise ValueError("nside_sparse must be a multiple of nside_coverage (NESTED).")
    ratio = nside_spa // nside_cov
    nfine = ratio * ratio

    iter_fn = getattr(_sk_mod, "_iter_valid_by_covpix", None)

    def _iter_rows():
        if callable(iter_fn):
            yield from iter_fn(hspmap)
        else:
            vp = hspmap.valid_pixels
            if vp.size == 0:
                return
            covpix = vp // nfine
            order = np.argsort(covpix, kind="mergesort")
            vp = vp[order]
            covpix = covpix[order]
            i, n_tot = 0, vp.size
            while i < n_tot:
                j = i
                cv = int(covpix[i])
                while j < n_tot and covpix[j] == cv:
                    j += 1
                yield cv, vp[i:j]
                i = j

    has_any = False
    for _peek in _iter_rows():
        has_any = True
        break

    if not has_any:
        with fitsio.FITS(str(stage_path), mode="rw", clobber=True) as f:
            f.create_table_hdu(
                names=["COVPIX", "ENC", "PACKED"],
                formats=["K", "B", "PB()"],
            )
            f[-1].write_keys(
                {
                    "NSIDE_COV": nside_cov,
                    "NSIDE_SPA": nside_spa,
                    "DTYPE": "bool",
                    "ENCOD": "BITPACK",
                    "NFINE": int(nfine),
                    "BITORD": "L",
                }
            )
        return

    def _pack_offsets_to_bytes(off_sorted: np.ndarray) -> np.ndarray:
        if off_sorted.size == 0:
            return np.empty(0, dtype=np.uint8)
        width = int(off_sorted[-1]) + 1
        byte_len = (width + 7) // 8
        byte_idx = (off_sorted >> 3).astype(np.int64)
        bit_pos = (off_sorted & 7).astype(np.int64)
        contrib = (1 << bit_pos).astype(np.uint16)
        out_u16 = np.zeros(byte_len, dtype=np.uint16)
        np.add.at(out_u16, byte_idx, contrib)
        return np.ascontiguousarray(out_u16, dtype=np.uint8)

    with fitsio.FITS(str(stage_path), mode="rw", clobber=True) as f:
        f.create_table_hdu(
            names=["COVPIX", "ENC", "PACKED"],
            formats=["K", "B", "PB()"],
        )
        f[-1].write_keys(
            {
                "NSIDE_COV": nside_cov,
                "NSIDE_SPA": nside_spa,
                "DTYPE": "bool",
                "ENCOD": "BITPACK",
                "NFINE": int(nfine),
                "BITORD": "L",
            }
        )

        batch_cov: list[int] = []
        batch_enc: list[int] = []
        batch_packed: list[np.ndarray] = []

        def _flush() -> None:
            if not batch_cov:
                return
            n_rows = len(batch_packed)
            # Construct an explicit 1D object array of uint8 ndarrays so NumPy never
            # coerces single-row or equal-length batches into a 2D object array!
            packed_col = np.empty(n_rows, dtype=object)
            for idx, arr_u8 in enumerate(batch_packed):
                packed_col[idx] = arr_u8
            rec = {
                "COVPIX": np.asarray(batch_cov, dtype="i8"),
                "ENC": np.asarray(batch_enc, dtype="u1"),
                "PACKED": packed_col,
            }
            f[-1].append(rec)
            batch_cov.clear()
            batch_enc.clear()
            batch_packed.clear()

        in_batch = 0
        for covpix, fine_pix in _iter_rows():
            if fine_pix.size == 0:
                continue
            base = covpix * nfine
            off = (fine_pix - base).astype(np.int64)
            packed = _pack_offsets_to_bytes(off)
            batch_cov.append(int(covpix))
            batch_enc.append(1)
            batch_packed.append(packed)
            in_batch += 1
            if in_batch >= rows_per_batch:
                _flush()
                in_batch = 0

        _flush()


def apply_skykatana_runtime_fixes() -> None:
    """Idempotently install ``fitsio`` VL-array, RSP thread-cap, and streaming ``plot_moc`` fixes."""
    global _PATCHED
    if _PATCHED:
        return

    _sk_mod._write_stage_fits_bitpack = _safe_write_stage_fits_bitpack

    orig_read_func = SkyMaskPipe.__dict__.get("read")
    if isinstance(orig_read_func, classmethod):
        raw_read = orig_read_func.__func__

        @functools.wraps(raw_read)
        def _safe_read(cls: type[SkyMaskPipe], indir: Any, *args: Any, **kwargs: Any) -> SkyMaskPipe:
            if kwargs.get("max_workers") is None:
                kwargs["max_workers"] = 1
            return raw_read(cls, indir, *args, **kwargs)

        SkyMaskPipe.read = classmethod(_safe_read)  # type: ignore[method-assign]

    orig_plot_moc = getattr(SkyMaskPipe, "plot_moc", None)
    if callable(orig_plot_moc):

        @functools.wraps(orig_plot_moc)
        def _streaming_plot_moc(
            self: SkyMaskPipe,
            stage: Any,
            center: Any = None,
            fov: Any = None,
            clipra: tuple[float, float] | None = None,
            clipdec: tuple[float, float] | None = None,
            order_force: int | None = None,
            frame: str = "icrs",
            projection: str = "SIN",
            figsize: tuple[float, float] = (10.0, 5.0),
            color: str = "green",
            alpha: float = 0.2,
            linewidth: float = 1.0,
            label: str | None = None,
            ax: Any = None,
            wcs: Any = None,
            show: bool = False,
            stream_pars: dict[str, object] | None = None,
        ) -> tuple[Any, Any, Any]:
            if clipra is not None and clipdec is not None:
                return orig_plot_moc(
                    self,
                    stage,
                    center=center,
                    fov=fov,
                    clipra=clipra,
                    clipdec=clipdec,
                    order_force=order_force,
                    frame=frame,
                    projection=projection,
                    figsize=figsize,
                    color=color,
                    alpha=alpha,
                    linewidth=linewidth,
                    label=label,
                    ax=ax,
                    wcs=wcs,
                    show=show,
                    stream_pars=stream_pars,
                )

            stage_map = stage if hasattr(stage, "valid_pixels") else getattr(self, stage)
            if ax is None or wcs is None:
                if center is None or fov is None:
                    raise ValueError("When ax/wcs are not provided, you must pass center and fov.")
                fig = plt.figure(figsize=figsize)
                with WCS(
                    fig,
                    fov=fov,
                    center=center,
                    coordsys=frame,
                    projection=projection,
                    rotation=Angle(0, u.deg),
                ) as _wcs:
                    ax = fig.add_subplot(1, 1, 1, projection=_wcs)
                    lon_c = ax.coords[0]
                    lat_c = ax.coords[1]
                    lon_c.set_format_unit(u.deg, decimal=True, show_decimal_unit=True)
                    lat_c.set_format_unit(u.deg, decimal=True, show_decimal_unit=True)
                    if str(frame).lower().startswith("gal"):
                        ax.set_xlabel("l")
                        ax.set_ylabel("b")
                    else:
                        ax.set_xlabel("ra")
                        ax.set_ylabel("dec")
                    ax.grid(color="black", linestyle="dotted")
                    wcs = _wcs
            else:
                fig = ax.figure

            order = int(round(np.log2(stage_map.nside_sparse)))
            order_display = int(self.get_plot_order(wcs))
            if order_force is not None:
                target_order = min(int(order_force), order)
            else:
                target_order = min(order, order_display)

            if int(stage_map.n_valid) > 0:
                moc = SkyMaskPipe.moc_from_stage(
                    stage_map,
                    order_forced=target_order,
                    progress=False,
                )
                moc.fill(
                    ax=ax,
                    wcs=wcs,
                    alpha=alpha,
                    fill=True,
                    color=color,
                    zorder=1,
                    label=label,
                    optimize=False,
                )
                moc.border(
                    ax=ax,
                    wcs=wcs,
                    alpha=max(0.6, alpha),
                    color="k",
                    linewidth=linewidth,
                    zorder=2,
                )

            if show:
                plt.show()
            return fig, ax, wcs

        SkyMaskPipe.plot_moc = _streaming_plot_moc  # type: ignore[method-assign]

    _PATCHED = True
