"""End-to-end DP2 SkyKatana spatial mask orchestration (`DP2MaskBuilder`).

Orchestrates:
  1. Stage 1 (`propmask`): Multi-band DP2 `sp_maps` filtering (exposure counts, `min_filters`,
     `required_bands`, 5-sigma PSF limiting magnitudes, PSF seeing, sky noise/background).
  2. Stage 2 (`starmask`, optional `boxmask`): Bright-star halo & bleed-trail masking streamed
     shard-by-shard from `the_monster` (`htm7`).
  3. Stage 3 (`mwmask`, optional): Milky Way plane & bulge exclusion.
  4. Stage 4 (`mask`): Streaming boolean combination (`positive - negative`) via
     `SkyMaskPipe.combine`, exporting native `SkyMaskPipe` FITS stages, standalone bit-packed
     HealSparse `.hs`, IVOA MOC `.fits`, fractional-detection `fracdet` map, uniform randoms
     (`makerans`), QA plots, and JSON build provenance.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from skykatana import SkyMaskPipe

from .butler_io import open_dp2_butler
from .config import MaskPipelineConfig
from .dust import (
    DustMaskStats,
    EBVProvider,
    build_dust_extinction_mask,
    get_rubin_band_extinction_rx,
)
from .memory import (
    MemoryGuard,
    configure_rsp_thread_env,
    estimate_working_set_gib,
)
from .monster_stars import MonsterStarMaskStats, build_monster_star_mask
from .property_mask import PropertyMaskStats, build_dp2_property_mask

logger = logging.getLogger(__name__)

FULL_SKY_AREA_DEG2 = 4.0 * math.pi * (180.0 / math.pi) ** 2


@dataclass
class MaskBuildSummary:
    """Provenance and area statistics produced by ``DP2MaskBuilder.run()``."""

    butler_repo: str
    butler_collections: list[str]
    region_mode: str
    order_out: int
    order_cov: int
    nside_out: int
    property_mask: PropertyMaskStats
    dust_mask: DustMaskStats | None
    monster_stars: MonsterStarMaskStats | None
    mwmask_area_deg2: float
    positive_stages: list[str]
    negative_stages: list[str]
    final_valid_pixels: int
    final_area_deg2: float
    footprint_retained_fraction: float
    full_sky_fraction: float
    peak_ram_gib: float
    memory_limit_gib: float
    memory_limit_source: str
    elapsed_seconds: float
    output_files: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def write_json(self, path: str | Path) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return out_path


class DP2MaskBuilder:
    """Memory-guarded builder for Rubin DP2 spatial footprints and bright-star masks."""

    def __init__(self, cfg: MaskPipelineConfig, butler: Any | None = None) -> None:
        cfg.validate()
        self.cfg = cfg
        self._butler = butler

        configure_rsp_thread_env(cfg.memory.skykatana_threads)
        self.guard = MemoryGuard(
            max_ram_gib=cfg.memory.max_ram_gib,
            soft_warning_fraction=cfg.memory.soft_warning_fraction,
            hard_abort_fraction=cfg.memory.hard_abort_fraction,
        )
        self.pipe = SkyMaskPipe()

    def estimate_memory(self, assumed_area_deg2: float = 6000.0) -> dict[str, float]:
        """Return analytic stage-by-stage RAM budget (GiB) before touching Butler."""
        return estimate_working_set_gib(
            area_deg2=assumed_area_deg2,
            order_prop=self.cfg.property_maps.order_sparse,
            order_star=self.cfg.monster_stars.order_sparse
            if self.cfg.monster_stars.enabled
            else self.cfg.property_maps.order_sparse,
            order_out=self.cfg.output.order_out,
            order_cov=self.cfg.output.order_cov,
            cov_batch_size=self.cfg.memory.cov_batch_size,
            circle_chunk_size=self.cfg.memory.circle_chunk_size,
            n_threads=self.cfg.memory.skykatana_threads,
        )

    def run(self) -> tuple[SkyMaskPipe, MaskBuildSummary]:
        """Execute all enabled masking stages, combine them, and export artifacts."""
        t0 = time.perf_counter()
        out_dir = Path(self.cfg.output.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        self.guard.check("pipeline start")

        butler = self._butler if self._butler is not None else open_dp2_butler(self.cfg)

        ebv_provider: EBVProvider | None = None
        rx_map: dict[str, float] | None = None
        if self.cfg.dust.enabled:
            ebv_provider = EBVProvider.from_config(self.cfg, butler=butler)
            rx_map = get_rubin_band_extinction_rx(
                r_v=self.cfg.dust.r_v,
                overrides=self.cfg.dust.r_x_overrides,
            )

        # Stage 1: DP2 Survey Property Maps (`sp_maps`, with optional Tutorial 311.1 dust dereddening)
        logger.info("=== Stage 1: Building DP2 survey property map mask (propmask) ===")
        propmask, prop_stats = build_dp2_property_mask(
            self.pipe,
            butler=butler,
            cfg=self.cfg,
            guard=self.guard,
            output_stage="propmask",
            ebv_provider=ebv_provider,
            rx_map=rx_map,
        )
        positive_stages = ["propmask"]
        negative_stages: list[str] = []

        # Stage 1b: Foreground Milky Way dust extinction ceiling (`dustmask`, Tutorial 311.1)
        dust_stats: DustMaskStats | None = None
        if self.cfg.dust.enabled and ebv_provider is not None:
            logger.info("=== Stage 1b: Evaluating Milky Way E(B-V) dust mask (dustmask) ===")
            dust_map, dust_stats = build_dust_extinction_mask(
                self.pipe,
                footprint_mask=propmask,
                cfg=self.cfg,
                ebv_provider=ebv_provider,
                guard=self.guard,
                rx_map=rx_map,
                output_stage="dustmask",
            )
            if dust_map is not None and dust_stats.dustmask_valid_pixels > 0:
                negative_stages.append("dustmask")

        # Stage 2: Bright stars from `the_monster`
        star_stats: MonsterStarMaskStats | None = None
        if self.cfg.monster_stars.enabled:
            logger.info("=== Stage 2: Building Monster bright-star mask (starmask) ===")
            star_stats = build_monster_star_mask(
                self.pipe,
                butler=butler,
                footprint_mask=propmask,
                cfg=self.cfg,
                guard=self.guard,
                work_dir=out_dir,
                star_stage="starmask",
                box_stage="boxmask",
            )
            if star_stats.starmask_valid_pixels > 0:
                negative_stages.append("starmask")
            if self.cfg.monster_stars.enable_bleed_boxes and star_stats.n_bleed_boxes > 0:
                negative_stages.append("boxmask")

        # Stage 3: Optional Milky Way plane + bulge exclusion
        mwmask_area_deg2 = 0.0
        if self.cfg.galactic_mask.enabled:
            logger.info("=== Stage 3: Building Milky Way plane & bulge mask (mwmask) ===")
            mw = self.pipe.build_milkyway_mask(
                order_sparse=int(self.cfg.galactic_mask.order_sparse),
                order_cov=int(self.cfg.output.order_cov),
                b0_deg=float(self.cfg.galactic_mask.b0_deg),
                bulge_a_deg=float(self.cfg.galactic_mask.bulge_a_deg),
                bulge_b_deg=float(self.cfg.galactic_mask.bulge_b_deg),
                bit_packed=True,
                output_stage="mwmask",
            )
            mwmask_area_deg2 = float(mw.get_valid_area(degrees=True))
            if int(mw.n_valid) > 0:
                negative_stages.append("mwmask")
            self.guard.check("mwmask finalized", collect_garbage=True)

        # Stage 4: Streaming multi-order boolean combination
        logger.info(
            "=== Stage 4: Combining positive=%s minus negative=%s -> order_out=%d ===",
            positive_stages,
            negative_stages,
            self.cfg.output.order_out,
        )
        combined_mask = self.pipe.combine(
            positive=positive_stages,
            negative=negative_stages if negative_stages else None,
            order_out=int(self.cfg.output.order_out),
            order_cov=int(self.cfg.output.order_cov),
            bit_packed=True,
            verbose=False,
            output_stage="mask",
        )
        gc.collect()
        self.guard.check("combined mask finalized")

        final_valid_pixels = int(combined_mask.n_valid)
        final_area_deg2 = float(combined_mask.get_valid_area(degrees=True))
        retained_frac = (
            final_area_deg2 / prop_stats.area_deg2 if prop_stats.area_deg2 > 0 else 0.0
        )

        # Stage 5: Export products
        output_files = self._export_outputs(out_dir)

        elapsed = time.perf_counter() - t0
        summary = MaskBuildSummary(
            butler_repo=self.cfg.butler_repo,
            butler_collections=list(self.cfg.butler_collections),
            region_mode=self.cfg.region.mode,
            order_out=int(self.cfg.output.order_out),
            order_cov=int(self.cfg.output.order_cov),
            nside_out=1 << int(self.cfg.output.order_out),
            property_mask=prop_stats,
            dust_mask=dust_stats,
            monster_stars=star_stats,
            mwmask_area_deg2=mwmask_area_deg2,
            positive_stages=positive_stages,
            negative_stages=negative_stages,
            final_valid_pixels=final_valid_pixels,
            final_area_deg2=final_area_deg2,
            footprint_retained_fraction=retained_frac,
            full_sky_fraction=final_area_deg2 / FULL_SKY_AREA_DEG2,
            peak_ram_gib=round(self.guard.peak_used_gib, 4),
            memory_limit_gib=round(self.guard.limit_gib, 2),
            memory_limit_source=self.guard.limit_source,
            elapsed_seconds=round(elapsed, 3),
            output_files=output_files,
        )

        if self.cfg.output.make_qa_plots:
            qa_path = out_dir / "dp2_mask_qa.png"
            _render_qa_plot(self.pipe, summary, qa_path)
            summary.output_files["qa_plot_png"] = str(qa_path)
            if self.pipe.mask is not None and int(self.pipe.mask.n_valid) > 0:
                import matplotlib.pyplot as plt

                from .visualization import plot_complete_mask_multiscale

                circ_p = (
                    summary.monster_stars.circles_parquet_path
                    if summary.monster_stars is not None
                    else None
                )
                box_p = out_dir / "monster_bleed_boxes.fits"
                ms_path = out_dir / "dp2_complete_mask_multiscale.png"
                fig_ms, _ = plot_complete_mask_multiscale(
                    self.pipe,
                    circles_path=circ_p,
                    boxes_path=box_p if box_p.is_file() else None,
                    show=False,
                    save_path=ms_path,
                )
                plt.close(fig_ms)
                summary.output_files["complete_mask_multiscale_png"] = str(ms_path)

        summary_path = summary.write_json(out_dir / "mask_build_summary.json")
        summary.output_files["summary_json"] = str(summary_path)

        logger.info(
            "Finished DP2 SkyKatana mask: valid_area=%.3f deg^2 (%.1f%% of propmask), "
            "peak_RAM=%.2f GiB, elapsed=%.1fs",
            summary.final_area_deg2,
            100.0 * summary.footprint_retained_fraction,
            summary.peak_ram_gib,
            summary.elapsed_seconds,
        )
        return self.pipe, summary

    def _export_outputs(self, out_dir: Path) -> dict[str, str]:
        out_cfg = self.cfg.output
        files: dict[str, str] = {}

        if out_cfg.write_skykatana_pipe:
            pipe_dir = out_dir / "skykatana_pipe"
            self.pipe.write(pipe_dir, overwrite=bool(out_cfg.overwrite))
            files["skykatana_pipe_dir"] = str(pipe_dir)

        if out_cfg.write_healsparse_fits and self.pipe.mask is not None:
            hs_path = out_dir / "dp2_combined_mask.hs"
            self.pipe.mask.write(str(hs_path), clobber=bool(out_cfg.overwrite))
            files["combined_healsparse_hs"] = str(hs_path)

        if out_cfg.write_moc_fits and self.pipe.mask is not None and int(self.pipe.mask.n_valid) > 0:
            eff_moc_order = (
                min(int(out_cfg.moc_max_order), int(out_cfg.order_out))
                if out_cfg.moc_max_order is not None
                else int(out_cfg.order_out)
            )
            moc_path = out_dir / "dp2_combined_mask_moc.fits"
            moc = SkyMaskPipe.moc_from_stage(
                self.pipe.mask,
                order_forced=eff_moc_order,
                progress=False,
            )
            moc.save(str(moc_path), format="fits", overwrite=bool(out_cfg.overwrite))
            files["moc_fits"] = str(moc_path)

        if out_cfg.write_fracdet_map and self.pipe.mask is not None and int(self.pipe.mask.n_valid) > 0:
            eff_frac_order = min(int(out_cfg.fracdet_order), int(out_cfg.order_out))
            frac_path = out_dir / f"dp2_fracdet_o{eff_frac_order}.hs"
            frac_map = SkyMaskPipe.frac_area_map(self.pipe.mask, order_frac=eff_frac_order)
            frac_map.write(str(frac_path), clobber=bool(out_cfg.overwrite))
            files["fracdet_healsparse_hs"] = str(frac_path)
            del frac_map

        if int(out_cfg.n_random_points) > 0 and self.pipe.mask is not None and int(self.pipe.mask.n_valid) > 0:
            rans_path = out_dir / "dp2_randoms.parquet"
            rng = np.random.RandomState(int(out_cfg.random_seed))
            rans_df = self.pipe.makerans(
                stage="mask",
                nr=int(out_cfg.n_random_points),
                file=None,
                rng=rng,
            )
            rans_df.to_parquet(rans_path, index=False)
            files["randoms_parquet"] = str(rans_path)
            del rans_df

        gc.collect()
        self.guard.check("outputs exported")
        return files


def _render_qa_plot(pipe: SkyMaskPipe, summary: MaskBuildSummary, out_path: Path) -> None:
    """Render a lightweight headless QA figure without materializing full-resolution pixel arrays."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    # Panel 1: Per-band & combined area bar chart
    ax0 = axes[0]
    labels: list[str] = []
    areas: list[float] = []
    colors: list[str] = []

    for band, b_area in summary.property_mask.per_band_area_deg2.items():
        labels.append(f"band {band}")
        areas.append(b_area)
        colors.append("#4C78A8")

    labels.append("propmask\n(multi-band)")
    areas.append(summary.property_mask.area_deg2)
    colors.append("#2B5C8F")

    if summary.dust_mask is not None and summary.dust_mask.dustmask_area_deg2 > 0:
        labels.append("dustmask\n(excluded)")
        areas.append(summary.dust_mask.dustmask_area_deg2)
        colors.append("#E45756")

    labels.append("final mask\n(-dust/stars/MW)")
    areas.append(summary.final_area_deg2)
    colors.append("#54A24B")

    bars = ax0.bar(labels, areas, color=colors, edgecolor="black", linewidth=0.6)
    ax0.set_ylabel("Valid Sky Area [deg$^2$]")
    dust_tag = " [dust-dereddened]" if summary.property_mask.dust_dereddened_maglim else ""
    ax0.set_title(
        f"DP2 Footprint Stage Waterfall (min_filters={summary.property_mask.min_filters}){dust_tag}"
    )
    ax0.tick_params(axis="x", rotation=28)
    for bar, val in zip(bars, areas):
        ax0.text(
            bar.get_x() + bar.get_width() * 0.5,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Panel 2: Wrap-aware uniform random sample over surviving mask (fast & bounded RAM)
    ax1 = axes[1]
    if pipe.mask is not None and int(pipe.mask.n_valid) > 0:
        from matplotlib.ticker import FuncFormatter

        from .visualization import compute_sky_extent

        ext = compute_sky_extent(pipe, "mask")
        rng = np.random.RandomState(42)
        sample_df = pipe.makerans(stage="mask", nr=min(25_000, max(3_000, int(pipe.mask.n_valid))), rng=rng)
        ra_vals = sample_df["ra"].to_numpy(dtype=np.float64) % 360.0
        if ext["wraps_zero"]:
            cut = float((ext["ra_start"] - 5.0) % 360.0)
            ra_vals = np.where(ra_vals >= cut, ra_vals - 360.0, ra_vals)
            ax1.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v % 360.0:.0f}"))
        ax1.scatter(
            ra_vals,
            sample_df["dec"],
            s=0.8,
            alpha=0.55,
            color="#0f766e",
            rasterized=True,
            label="Unmasked sky (makerans)",
        )
        ax1.invert_xaxis()
        ax1.set_xlabel("RA [deg]")
        ax1.set_ylabel("Dec [deg]")
        star_info = (
            f", Monster stars={summary.monster_stars.n_stars_selected:,}"
            if summary.monster_stars is not None
            else ""
        )
        ax1.set_title(f"Surviving DP2 Mask ({summary.final_area_deg2:,.2f} deg$^2${star_info})")
        ax1.grid(True, alpha=0.25, linestyle="--")
    else:
        ax1.text(0.5, 0.5, "Empty final mask", ha="center", va="center", transform=ax1.transAxes)

    fig.savefig(out_path, dpi=160)
    plt.close(fig)
