"""Typed YAML configuration schema for DP2 SkyKatana mask construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

VALID_LSST_BANDS = ("u", "g", "r", "i", "z", "y")
VALID_COMPARISONS = ("gt", "ge", "lt", "le")

# LSSTCam nominal plate scale and Gaussian FWHM conversion factor:
# sp_maps `psf_size` is determinant radius sigma_det in pixels (~0.2 arcsec/pix).
LSSTCAM_PIXEL_SCALE_ARCSEC = 0.2
SIGMA_TO_FWHM = 2.3548200450309493


@dataclass
class RegionSelector:
    """Optional spatial subset for fast pilot runs (e.g. ECDFS) before full DP2."""

    mode: str = "full"  # "full", "cone", or "box"
    ra_deg: float = 53.1222
    dec_deg: float = -28.0870
    radius_deg: float = 2.0
    ra_min_deg: float = 0.0
    ra_max_deg: float = 360.0
    dec_min_deg: float = -90.0
    dec_max_deg: float = 90.0

    def validate(self) -> None:
        if self.mode not in ("full", "cone", "box"):
            raise ValueError(f"region.mode must be 'full', 'cone', or 'box', got {self.mode!r}")
        if self.mode == "cone" and self.radius_deg <= 0:
            raise ValueError("region.radius_deg must be > 0 when mode='cone'")
        if self.mode == "box" and self.dec_min_deg >= self.dec_max_deg:
            raise ValueError("region.dec_min_deg must be < dec_max_deg when mode='box'")


@dataclass
class ExtraMapCondition:
    """Threshold condition on any additional DP2 consolidated HealSparse map."""

    dataset_type: str
    comparison: str  # "gt", "ge", "lt", "le"
    threshold: float

    def validate(self) -> None:
        if self.comparison not in VALID_COMPARISONS:
            raise ValueError(
                f"Invalid comparison {self.comparison!r} for {self.dataset_type}; "
                f"expected one of {VALID_COMPARISONS}"
            )


@dataclass
class BandCriteria:
    """Per-pixel survey property map cuts for a single LSST band."""

    min_exposures: float | None = None
    nominal_exptime_sec: float = 30.0
    # Subtract a tiny sub-pixel tolerance (e.g. 0.05 exposures) if desired when degraded maps
    # average across visit borders; default 0.0 enforces strict >= min_exposures * nominal_exptime_sec.
    exposure_Subpixel_tol: float = 0.0
    min_exptime_sec: float | None = None
    min_maglim: float | None = None
    max_maglim: float | None = None
    max_psf_fwhm_arcsec: float | None = None
    max_psf_size_pixels: float | None = None
    max_sky_noise_njy: float | None = None
    max_sky_background_njy: float | None = None
    extra_conditions: list[ExtraMapCondition] = field(default_factory=list)

    def effective_min_exptime_sec(self) -> float | None:
        """Resolve minimal accumulated exposure time (seconds) for this band."""
        thresholds: list[float] = []
        if self.min_exptime_sec is not None:
            thresholds.append(float(self.min_exptime_sec))
        if self.min_exposures is not None:
            eff_n = max(0.0, float(self.min_exposures) - float(self.exposure_Subpixel_tol))
            thresholds.append(eff_n * float(self.nominal_exptime_sec))
        return max(thresholds) if thresholds else None

    def effective_max_psf_pixels(self) -> float | None:
        """Resolve maximal PSF determinant radius in LSSTCam pixels."""
        limits: list[float] = []
        if self.max_psf_size_pixels is not None:
            limits.append(float(self.max_psf_size_pixels))
        if self.max_psf_fwhm_arcsec is not None:
            sigma_arcsec = float(self.max_psf_fwhm_arcsec) / SIGMA_TO_FWHM
            limits.append(sigma_arcsec / LSSTCAM_PIXEL_SCALE_ARCSEC)
        return min(limits) if limits else None

    def validate(self, band: str) -> None:
        if self.nominal_exptime_sec <= 0:
            raise ValueError(f"Band {band}: nominal_exptime_sec must be > 0")
        for cond in self.extra_conditions:
            cond.validate()


@dataclass
class PropertyMapsConfig:
    """Configuration for Stage 1: DP2 Survey Property Maps (`sp_maps`)."""

    enabled: bool = True
    skymap: str = "lsst_cells_v2"
    order_sparse: int = 14  # NSIDE=16384 (~12.9"); set 15 for native NSIDE=32768 (~6.4")
    order_cov: int = 5  # NSIDE=32 matches DP2 sp_maps natively
    bands: list[str] = field(default_factory=lambda: ["u", "g", "r", "i", "z", "y"])
    min_filters: int = 4  # Minimum number of bands passing per pixel (K-of-N)
    required_bands: list[str] = field(default_factory=list)  # Mandatory bands that MUST pass
    keep_per_band_stages: bool = False  # Keep prop_{band}_mask stages in SkyMaskPipe (uses extra RAM)
    exptime_dataset: str = "deepCoadd_exposure_time_consolidated_map_sum"
    maglim_dataset: str = "deepCoadd_psf_maglim_consolidated_map_weighted_mean"
    psf_size_dataset: str = "deepCoadd_psf_size_consolidated_map_weighted_mean"
    sky_noise_dataset: str = "deepCoadd_sky_noise_consolidated_map_weighted_mean"
    sky_bg_dataset: str = "deepCoadd_sky_background_consolidated_map_weighted_mean"
    per_band: dict[str, BandCriteria] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.enabled:
            return
        if not (4 <= self.order_cov <= 8):
            raise ValueError(f"order_cov must be in [4, 8], got {self.order_cov}")
        if not (self.order_cov < self.order_sparse <= 15):
            raise ValueError(
                f"order_sparse must be > order_cov ({self.order_cov}) and <= 15 (DP2 native), "
                f"got {self.order_sparse}"
            )
        if not self.bands:
            raise ValueError("property_maps.bands cannot be empty")
        for b in self.bands:
            if b not in VALID_LSST_BANDS:
                raise ValueError(f"Unknown LSST band {b!r}; expected one of {VALID_LSST_BANDS}")
        for rb in self.required_bands:
            if rb not in self.bands:
                raise ValueError(
                    f"required_band {rb!r} must also appear in property_maps.bands ({self.bands})"
                )
        if not (1 <= self.min_filters <= len(self.bands)):
            raise ValueError(
                f"min_filters ({self.min_filters}) must be between 1 and len(bands) ({len(self.bands)})"
            )
        for b, crit in self.per_band.items():
            crit.validate(b)


@dataclass
class DustExtinctionConfig:
    """Milky Way foreground dust extinction correction & masking (DP2 Tutorial 311.1).

    Uses ``rubin_sim.phot_utils.DustValues(r_v=...).r_x`` ($R_b = A_b / E(B-V)$) and
    SFD98 $E(B-V)$ maps to:
      1. De-redden per-pixel $5\\sigma$ ``psf_maglim`` values ($m_{\\mathrm{lim}}^{\\mathrm{dered}} =
         m_{\\mathrm{lim}}^{\\mathrm{obs}} - R_b E(B-V)$) when ``apply_to_maglim=True``, and
      2. Build a negative ``dustmask`` stage excluding pixels with $E(B-V) > \\text{max\\_ebv}$
         (or $|b| < \\text{min\\_galactic\\_lat\\_deg}$).
    """

    enabled: bool = True
    apply_to_maglim: bool = True
    r_v: float = 3.1
    r_x_overrides: dict[str, float] = field(default_factory=dict)
    # "auto", "butler", "rubin_sim_sfd", "healsparse_file", "healpix_file",
    # "dustmaps_sfd", "tap_object", or "analytic_galactic"
    ebv_source: str = "auto"
    ebv_dataset: str = "ebv_sfd98_consolidated_map"
    ebv_map_path: str | None = None
    # Nominal extragalactic E(B-V) ceiling (e.g. 0.20 general, 0.12 WFD nominal, 0.08 strict DESC)
    max_ebv: float | None = 0.20
    # Optional SFD98 reliability guard for low Galactic latitudes (Tutorial 311.1 warns |b| < 5 deg)
    min_galactic_lat_deg: float | None = None
    order_sparse: int = 11  # NSIDE=2048 (~1.7 arcmin, finer than SFD98 6.1 arcmin FWHM)

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.r_v <= 0:
            raise ValueError(f"dust.r_v must be > 0, got {self.r_v}")
        valid_sources = (
            "auto",
            "butler",
            "rubin_sim_sfd",
            "rubin_scheduler",
            "healsparse_file",
            "healpix_file",
            "npz_file",
            "dustmaps_sfd",
            "tap_object",
            "analytic_galactic",
        )
        if self.ebv_source not in valid_sources:
            raise ValueError(
                f"dust.ebv_source must be one of {valid_sources}, got {self.ebv_source!r}"
            )
        if self.max_ebv is not None and self.max_ebv <= 0:
            raise ValueError(f"dust.max_ebv must be > 0 when set, got {self.max_ebv}")
        if self.min_galactic_lat_deg is not None and not (0.0 <= self.min_galactic_lat_deg <= 90.0):
            raise ValueError("dust.min_galactic_lat_deg must lie in [0, 90]")
        if not (5 <= self.order_sparse <= 15):
            raise ValueError(f"dust.order_sparse must lie in [5, 15], got {self.order_sparse}")


@dataclass
class MonsterStarConfig:
    """Configuration for Stage 2: Bright-star masking from `the_monster` catalog."""

    enabled: bool = True
    # "auto" discovers the latest `the_monster_*` dataset in Butler (e.g. `the_monster_20250219`)
    dataset_type: str = "auto"
    collections: list[str] | None = None
    order_sparse: int = 15  # NSIDE=32768 (~6.4") for crisp star halos
    # Band(s) used to compute synthetic LSST AB magnitude from `monster_*_{band}_flux` (nJy).
    # Note: Bright-star diffraction/ghost halos depend on *observed* apparent flux on LSSTCam
    # (not dust-dereddened intrinsic stellar luminosity).
    reference_bands: list[str] = field(default_factory=lambda: ["g", "r", "i", "z"])
    # Optional explicit flux column override (e.g. "monster_ComCam_r_flux" or "phot_g_mean_flux")
    explicit_flux_column: str | None = None
    mag_min: float = -5.0
    mag_max: float = 18.0  # Faintest Monster LSST star magnitude to mask

    # Radius scaling model:
    # - "skykatana_rubin": Native SkyKatana `radfunction(df, **kwargs)` empirical Rubin profile
    #                      fits as a function of LSST magnitude (`profile_band`, `completeness`)
    # - "exponential":     Custom R(m) = A_arcsec * exp(B_per_mag * m)
    # - "powerlaw10":      Custom R(m) = r0_arcsec * 10**(slope_dex_per_mag * (m0 - m))
    # - "csv_fit":         Load A_arcsec, B_per_mag from `radius_magnitude_fit_coefficients.csv`
    radius_model: str = "skykatana_rubin"
    # SkyKatana Rubin profile band: "u", "g", "r", "i", "z", "y", "r_halo" (3-piece wing+core),
    # or "multi" (envelope across `reference_bands`)
    profile_band: str = "r"
    completeness: float = 0.85  # SkyKatana completeness target (e.g. 0.80, 0.85, 0.90)
    use_faint_floor: bool = False  # Apply SkyKatana faint-star plateau above knee magnitude

    A_arcsec: float = 1068.426
    B_per_mag: float = -0.2398
    mag_knee: float | None = 18.0
    floor_arcsec: float = 8.0
    r0_arcsec: float = 150.0
    m0: float = 10.0
    slope_dex_per_mag: float = 0.16
    fit_coefficients_csv: str | None = None
    fit_behavior_branch: str = "deficit"  # "deficit" or "exceso" when reading CSV

    min_radius_arcsec: float = 5.0
    max_radius_arcsec: float = 600.0  # Upper cap (also sets HTM7 search buffer around footprint!)

    # Optional bleed-trail rectangular box masks for ultra-bright saturated stars
    enable_bleed_boxes: bool = False
    bleed_mag_max: float = 11.5
    bleed_width_factor: float = 0.25   # Box RA/X width in units of circle radius
    bleed_height_factor: float = 2.5   # Box Dec/Y length in units of circle radius

    save_star_circles_parquet: bool = True

    def validate(self) -> None:
        if not self.enabled:
            return
        if not (8 <= self.order_sparse <= 16):
            raise ValueError(f"monster_stars.order_sparse must be in [8, 16], got {self.order_sparse}")
        valid_models = ("skykatana_rubin", "exponential", "powerlaw10", "csv_fit")
        if self.radius_model not in valid_models:
            raise ValueError(
                f"monster_stars.radius_model must be one of {valid_models}, "
                f"got {self.radius_model!r}"
            )
        valid_prof_bands = (*VALID_LSST_BANDS, "r_halo", "multi")
        if self.profile_band not in valid_prof_bands:
            raise ValueError(
                f"monster_stars.profile_band must be one of {valid_prof_bands}, "
                f"got {self.profile_band!r}"
            )
        if not (0.50 <= float(self.completeness) < 1.0):
            raise ValueError(
                f"monster_stars.completeness must lie in [0.50, 1.0), got {self.completeness}"
            )
        if self.radius_model == "csv_fit" and not self.fit_coefficients_csv:
            raise ValueError("fit_coefficients_csv must be provided when radius_model='csv_fit'")
        if self.max_radius_arcsec <= 0 or self.min_radius_arcsec < 0:
            raise ValueError("Star mask radii must be positive")
        if self.min_radius_arcsec > self.max_radius_arcsec:
            raise ValueError("min_radius_arcsec cannot exceed max_radius_arcsec")


@dataclass
class GalacticMaskConfig:
    """Optional Stage 3: Milky Way plane & bulge exclusion via `SkyMaskPipe.build_milkyway_mask`."""

    enabled: bool = False
    order_sparse: int = 10
    b0_deg: float = 15.0
    bulge_a_deg: float = 25.0
    bulge_b_deg: float = 20.0


@dataclass
class MemoryConfig:
    """Memory & concurrency controls tailored to 32 GiB RSP Nublado instances."""

    max_ram_gib: float = 32.0
    soft_warning_fraction: float = 0.75
    hard_abort_fraction: float = 0.90
    cov_batch_size: int = 64  # Number of nside=32 coverage pixels processed per Butler IO batch
    htm7_query_batch_size: int = 200  # HTM7 IDs per Butler registry query
    htm7_stream_flush_rows: int = 250_000  # Rows buffered before flushing star circles to Parquet
    circle_chunk_size: int = 100_000  # SkyKatana build_circ_mask chunk_size (lowered from 600k for 32GB!)
    skykatana_threads: int = 2  # Threads for circle/box pixelization inside 32GB container


@dataclass
class OutputConfig:
    """Output products written by the pipeline."""

    output_dir: str = "outputs/dp2_mask"
    overwrite: bool = True
    order_out: int = 15  # Final combined mask sparse order (14 -> 12.9", 15 -> 6.4")
    order_cov: int = 5   # Final combined mask coverage order (5 -> NSIDE=32)
    write_skykatana_pipe: bool = True  # Native SkyMaskPipe directory (FITS stages + metadata.json)
    write_healsparse_fits: bool = True  # Standalone bit-packed HealSparse `.hs` file for direct use
    write_moc_fits: bool = True
    moc_max_order: int | None = 13  # Cap MOC export order if desired for fast visualization
    write_fracdet_map: bool = True
    fracdet_order: int = 10  # NSIDE=1024 fractional area map via SkyMaskPipe.frac_area_map
    n_random_points: int = 200_000  # 0 disables random catalog generation via SkyMaskPipe.makerans
    random_seed: int = 20261111
    make_qa_plots: bool = True


@dataclass
class MaskPipelineConfig:
    """Top-level configuration for DP2 SkyKatana mask creation."""

    butler_repo: str = "dp2"
    butler_collections: list[str] = field(default_factory=lambda: ["dp2"])
    butler_Repository_index: str | None = None
    access_token_file: str | None = None  # Only needed when using RemoteButler outside RSP
    region: RegionSelector = field(default_factory=RegionSelector)
    property_maps: PropertyMapsConfig = field(default_factory=PropertyMapsConfig)
    dust: DustExtinctionConfig = field(default_factory=DustExtinctionConfig)
    monster_stars: MonsterStarConfig = field(default_factory=MonsterStarConfig)
    galactic_mask: GalacticMaskConfig = field(default_factory=GalacticMaskConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        self.region.validate()
        self.property_maps.validate()
        self.dust.validate()
        self.monster_stars.validate()
        if not self.property_maps.enabled:
            raise ValueError("property_maps.enabled must be True to define a positive survey footprint")
        if not (self.output.order_cov < self.output.order_out <= 16):
            raise ValueError(
                f"output.order_out ({self.output.order_out}) must be > output.order_cov "
                f"({self.output.order_cov}) and <= 16"
            )


def _parse_band_criteria(raw: dict[str, Any]) -> BandCriteria:
    data = dict(raw)
    extra_raw = data.pop("extra_conditions", []) or []
    extras = [ExtraMapCondition(**item) for item in extra_raw]
    # Support case-insensitive alias `exposure_subpixel_tol`
    if "exposure_subpixel_tol" in data:
        data["exposure_Subpixel_tol"] = data.pop("exposure_subpixel_tol")
    return BandCriteria(**data, extra_conditions=extras)


def load_config(path: str | Path) -> MaskPipelineConfig:
    """Load and validate a ``MaskPipelineConfig`` from a YAML file."""
    cfg_path = Path(path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    region = RegionSelector(**(raw.get("region") or {}))

    pm_raw = dict(raw.get("property_maps") or {})
    per_band_raw = pm_raw.pop("per_band", {}) or {}
    per_band = {str(b): _parse_band_criteria(crit or {}) for b, crit in per_band_raw.items()}
    property_maps = PropertyMapsConfig(**pm_raw, per_band=per_band)

    dust = DustExtinctionConfig(**(raw.get("dust") or {}))
    monster_stars = MonsterStarConfig(**(raw.get("monster_stars") or {}))
    galactic_mask = GalacticMaskConfig(**(raw.get("galactic_mask") or {}))
    memory = MemoryConfig(**(raw.get("memory") or {}))
    output = OutputConfig(**(raw.get("output") or {}))

    butler_raw = raw.get("butler") or {}
    repo = butler_raw.get("repo", raw.get("butler_repo", "dp2"))
    collections = butler_raw.get("collections", raw.get("butler_collections", ["dp2"]))
    if isinstance(collections, str):
        collections = [collections]
    repo_index = butler_raw.get("repository_index", raw.get("butler_Repository_index"))
    token_file = butler_raw.get("access_token_file", raw.get("access_token_file"))

    cfg = MaskPipelineConfig(
        butler_repo=repo,
        butler_collections=list(collections),
        butler_Repository_index=repo_index,
        access_token_file=token_file,
        region=region,
        property_maps=property_maps,
        dust=dust,
        monster_stars=monster_stars,
        galactic_mask=galactic_mask,
        memory=memory,
        output=output,
    )
    cfg.validate()
    return cfg
