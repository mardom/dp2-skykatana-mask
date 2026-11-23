"""DP2 spatial footprint, Milky Way dust, and Monster bright-star masking with SkyKatana."""

from ._skykatana_compat import apply_skykatana_runtime_fixes

apply_skykatana_runtime_fixes()

from .config import (
    BandCriteria,
    DustExtinctionConfig,
    ExtraMapCondition,
    GalacticMaskConfig,
    MaskPipelineConfig,
    MemoryConfig,
    MonsterStarConfig,
    OutputConfig,
    PropertyMapsConfig,
    RegionSelector,
    load_config,
)
from .dust import (
    DEFAULT_RUBIN_RX_RV31,
    DustMaskStats,
    EBVProvider,
    build_dust_extinction_mask,
    compute_band_extinction_mag,
    deredden_magnitude,
    get_rubin_band_extinction_rx,
)
from .monster_stars import (
    AB_MAG_ZEROPOINT_NJY,
    build_monster_hats_star_mask_online,
    build_monster_star_mask,
    flux_njy_to_abmag,
    make_skykatana_radfunction,
    skykatana_lsst_radfunction,
)
from .pipeline import DP2MaskBuilder, MaskBuildSummary
from .property_mask import PropertyMaskStats, build_dp2_property_mask
from .tap_local import (
    DEFAULT_GAIA_TAP_URL,
    DEFAULT_RSP_TAP_URL,
    DP2LocalTapMaskBuilder,
    IVOATapClient,
    SyntheticDP2TapService,
    TapSchemaInfo,
    build_local_tap_star_mask,
    build_tap_ebv_provider,
    build_tap_property_mask,
    discover_dp2_tap_schema,
    gaia_to_synthetic_lsst_magnitudes,
    open_rsp_tap_client,
    parse_stcs_region_pixels,
    resolve_rsp_tap_token,
)
from .visualization import (
    compute_sky_extent,
    extract_zoom_subpipe,
    plot_complete_mask_allsky,
    plot_complete_mask_fracmap,
    plot_complete_mask_multiscale,
    select_dense_zoom_box,
)

__version__ = "0.2.0"

__all__ = [
    "AB_MAG_ZEROPOINT_NJY",
    "BandCriteria",
    "DEFAULT_GAIA_TAP_URL",
    "DEFAULT_RSP_TAP_URL",
    "DEFAULT_RUBIN_RX_RV31",
    "DP2LocalTapMaskBuilder",
    "DP2MaskBuilder",
    "DustExtinctionConfig",
    "DustMaskStats",
    "EBVProvider",
    "ExtraMapCondition",
    "GalacticMaskConfig",
    "IVOATapClient",
    "MaskBuildSummary",
    "MaskPipelineConfig",
    "MemoryConfig",
    "MonsterStarConfig",
    "OutputConfig",
    "PropertyMaskStats",
    "PropertyMapsConfig",
    "RegionSelector",
    "SyntheticDP2TapService",
    "TapSchemaInfo",
    "build_dp2_property_mask",
    "build_dust_extinction_mask",
    "build_local_tap_star_mask",
    "build_monster_hats_star_mask_online",
    "build_monster_star_mask",
    "build_tap_ebv_provider",
    "build_tap_property_mask",
    "compute_band_extinction_mag",
    "compute_sky_extent",
    "deredden_magnitude",
    "discover_dp2_tap_schema",
    "extract_zoom_subpipe",
    "flux_njy_to_abmag",
    "gaia_to_synthetic_lsst_magnitudes",
    "get_rubin_band_extinction_rx",
    "load_config",
    "make_skykatana_radfunction",
    "open_rsp_tap_client",
    "parse_stcs_region_pixels",
    "plot_complete_mask_allsky",
    "plot_complete_mask_fracmap",
    "plot_complete_mask_multiscale",
    "resolve_rsp_tap_token",
    "select_dense_zoom_box",
    "skykatana_lsst_radfunction",
]
