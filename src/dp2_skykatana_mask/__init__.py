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

__version__ = "0.2.0"

__all__ = [
    "AB_MAG_ZEROPOINT_NJY",
    "BandCriteria",
    "DEFAULT_RUBIN_RX_RV31",
    "DP2MaskBuilder",
    "DustExtinctionConfig",
    "DustMaskStats",
    "EBVProvider",
    "ExtraMapCondition",
    "GalacticMaskConfig",
    "MaskBuildSummary",
    "MaskPipelineConfig",
    "MemoryConfig",
    "MonsterStarConfig",
    "OutputConfig",
    "PropertyMaskStats",
    "PropertyMapsConfig",
    "RegionSelector",
    "build_dp2_property_mask",
    "build_dust_extinction_mask",
    "build_monster_hats_star_mask_online",
    "build_monster_star_mask",
    "compute_band_extinction_mag",
    "deredden_magnitude",
    "flux_njy_to_abmag",
    "get_rubin_band_extinction_rx",
    "load_config",
    "make_skykatana_radfunction",
    "skykatana_lsst_radfunction",
]
