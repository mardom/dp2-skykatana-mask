"""DP2 spatial footprint and bright-star masking pipeline powered by SkyKatana."""

from .config import (
    BandCriteria,
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
from .pipeline import DP2MaskBuilder, MaskBuildSummary

__version__ = "0.1.0"

__all__ = [
    "BandCriteria",
    "DP2MaskBuilder",
    "ExtraMapCondition",
    "GalacticMaskConfig",
    "MaskBuildSummary",
    "MaskPipelineConfig",
    "MemoryConfig",
    "MonsterStarConfig",
    "OutputConfig",
    "PropertyMapsConfig",
    "RegionSelector",
    "load_config",
]
