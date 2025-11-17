"""
Data Collection Package
"""

from .base_collector import BaseCollector
from .red_cross import RedCrossCollector
from .web_collector import (
    EnhancedWebCollector,
    MayoClinicCollector,
    ClevelandClinicCollector,
    HealthlineCollector,
    CDCEmergencyCollector,
    NHSCollector,
    StJohnCollector,
    WebMDCollector,
)
from .merge_pipeline import MasterDataPipeline
from .new_sources_collector import NewSourcesCollector
from .augmentation import ScenarioAugmentor

__all__ = [
    'BaseCollector',
    'RedCrossCollector',
    'EnhancedWebCollector',
    'MayoClinicCollector',
    'ClevelandClinicCollector',
    'HealthlineCollector',
    'CDCEmergencyCollector',
    'NHSCollector',
    'StJohnCollector',
    'WebMDCollector',
    'MasterDataPipeline',
    'NewSourcesCollector',
    'ScenarioAugmentor',
]

__version__ = '3.0.0'