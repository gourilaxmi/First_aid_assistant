from .base_collector import BaseCollector
from .red_cross import RedCrossCollector
from .base_web_collector import WebData
from .clinic_collectors import (
    MayoClinicCollector,
    ClevelandClinicCollector,
    HealthlineCollector
)
from .health_authority_collectors import (
    CDCEmergencyCollector,
    NHSCollector,
    StJohnCollector,
    WebMDCollector
)
from .merge_pipeline import MasterDataPipeline
from .new_sources_collector import NewSourcesCollector
from .augmentation import ScenarioAugmentation

__all__ = [
    'BaseCollector',
    'RedCrossCollector',
    'WebData',
    'MayoClinicCollector',
    'ClevelandClinicCollector',
    'HealthlineCollector',
    'CDCEmergencyCollector',
    'NHSCollector',
    'StJohnCollector',
    'WebMDCollector',
    'MasterDataPipeline',
    'NewSourcesCollector',
    'ScenarioAugmentation',
]

__version__ = '3.0.0'