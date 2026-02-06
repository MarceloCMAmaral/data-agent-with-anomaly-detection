"""Model lifecycle management: training, persistence, retraining decisions."""
import logging
from datetime import datetime

from src.config import Config
from src.ml.anomaly_detector import get_anomaly_detector
from src.ml.data_buffer import get_data_buffer

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model lifecycle and retraining decisions."""
    
    def __init__(self, retrain_interval: int = None):
        self.retrain_interval = retrain_interval or Config.RETRAIN_INTERVAL
        self._query_count = 0
        self._last_retrain = datetime.now()
    
    def should_retrain(self) -> bool:
        """Check if model should be retrained based on criteria."""
        hours_since_retrain = (datetime.now() - self._last_retrain).total_seconds() / 3600
        return (
            self._query_count >= self.retrain_interval or
            hours_since_retrain >= 24
        )
    
    def increment_query_count(self) -> None:
        """Increment query counter after successful detection."""
        self._query_count += 1
    
    def retrain_if_needed(self) -> bool:
        """Retrain model if criteria are met. Returns True if retrained."""
        if not self.should_retrain():
            return False
        
        buffer = get_data_buffer()
        if buffer.size() < 100:
            logger.info("Buffer too small for retraining")
            return False
        
        detector = get_anomaly_detector()
        training_data = buffer.get_training_data()
        
        if detector.train(training_data):
            self._query_count = 0
            self._last_retrain = datetime.now()
            logger.info("Model retrained successfully")
            return True
        
        return False


# Cached manager instance
_manager: ModelManager = None


def get_model_manager() -> ModelManager:
    """Get or create the model manager (singleton)."""
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager
