"""Anomaly detection using Isolation Forest with graceful degradation."""
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

from src.config import Config

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Anomaly detection with graceful fallback."""
    
    def __init__(
        self,
        model_path: str = None,
        threshold: float = None,
        contamination: float = None,
    ):
        self.model_path = Path(model_path or Config.MODEL_PATH)
        self.threshold = threshold if threshold is not None else Config.ANOMALY_THRESHOLD
        self.contamination = contamination if contamination is not None else Config.CONTAMINATION_RATE
        self.model: Optional[IsolationForest] = self._load_model()
    
    def _load_model(self) -> Optional[IsolationForest]:
        """Load existing model or return None (fail-soft)."""
        try:
            if self.model_path.exists():
                logger.info(f"Loading anomaly model from {self.model_path}")
                return joblib.load(self.model_path)
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
        return None
    
    def train(self, df: pd.DataFrame) -> bool:
        """Train model on provided data. Returns True on success."""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                logger.warning("No numeric columns for training")
                return False
            
            X = df[numeric_cols].fillna(0).values
            if len(X) < 10:
                logger.warning("Insufficient data for training")
                return False
            
            self.model = IsolationForest(
                n_estimators=100,
                max_samples=min(256, len(X)),
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1,
            )
            self.model.fit(X)
            self._save_model()
            logger.info(f"Model trained on {len(X)} samples")
            return True
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def _save_model(self) -> None:
        """Save model atomically."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.model_path.with_suffix('.tmp')
        joblib.dump(self.model, temp_path)
        temp_path.replace(self.model_path)
    
    def detect(self, df: pd.DataFrame) -> dict:
        """Detect anomalies with graceful fallback."""
        if self.model is None or df.empty:
            return {
                "is_anomalous": False,
                "score": 0.0,
                "details": "Modelo não disponível" if self.model is None else "DataFrame vazio",
            }
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return {
                    "is_anomalous": False,
                    "score": 0.0,
                    "details": "Sem colunas numéricas para análise",
                }
            
            X = df[numeric_cols].fillna(0).values
            scores = self.model.decision_function(X)
            min_score = float(scores.min())
            
            # Identify anomalous rows
            anomalous_mask = scores < self.threshold
            anomalous_count = int(anomalous_mask.sum())
            
            details = f"Score mínimo: {min_score:.3f}"
            if anomalous_count > 0:
                details += f" | {anomalous_count} linha(s) anômala(s) detectada(s)"
            
            return {
                "is_anomalous": min_score < self.threshold,
                "score": min_score,
                "details": details,
            }
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return {
                "is_anomalous": False,
                "score": 0.0,
                "details": f"Erro na detecção: {str(e)}",
            }


# Cached detector instance
_anomaly_detector: Optional[AnomalyDetector] = None


def get_anomaly_detector() -> AnomalyDetector:
    """Get or create the anomaly detector (singleton)."""
    global _anomaly_detector
    if _anomaly_detector is None:
        _anomaly_detector = AnomalyDetector()
    return _anomaly_detector
