"""Sliding window buffer for training data accumulation."""
import logging
import threading
from pathlib import Path
import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)


class SlidingWindowBuffer:
    """Thread-safe sliding window buffer for training data."""
    
    def __init__(
        self,
        max_size: int = None,
        buffer_path: str = None,
    ):
        self.max_size = max_size or Config.BUFFER_SIZE
        self.buffer_path = Path(buffer_path or Config.BUFFER_PATH)
        self.buffer_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data: pd.DataFrame = self._load_buffer()
    
    @property
    def _buffer_file(self) -> Path:
        return self.buffer_path / "buffer.parquet"
    
    def _load_buffer(self) -> pd.DataFrame:
        """Load buffer from disk."""
        try:
            if self._buffer_file.exists():
                return pd.read_parquet(self._buffer_file)
        except Exception as e:
            logger.warning(f"Could not load buffer: {e}")
        return pd.DataFrame()
    
    def _save_buffer(self) -> None:
        """Persist buffer to disk."""
        try:
            self._data.to_parquet(self._buffer_file, index=False)
        except Exception as e:
            logger.error(f"Could not save buffer: {e}")
    
    def add(self, df: pd.DataFrame) -> None:
        """Add new data to buffer, evicting oldest if needed."""
        if df.empty:
            return
        
        with self._lock:
            # Only keep numeric columns for ML training
            numeric_df = df.select_dtypes(include=['number'])
            if numeric_df.empty:
                return
            
            self._data = pd.concat([self._data, numeric_df], ignore_index=True)
            if len(self._data) > self.max_size:
                self._data = self._data.tail(self.max_size)
            self._save_buffer()
            logger.debug(f"Buffer size: {len(self._data)}")
    
    def get_training_data(self) -> pd.DataFrame:
        """Get all buffered data for training."""
        with self._lock:
            return self._data.copy()
    
    def size(self) -> int:
        """Return current buffer size."""
        with self._lock:
            return len(self._data)


# Cached buffer instance
_buffer: SlidingWindowBuffer = None


def get_data_buffer() -> SlidingWindowBuffer:
    """Get or create the data buffer (singleton)."""
    global _buffer
    if _buffer is None:
        _buffer = SlidingWindowBuffer()
    return _buffer
