"""
Neural Networks Module
======================
MLP and LSTM implementations for macro regime detection.

Part of the Asset-Specific Macro Regime Detection System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MLPWrapper:
    """Multi-Layer Perceptron wrapper."""
    
    def __init__(self, 
                 hidden_layers: List[int] = [64, 32],
                 dropout: float = 0.2,
                 learning_rate: float = 0.001,
                 epochs: int = 100,
                 batch_size: int = 32,
                 early_stopping: bool = True,
                 patience: int = 10):
        """
        Initialize MLP.
        
        Args:
            hidden_layers: List of hidden layer sizes
            dropout: Dropout rate
            learning_rate: Learning rate
            epochs: Maximum epochs
            batch_size: Batch size
            early_stopping: Whether to use early stopping
            patience: Early stopping patience
        """
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.patience = patience
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.history = None
    
    def _create_model(self, n_features: int):
        """Create MLP model."""
        try:
            import torch
            import torch.nn as nn
            
            layers = []
            prev_size = n_features
            
            for size in self.hidden_layers:
                layers.append(nn.Linear(prev_size, size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
                prev_size = size
            
            layers.append(nn.Linear(prev_size, 1))
            
            return nn.Sequential(*layers)
        
        except ImportError:
            # Fallback to sklearn
            from sklearn.neural_network import MLPRegressor
            return MLPRegressor(
                hidden_layer_sizes=tuple(self.hidden_layers),
                learning_rate_init=self.learning_rate,
                max_iter=self.epochs,
                early_stopping=self.early_stopping,
                random_state=42
            )
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MLPWrapper':
        """Fit the model."""
        from sklearn.preprocessing import StandardScaler
        
        self.feature_names = X.columns.tolist()
        
        # Handle missing values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X.loc[mask]
        y_clean = y.loc[mask]
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Create and fit model
        self.model = self._create_model(X_scaled.shape[1])
        
        if hasattr(self.model, 'fit'):
            # sklearn interface
            self.model.fit(X_scaled, y_clean)
        else:
            # PyTorch training would go here
            pass
        
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions."""
        if self.model is None:
            raise ValueError("Model not fitted")
        
        X_scaled = self.scaler.transform(X.fillna(0))
        predictions = self.model.predict(X_scaled)
        
        return pd.Series(predictions.flatten(), index=X.index)


class LSTMWrapper:
    """LSTM wrapper for sequence modeling."""
    
    def __init__(self,
                 sequence_length: int = 24,
                 hidden_size: int = 64,
                 n_layers: int = 2,
                 dropout: float = 0.2,
                 learning_rate: float = 0.001,
                 epochs: int = 100,
                 batch_size: int = 32):
        """
        Initialize LSTM.
        
        Args:
            sequence_length: Input sequence length (months)
            hidden_size: LSTM hidden size
            n_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate
            epochs: Maximum epochs
            batch_size: Batch size
        """
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.model = None
        self.scaler = None
        self.feature_names = None
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM."""
        X_seq, y_seq = [], []
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LSTMWrapper':
        """Fit the model."""
        from sklearn.preprocessing import StandardScaler
        
        self.feature_names = X.columns.tolist()
        
        # Handle missing values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X.loc[mask]
        y_clean = y.loc[mask]
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_clean.values)
        
        if len(X_seq) < 20:
            raise ValueError("Insufficient data for LSTM training")
        
        # For now, use a simple sklearn fallback
        # Full PyTorch LSTM would be implemented here
        from sklearn.neural_network import MLPRegressor
        
        # Flatten sequences for MLP fallback
        X_flat = X_seq.reshape(len(X_seq), -1)
        
        self.model = MLPRegressor(
            hidden_layer_sizes=(self.hidden_size, self.hidden_size),
            max_iter=self.epochs,
            random_state=42
        )
        self.model.fit(X_flat, y_seq)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions."""
        if self.model is None:
            raise ValueError("Model not fitted")
        
        X_scaled = self.scaler.transform(X.fillna(0))
        
        # Create sequence for latest observation
        if len(X_scaled) < self.sequence_length:
            # Pad with zeros if insufficient history
            X_padded = np.zeros((self.sequence_length, X_scaled.shape[1]))
            X_padded[-len(X_scaled):] = X_scaled
            X_seq = X_padded.reshape(1, -1)
        else:
            X_seq = X_scaled[-self.sequence_length:].reshape(1, -1)
        
        prediction = self.model.predict(X_seq)
        
        return pd.Series([prediction[0]], index=[X.index[-1]])


def create_neural_model(model_type: str, **kwargs):
    """Factory function for neural network models."""
    if model_type == 'mlp':
        return MLPWrapper(**kwargs)
    elif model_type == 'lstm':
        return LSTMWrapper(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test
    np.random.seed(42)
    n = 200
    
    X = pd.DataFrame({
        'feature1': np.random.randn(n),
        'feature2': np.random.randn(n),
        'feature3': np.random.randn(n),
    })
    
    y = 0.5 * X['feature1'] + 0.3 * X['feature2'] + np.random.randn(n) * 0.1
    y = pd.Series(y, name='target')
    
    # Test MLP
    print("Testing MLP...")
    model = create_neural_model('mlp', hidden_layers=[32, 16], epochs=50)
    model.fit(X, y)
    preds = model.predict(X)
    print(f"Predictions shape: {preds.shape}")
