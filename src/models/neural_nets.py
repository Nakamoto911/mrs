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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from preprocessing import TimeSeriesScaler, PointInTimeImputer
from .lstm_v2 import LSTMWrapperV2

logger = logging.getLogger(__name__)


class MLPWrapper(BaseEstimator):
    """Multi-Layer Perceptron wrapper."""
    
    def __init__(self, 
                 hidden_layers: List[int] = [64, 32],
                 dropout: float = 0.2,
                 learning_rate: float = 0.001,
                 epochs: int = 100,
                 batch_size: int = 32,
                 early_stopping: bool = True,
                 patience: int = 10,
                 **kwargs):
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
        self.kwargs = kwargs
        
        self.model = None
        self.scaler = TimeSeriesScaler(method='standard')
        self.imputer = PointInTimeImputer(strategy='median')
        self.feature_names = None
        self.input_dim = None
        self.fitted_ = False
        
    def __sklearn_tags__(self):
        return super().__sklearn_tags__()
        
        # Training params
    
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
        
        # 1. Align features and target
        y_clean = y.dropna()
        common_idx = X.index.intersection(y_clean.index)
        X_sync = X.loc[common_idx]
        y_sync = y_clean.loc[common_idx]
        
        if len(y_sync) < 20:
             raise ValueError(f"Insufficient total samples ({len(y_sync)}) for alignment")

        # 2. Prune all-NaN features (critical for imputation)
        all_nan_features = X_sync.columns[X_sync.isna().all()].tolist()
        if all_nan_features:
            logger.debug(f"Removing {len(all_nan_features)} all-NaN features")
            
        self.feature_names = [c for c in X_sync.columns if c not in all_nan_features]
        X_filtered = X_sync[self.feature_names]
        
        # 3. Point-in-Time Imputation
        X_imputed = self.imputer.transform_expanding(X_filtered)
        
        # 4. Point-in-Time Scaling
        X_scaled = self.scaler.fit_transform_rolling(X_imputed)
        
        # Final safety check: ensure no NaNs and align
        final_mask = ~(X_scaled.isna().any(axis=1) | y_sync.isna())
        if final_mask.sum() < 20:
             raise ValueError(f"Insufficient valid data after scaling (N={final_mask.sum()})")
             
        X_final = X_scaled[final_mask]
        y_final = y_sync[final_mask]
        
        self.input_dim = X_final.shape[1]
        
        # Create and fit model
        self.model = self._create_model(self.input_dim)
        
        if hasattr(self.model, 'fit'):
            # sklearn interface
            self.model.fit(X_final, y_final)
        else:
            # Prepare tensors
            try:
                X_vals = X_final.values
                y_vals = y_final.values
                # Check for object dtype
                if X_vals.dtype == object:
                     logger.warning("X_final.values is object type! Converting to float32...")
                     X_vals = X_vals.astype(np.float32)
                if y_vals.dtype == object:
                     logger.warning("y_final.values is object type! Converting to float32...")
                     y_vals = y_vals.astype(np.float32)

                X_tensor = torch.FloatTensor(X_vals)
                y_tensor = torch.FloatTensor(y_vals).reshape(-1, 1)
            except Exception as e:
                logger.error(f"Tensor conversion failed: {e}")
                logger.error(f"X_final shape: {X_final.shape}, dtypes: {X_final.dtypes}")
                raise e
            
            # PyTorch training
            self.model.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            criterion = nn.MSELoss()
            
            # Simple batching
            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            for epoch in range(self.epochs):
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
        self.fitted_ = True
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions."""
        if self.model is None:
            raise ValueError("Model not fitted")
        
        # Ensure X has the same features as during training
        X_filtered = X[self.feature_names]
        
        # Impute
        X_imputed = self.imputer.transform(X_filtered)
        
        # Scale
        X_scaled = self.scaler.transform(X_imputed)
        
        if hasattr(self.model, 'predict'):
            # sklearn interface
            predictions = self.model.predict(X_scaled)
        else:
            # PyTorch inference
            self.model.eval()
            with torch.no_grad():
                # Fix: Handle DataFrame input and object types
                if hasattr(X_scaled, 'values'):
                    X_vals = X_scaled.values
                else:
                    X_vals = X_scaled
                
                if X_vals.dtype == object:
                    logger.warning("X_scaled in predict is object type! Converting to float32...")
                    X_vals = X_vals.astype(np.float32)
                    
                X_tensor = torch.FloatTensor(X_vals)
                predictions = self.model(X_tensor).numpy()
        
        return pd.Series(predictions.flatten(), index=X.index)


class LSTMModel(nn.Module):
    """
    Pure PyTorch LSTM Architecture.
    
    Args:
        input_size: Number of input features
        hidden_size: Number of hidden units
        num_layers: Number of LSTM layers
        dropout: Probability of dropout
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        
        # 1. LSTM Layer
        # batch_first=True ensures input is (Batch, Seq_Len, Features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        
        # 2. Dropout Layer (Post-LSTM regularization)
        self.dropout = nn.Dropout(dropout)
        
        # 3. Projection Head (Many-to-One)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_size)
        
        # LSTM output shape: (batch_size, seq_len, hidden_size)
        # We only care about the final hidden state output
        out, (h_n, c_n) = self.lstm(x)
        
        # Extract last time step
        last_step_out = out[:, -1, :] 
        
        # Apply dropout and project
        out = self.dropout(last_step_out)
        out = self.fc(out)
        return out


class LSTMWrapper(BaseEstimator):
    """
    LSTM wrapper for sequence modeling using PyTorch.
    
    Replaces the previous 'placebo' MLP implementation with a genuine
    Recurrent Neural Network process.
    """
    
    def __init__(self,
                 sequence_length: int = 24,
                 hidden_size: int = 64,
                 n_layers: int = 2,
                 dropout: float = 0.2,
                 learning_rate: float = 0.001,
                 epochs: int = 100,
                 batch_size: int = 32):
        """
        Initialize LSTM Wrapper.
        
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
        self.scaler = TimeSeriesScaler(method='standard')
        self.imputer = PointInTimeImputer(strategy='median')
        self.feature_names = None
        self.fitted_ = False
        
        # Auto-detect device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
    def __sklearn_tags__(self):
        return super().__sklearn_tags__()
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM."""
        X_seq, y_seq = [], []
        
        if len(X) < self.sequence_length:
            return np.array([]), np.array([])
            
        for i in range(len(X) - self.sequence_length + 1):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length - 1])
        
        return np.array(X_seq), np.array(y_seq)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LSTMWrapper':
        """Fit the PyTorch LSTM model."""
        
        # 1. Align features and target
        y_clean = y.dropna()
        common_idx = X.index.intersection(y_clean.index)
        X_sync = X.loc[common_idx]
        y_sync = y_clean.loc[common_idx]
        
        if len(y_sync) < self.sequence_length + 20:
             raise ValueError(f"Insufficient total samples ({len(y_sync)}) for LSTM training")
        
        # 2. Prune all-NaN features
        all_nan_features = X_sync.columns[X_sync.isna().all()].tolist()
        self.feature_names = [c for c in X_sync.columns if c not in all_nan_features]
        X_filtered = X_sync[self.feature_names]
        
        # 3. Point-in-Time Imputation
        X_imputed = self.imputer.transform_expanding(X_filtered)
        
        # 4. Point-in-Time Scaling
        X_scaled = self.scaler.fit_transform_rolling(X_imputed)
        
        # 5. Create Sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_sync.values)
        
        if len(X_seq) < 10:
             raise ValueError("Insufficient sequence data for training")
             
        # 6. Tensor Setup
        X_tensor = torch.FloatTensor(X_seq)
        y_tensor = torch.FloatTensor(y_seq).reshape(-1, 1)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 7. Model Initialization
        input_dim = X_scaled.shape[1]
        self.model = LSTMModel(
            input_size=input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            dropout=self.dropout
        )
        self.model.to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 8. Training Loop
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        self.fitted_ = True
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions using sliding window inference."""
        if self.model is None or not self.fitted_:
            raise ValueError("Model not fitted")
        
        # Ensure features match
        X_filtered = X[self.feature_names]
        
        # Impute
        X_imputed = self.imputer.transform(X_filtered)
        
        # Scale
        X_scaled = self.scaler.transform(X_imputed)
        
        # Sliding Window Generation
        # We need alignment with X.index.
        # Pad the beginning with zeros for the first (sequence_length - 1) indices.
        
        padding = np.zeros((self.sequence_length - 1, X_scaled.shape[1]))
        X_padded = np.vstack([padding, X_scaled])
        
        # Create sliding window views
        n_samples = len(X_scaled)
        X_windows = []
        
        for i in range(n_samples):
            # Window length is self.sequence_length
            X_windows.append(X_padded[i : i + self.sequence_length])
            
        X_batch = np.array(X_windows)
        
        # Batched Inference
        self.model.eval()
        
        try:
            return self._batched_inference(X_batch, X.index)
        except Exception as e:
            if self.device.type != 'cpu':
                logger.warning(f"Inference failed on {self.device}: {e}. Falling back to CPU.")
                original_device = self.device
                self.model.to('cpu')
                try:
                    results = self._batched_inference(X_batch, X.index, device=torch.device('cpu'))
                    return results
                finally:
                    self.model.to(original_device)
            else:
                raise e

    def _batched_inference(self, X_batch: np.ndarray, index: pd.Index, device: Optional[torch.device] = None) -> pd.Series:
        """Helper for batched inference on a specific device."""
        dev = device or self.device
        X_tensor = torch.FloatTensor(X_batch).to(dev)
        
        batch_predictions = []
        inference_batch_size = 1024
        
        with torch.no_grad():
            for i in range(0, len(X_tensor), inference_batch_size):
                batch = X_tensor[i : i + inference_batch_size]
                preds = self.model(batch)
                batch_predictions.append(preds.cpu().numpy())
                
        final_preds = np.concatenate(batch_predictions).flatten()
        return pd.Series(final_preds, index=index)


def create_neural_model(model_type: str, **kwargs):
    """Factory function for neural network models."""
    if model_type == 'mlp':
        return MLPWrapper(**kwargs)
    elif model_type == 'lstm':
        return LSTMWrapperV2(**kwargs)
    elif model_type == 'lstm_legacy':
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
