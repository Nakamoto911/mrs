import pytest
import numpy as np
import pandas as pd
from src.models.lstm_validation import (
    LSTMSequenceValidator,
    SequenceRequirements,
    SequenceValidationResult
)


def test_theoretical_sequence_count():
    """Test theoretical sequence counting."""
    req = SequenceRequirements(seq_length=12, stride=1)
    validator = LSTMSequenceValidator(req)
    
    # 100 samples, seq_len=12, stride=1 -> (100-12)//1 + 1 = 89 sequences
    assert validator.count_theoretical_sequences(100) == 89
    
    # With stride=6 -> (100-12)//6 + 1 = 15 sequences
    req.stride = 6
    validator = LSTMSequenceValidator(req)
    # (100-12)//6 + 1 = 88//6 + 1 = 14 + 1 = 15
    assert validator.count_theoretical_sequences(100) == 15


def test_nan_reduces_sequence_count():
    """Test that NaN values reduce actual sequence count."""
    req = SequenceRequirements(seq_length=12, stride=1)
    validator = LSTMSequenceValidator(req)
    
    dates = pd.date_range('2000-01-01', periods=100, freq='ME')
    X = pd.DataFrame({'a': np.random.randn(100)}, index=dates)
    y = pd.Series(np.random.randn(100), index=dates)
    
    # No NaN
    _, n_after_nan = validator.count_actual_sequences(X, y)
    assert n_after_nan == 89
    
    # Add NaN to middle of data
    X_with_nan = X.copy()
    X_with_nan.iloc[50:55, 0] = np.nan
    
    # Sequences affected: those that overlap with index 50-54
    # A sequence [i, i+11] overlaps with 50-54 if:
    # i <= 54 and i+11 >= 50
    # i <= 54 and i >= 39
    # So i in [39, 54] (16 indices)
    _, n_after_nan_2 = validator.count_actual_sequences(X_with_nan, y)
    assert n_after_nan_2 == 89 - 16


def test_validation_fails_insufficient_sequences():
    """Test validation fails when sequences insufficient."""
    req = SequenceRequirements(
        seq_length=12,
        min_total_sequences=100  # Too high for 50 samples
    )
    validator = LSTMSequenceValidator(req)
    
    dates = pd.date_range('2000-01-01', periods=50, freq='ME')
    X = pd.DataFrame({'a': np.random.randn(50)}, index=dates)
    y = pd.Series(np.random.randn(50), index=dates)
    
    result = validator.validate(X, y)
    
    assert not result.is_valid
    assert 'Total sequences' in result.failure_reason


def test_validation_passes_sufficient_sequences():
    """Test validation passes when sequences sufficient."""
    req = SequenceRequirements(
        seq_length=12,
        min_total_sequences=30,
        min_train_sequences=20,
        min_val_sequences=5
    )
    validator = LSTMSequenceValidator(req)
    
    dates = pd.date_range('2000-01-01', periods=200, freq='ME')
    X = pd.DataFrame({'a': np.random.randn(200)}, index=dates)
    y = pd.Series(np.random.randn(200), index=dates)
    
    result = validator.validate(X, y)
    
    assert result.is_valid
    assert result.failure_reason is None


def test_per_fold_validation():
    """Test per-fold validation catches fold-specific issues."""
    req = SequenceRequirements(
        seq_length=12,
        min_train_sequences=50,  # High requirement
        min_val_sequences=5
    )
    validator = LSTMSequenceValidator(req)
    
    dates = pd.date_range('2000-01-01', periods=100, freq='ME')
    X = pd.DataFrame({'a': np.random.randn(100)}, index=dates)
    y = pd.Series(np.random.randn(100), index=dates)
    
    # Create fold with insufficient training data
    cv_folds = [
        (np.arange(20), np.arange(20, 60)),  # Only 20 train samples -> approx 9 sequences
        (np.arange(80), np.arange(80, 100)),  # 80 train samples -> approx 69 sequences
    ]
    
    fold_results = validator.validate_per_fold(X, y, cv_folds)
    
    assert not fold_results[0]['is_valid']  # First fold fails (approx 9 < 50)
    assert fold_results[1]['is_valid']      # Second fold passes (approx 69 > 50, approx 9 > 5)
