import pandas as pd
import numpy as np
from src.models.lstm_validation import LSTMSequenceValidator, SequenceRequirements

def test_logic():
    print("Testing LSTM logic...")
    req = SequenceRequirements(seq_length=12, stride=1, min_total_sequences=50)
    validator = LSTMSequenceValidator(req)
    
    # 100 samples -> 89 sequences
    n_theo = validator.count_theoretical_sequences(100)
    assert n_theo == 89
    print(f"✓ Theoretical: {n_theo}")
    
    # With NaNs
    dates = pd.date_range('2000-01-01', periods=100, freq='ME')
    X = pd.DataFrame({'a': np.arange(100)}, index=dates)
    y = pd.Series(np.arange(100), index=dates)
    
    X.iloc[50, 0] = np.nan
    # Sequence [i, i+11] is invalid if i <= 50 <= i+11
    # i <= 50 and i >= 39
    # i in [39, 50] (12 indices)
    _, n_after = validator.count_actual_sequences(X, y)
    assert n_after == 89 - 12
    print(f"✓ After NaN: {n_after}")
    
    # Validation
    res = validator.validate(X, y)
    assert res.is_valid == True
    print(f"✓ Overall validation: {res.is_valid}")
    
    # Fail validation
    req_fail = SequenceRequirements(min_total_sequences=100)
    val_fail = LSTMSequenceValidator(req_fail)
    res_fail = val_fail.validate(X, y)
    assert res_fail.is_valid == False
    print(f"✓ Failure detection: {res_fail.is_valid} (Reason: {res_fail.failure_reason})")

if __name__ == "__main__":
    try:
        test_logic()
        print("\nAll LSTM Gating tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
