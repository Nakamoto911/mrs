# Cointegration Testing and Validation Implementation Plan

Implement a robust cointegration testing and validation framework to empirically validate theory-driven macro relationships.

## Bug Fixes & Robustness

### [MODIFY] [run_tournament.py](file:///Volumes/PRO-G40/Code/mrs/run_tournament.py)
Update `load_features` to dynamically identify all level variables needed from `experiment_config.yaml` to prevent "INSUFFICIENT" data errors during validation.

### [MODIFY] [cointegration.py](file:///Volumes/PRO-G40/Code/mrs/src/feature_engineering/cointegration.py)
Ensure `fit()` clears `results`, `validation_results`, and `vectors_` to prevent state leakage across CV folds.

### [MODIFY] [hierarchical_clustering.py](file:///Volumes/PRO-G40/Code/mrs/src/feature_engineering/hierarchical_clustering.py)
Ensure `fit()` clears `clusters` and `representatives` to force re-clustering on each fold's training data. Update `select_representative` to handle missing columns gracefully.

### [MODIFY] [cointegration_validator.py](file:///Volumes/PRO-G40/Code/mrs/src/feature_engineering/cointegration_validator.py)
Implement `functools.lru_cache` or a custom class-level cache for `test_pair`. Using a custom cache with a key based on `(pair_name, start_date, end_date, n_obs)` is safer and sufficient for CV folds.

### [MODIFY] [cointegration.py](file:///Volumes/PRO-G40/Code/mrs/src/feature_engineering/cointegration.py)
Update to ensure it leverages the validator's caching.

## Proposed Changes

### Feature Engineering

#### [NEW] [cointegration_validator.py](file:///Volumes/PRO-G40/Code/mrs/src/feature_engineering/cointegration_validator.py)
- Implement `CointegrationValidator` class as specified.
- Support Johansen and Engle-Granger tests.
- Add rolling stability analysis.
- Support "theory override" for strong economic priors.

#### [MODIFY] [cointegration.py](file:///Volumes/PRO-G40/Code/mrs/src/feature_engineering/cointegration.py)
- integrate `CointegrationValidator` into `CointegrationAnalyzer`.
- Update `fit` to perform validation.
- Update `transform` to use validated pairs only.
- Maintain scikit-learn compatibility.

### Configuration

#### [MODIFY] [experiment_config.yaml](file:///Volumes/PRO-G40/Code/mrs/configs/experiment_config.yaml)
- Add validation settings to `features.cointegration`.
- Add theory override settings.
- Update default pairs with theory descriptions.

### Evaluation & Pipeline

#### [MODIFY] [cross_validation.py](file:///Volumes/PRO-G40/Code/mrs/src/evaluation/cross_validation.py)
- Update `CVResult` to include `fold_metadata`.
- Update `CrossValidator.evaluate` to capture metadata from pipeline steps (e.g., validation results).

#### [MODIFY] [run_tournament.py](file:///Volumes/PRO-G40/Code/mrs/run_tournament.py)
- Update pairs configuration to include theory.
- Track cointegration stability across CV folds.
- Generate and log cointegration reports.

## Verification Plan

### Automated Tests
1. **Unit Tests for Validator**:
   - Create `tests/test_cointegration_validator.py`.
   - Test Johansen stats on synthetic cointegrated vs independent data.
   - Test Engle-Granger stats.
   - Test theory override logic.
   - Test insufficient data handling.
   ```bash
   pytest tests/test_cointegration_validator.py
   ```

2. **Integration Test**:
   - Run `run_tournament.py` with a subset of assets/models and verify cointegration reports are generated.
   - Verify `quantity_theory` is rejected on real data as expected.
   ```bash
   python run_tournament.py --assets BOND --models ridge --fred-api-key YOUR_KEY
   ```

### Manual Verification
- Inspect the logged "COINTEGRATION VALIDATION REPORT" for correctness and clarity.
- Verify that features for rejected pairs are NOT present in the final model inputs.
