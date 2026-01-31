import pytest
import numpy as np
import pandas as pd
from src.evaluation.multiple_testing import (
    MultipleTestingCorrector,
    HypothesisTest
)


def test_bonferroni_correction():
    """Test Bonferroni correction logic."""
    corrector = MultipleTestingCorrector(fwer_alpha=0.05)
    
    # 10 tests, one extremely significant, others not
    tests = [
        HypothesisTest("t1", "A", "M1", "T", 0.001, 0.5, 4.0),
        HypothesisTest("t2", "A", "M2", "T", 0.04, 0.2, 2.1),  # Sig original, but not adj
    ] + [
        HypothesisTest(f"t{i}", "A", f"M{i}", "T", 0.5, 0.0, 0.0)
        for i in range(3, 11)
    ]
    
    result = corrector.bonferroni(tests)
    
    assert result.n_tests == 10
    assert result.alpha_adjusted == 0.005
    assert result.n_significant_original == 2
    assert result.n_significant_adjusted == 1
    assert "t1" in result.significant_tests
    assert "t2" not in result.significant_tests


def test_benjamini_hochberg():
    """Test B-H FDR control."""
    corrector = MultipleTestingCorrector(fdr_alpha=0.10)
    
    # Tests that should pass FDR but fail Bonferroni
    tests = [
        HypothesisTest("t1", "A", "M1", "T", 0.001, 0.5, 4.0),
        HypothesisTest("t2", "A", "M2", "T", 0.02, 0.3, 2.5),
        HypothesisTest("t3", "A", "M3", "T", 0.03, 0.2, 2.2),
    ] + [
        HypothesisTest(f"t{i}", "A", f"M{i}", "T", 0.8, 0.0, 0.0)
        for i in range(4, 11)
    ]
    
    result = corrector.benjamini_hochberg(tests)
    
    # B-H threshold for k=3: (3/10) * 0.10 = 0.03
    # t3 (0.03) <= 0.03, so t1, t2, t3 are significant
    assert result.n_significant_adjusted == 3
    assert "t3" in result.significant_tests


def test_holm_bonferroni():
    """Test Holm-Bonferroni stepdown."""
    corrector = MultipleTestingCorrector(fwer_alpha=0.05)
    
    tests = [
        HypothesisTest("t1", "A", "M1", "T", 0.001, 0.5, 4.0),
        HypothesisTest("t2", "A", "M2", "T", 0.01, 0.3, 2.5), # (0.05 / 9) = 0.0055 -> NOT sig if t1 was at 0.01
    ]
    
    # Let's try: t1=0.001 (thresh 0.05/2=0.025), t2=0.01 (thresh 0.05/1=0.05)
    t1 = HypothesisTest("t1", "A", "M1", "T", 0.001, 0.5, 4.0)
    t2 = HypothesisTest("t2", "A", "M2", "T", 0.04, 0.3, 2.5)
    
    # m=2. 
    # k=1: p=0.001, threshold=0.05/(2-0) = 0.025. p < threshold, sig.
    # k=2: p=0.04, threshold=0.05/(2-1) = 0.05. p < threshold, sig.
    result = corrector.holm([t1, t2])
    assert result.n_significant_adjusted == 2
    
    # But if t1 failed:
    t1_fail = HypothesisTest("t1", "A", "M1", "T", 0.03, 0.5, 4.0)
    # k=1: p=0.03, threshold=0.05/2 = 0.025. p > threshold, NOT sig. Stop.
    result2 = corrector.holm([t1_fail, t2])
    assert result2.n_significant_adjusted == 0
