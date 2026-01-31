from src.evaluation.multiple_testing import MultipleTestingCorrector, HypothesisTest

def test_bonferroni_correction():
    print("Testing Bonferroni correction...")
    corrector = MultipleTestingCorrector(fwer_alpha=0.05)
    tests = [
        HypothesisTest("t1", "A", "M1", "T", 0.001, 0.5, 4.0),
        HypothesisTest("t2", "A", "M2", "T", 0.04, 0.2, 2.1),
    ] + [
        HypothesisTest(f"t{i}", "A", f"M{i}", "T", 0.5, 0.0, 0.0)
        for i in range(3, 11)
    ]
    result = corrector.bonferroni(tests)
    assert result.alpha_adjusted == 0.005
    assert result.n_significant_adjusted == 1
    print("✓ Success")

def test_benjamini_hochberg():
    print("Testing Benjamini-Hochberg...")
    corrector = MultipleTestingCorrector(fdr_alpha=0.10)
    tests = [
        HypothesisTest("t1", "A", "M1", "T", 0.001, 0.5, 4.0),
        HypothesisTest("t2", "A", "M2", "T", 0.02, 0.3, 2.5),
        HypothesisTest("t3", "A", "M3", "T", 0.03, 0.2, 2.2),
    ] + [
        HypothesisTest(f"t{i}", "A", f"M{i}", "T", 0.8, 0.0, 0.0)
        for i in range(4, 11)
    ]
    result = corrector.benjamini_hochberg(tests)
    assert result.n_significant_adjusted == 3
    print("✓ Success")

if __name__ == "__main__":
    try:
        test_bonferroni_correction()
        test_benjamini_hochberg()
        print("\nAll Multiple Testing Correction tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
