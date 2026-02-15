"""Tests for confidence interval calculations in analysis tools."""

import unittest
from tools.analyze_prompt_injection_with_ci import wilson_score_interval, clopper_pearson_interval


class TestWilsonScoreInterval(unittest.TestCase):
    """Test Wilson score interval calculations."""

    def test_perfect_rate(self):
        """Wilson CI for perfect rate: upper is 1, lower shows uncertainty."""
        lower, upper = wilson_score_interval(10, 10, confidence=0.95)
        self.assertEqual(upper, 1.0)
        # Wilson gives non-degenerate interval for perfect rate
        # Lower bound should be < 1 (reflecting uncertainty)
        self.assertLess(lower, 1.0)
        self.assertGreater(lower, 0.7)  # Should be high though

    def test_zero_rate(self):
        """Wilson CI for zero rate: lower is 0, upper shows uncertainty."""
        lower, upper = wilson_score_interval(0, 10, confidence=0.95)
        self.assertEqual(lower, 0.0)
        # Wilson gives non-degenerate interval for zero rate
        # Upper bound should be > 0 (reflecting uncertainty)
        self.assertGreater(upper, 0.0)
        self.assertLess(upper, 0.3)  # Should be low though

    def test_fifty_percent(self):
        """CI for 50% with decent sample size."""
        lower, upper = wilson_score_interval(50, 100, confidence=0.95)
        # Wilson interval for 50/100 should be approximately symmetric
        # around 0.5, roughly [0.40, 0.60]
        self.assertAlmostEqual(lower, 0.40, places=1)
        self.assertAlmostEqual(upper, 0.60, places=1)
        # Should be roughly symmetric
        self.assertAlmostEqual(0.5 - lower, upper - 0.5, places=2)

    def test_small_sample_high_rate(self):
        """CI should be wide for small samples."""
        lower, upper = wilson_score_interval(5, 10, confidence=0.95)
        # Wilson interval for 5/10 is approximately [0.19, 0.81]
        self.assertLess(lower, 0.25)
        self.assertGreater(upper, 0.75)

    def test_narrower_at_99(self):
        """99% CI should be wider than 95% CI."""
        l95, u95 = wilson_score_interval(50, 100, confidence=0.95)
        l99, u99 = wilson_score_interval(50, 100, confidence=0.99)
        self.assertLess(l99, l95)
        self.assertGreater(u99, u95)

    def test_edge_case_single_trial_success(self):
        """1/1 should have CI containing values less than 1."""
        lower, upper = wilson_score_interval(1, 1, confidence=0.95)
        self.assertLess(lower, 1.0)
        self.assertEqual(upper, 1.0)

    def test_edge_case_single_trial_failure(self):
        """0/1 should have CI containing values greater than 0."""
        lower, upper = wilson_score_interval(0, 1, confidence=0.95)
        self.assertEqual(lower, 0.0)
        self.assertGreater(upper, 0.0)


class TestClopperPearsonInterval(unittest.TestCase):
    """Test Clopper-Pearson (exact) interval calculations."""

    def test_perfect_rate(self):
        """CI upper should be 1, lower should be < 1 for perfect rate."""
        lower, upper = clopper_pearson_interval(10, 10, confidence=0.95)
        self.assertLess(lower, 1.0)
        self.assertEqual(upper, 1.0)

    def test_zero_rate(self):
        """CI lower should be 0, upper should be > 0 for zero rate."""
        lower, upper = clopper_pearson_interval(0, 10, confidence=0.95)
        self.assertEqual(lower, 0.0)
        self.assertGreater(upper, 0.0)

    def test_fifty_percent(self):
        """CI for 50% with decent sample size."""
        lower, upper = clopper_pearson_interval(50, 100, confidence=0.95)
        # Exact interval should be approximately symmetric
        self.assertAlmostEqual(lower, 0.40, places=1)
        self.assertAlmostEqual(upper, 0.60, places=1)

    def test_conservative_coverage(self):
        """Clopper-Pearson should be more conservative (wider) than Wilson."""
        n_success, n_total = 5, 10
        w_lower, w_upper = wilson_score_interval(n_success, n_total, confidence=0.95)
        cp_lower, cp_upper = clopper_pearson_interval(n_success, n_total, confidence=0.95)
        # CP should be wider
        self.assertLessEqual(cp_lower, w_lower)
        self.assertGreaterEqual(cp_upper, w_upper)


class TestCIWithMockData(unittest.TestCase):
    """Test CI integration with analysis patterns."""

    def test_format_ci_for_display(self):
        """Test formatting CI for table output."""
        from tools.analyze_prompt_injection_with_ci import format_rate_with_ci

        # High rate with tight CI
        result = format_rate_with_ci(45, 50, method="wilson")
        self.assertIn("%", result)
        self.assertIn("[", result)
        self.assertIn("]", result)

        # Low N case shows warning
        result_small = format_rate_with_ci(1, 5, method="wilson")
        self.assertIn("n=5", result_small)


if __name__ == "__main__":
    unittest.main()
