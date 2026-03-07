"""Tests for b1_method.core -- the engine."""
import math
from b1_method.core import classify_tier, count_convergence, independence_check, B1Analysis


def test_count_convergence_basic():
    assert count_convergence(["Y", "Y", "N"]) == 2
    assert count_convergence(["Y*", "N*", "Y"]) == 2
    assert count_convergence(["N", "N", "N"]) == 0
    assert count_convergence(["Y", "Y", "Y", "Y"]) == 4


def test_count_convergence_star_variants():
    """Y* counts as present, N* counts as absent."""
    assert count_convergence(["Y*"]) == 1
    assert count_convergence(["N*"]) == 0


def test_classify_tier_k6():
    """K=6: Tier 1 >= 4, Tier 2 >= 2, Tier 3 < 2."""
    assert classify_tier(6, 6) == "Tier 1"
    assert classify_tier(5, 6) == "Tier 1"
    assert classify_tier(4, 6) == "Tier 1"
    assert classify_tier(3, 6) == "Tier 2"
    assert classify_tier(2, 6) == "Tier 2"
    assert classify_tier(1, 6) == "Tier 3"
    assert classify_tier(0, 6) == "Tier 3"


def test_classify_tier_k5():
    """K=5: Tier 1 >= 4, Tier 2 >= 2."""
    assert classify_tier(5, 5) == "Tier 1"
    assert classify_tier(4, 5) == "Tier 1"
    assert classify_tier(3, 5) == "Tier 2"
    assert classify_tier(2, 5) == "Tier 2"
    assert classify_tier(1, 5) == "Tier 3"


def test_classify_tier_k4():
    """K=4: Tier 1 >= 3, Tier 2 >= 2."""
    assert classify_tier(4, 4) == "Tier 1"
    assert classify_tier(3, 4) == "Tier 1"
    assert classify_tier(2, 4) == "Tier 2"
    assert classify_tier(1, 4) == "Tier 3"


def test_classify_tier_k1():
    """K=1: everything is Tier 1 (trivially)."""
    assert classify_tier(1, 1) == "Tier 1"
    assert classify_tier(0, 1) == "Tier 3"


def test_independence_check_all_same():
    sources = [
        {"method_type": "lexical FA", "languages": ["English"], "tradition": "lexical"},
        {"method_type": "lexical FA", "languages": ["English"], "tradition": "lexical"},
    ]
    warnings = independence_check(sources)
    assert len(warnings) == 3  # same method, same language, same tradition


def test_independence_check_diverse():
    sources = [
        {"method_type": "lexical FA", "languages": ["English"], "tradition": "lexical"},
        {"method_type": "questionnaire", "languages": ["English", "German"], "tradition": "questionnaire"},
    ]
    warnings = independence_check(sources)
    assert not any("same method" in w for w in warnings)
    assert not any("English-only" in w for w in warnings)


def test_b1analysis_personality():
    """Smoke test: personality alignment produces expected tiers."""
    alignment = {
        "Extraversion":       ["Y", "Y", "Y", "Y", "Y", "Y"],
        "Agreeableness":      ["Y", "Y", "Y*", "Y", "Y", "Y"],
        "Conscientiousness":  ["Y", "Y", "Y", "Y", "Y", "Y"],
        "Neuroticism":        ["Y", "Y", "Y*", "N*", "Y", "Y"],
        "Openness":           ["Y", "Y", "Y", "N*", "Y", "Y"],
        "Honesty-Humility":   ["N", "N", "Y", "Y*", "N", "N"],
    }
    analysis = B1Analysis(alignment, domain="Personality")
    result = analysis.run()
    assert result.k == 6
    assert "Extraversion" in result.tier1_candidates
    assert "Honesty-Humility" in result.tier2_candidates
    assert result.lower_bound == 5  # 5 Tier 1


def test_b1analysis_brand():
    """Brand personality: Ruggedness should be Tier 3."""
    alignment = {
        "Sincerity":      ["Y", "Y", "Y*", "Y"],
        "Excitement":     ["Y", "Y", "Y*", "Y"],
        "Competence":     ["Y", "Y*", "Y*", "N"],
        "Sophistication": ["Y", "Y*", "N*", "N"],
        "Ruggedness":     ["Y", "N", "N*", "N"],
    }
    analysis = B1Analysis(alignment, domain="Brand Personality")
    result = analysis.run()
    assert "Ruggedness" in result.tier3_candidates
    assert result.lower_bound <= 3
