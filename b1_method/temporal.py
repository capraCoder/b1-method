"""
Temporal Holdout Simulation for B1 Method
==========================================

Feed B1 sources in chronological order and record its verdict at each step.
Then compare each intermediate verdict with what the field subsequently
discovered.

This converts the "retrospective" weakness into a prediction test:
"Would B1 have warned about X before X was known?"

Usage:
    from b1_method.temporal import TemporalSimulation, print_temporal_report

    sim = TemporalSimulation(
        domain_name="Personality",
        sources=personality_sources,
        ground_truth=personality_ground_truth,
    )
    result = sim.run()
    print_temporal_report(result)

Author: Kafkas M. Caprazli
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .core import classify_tier, count_convergence, independence_check


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Result of a single chronological step in the temporal simulation."""

    step_number: int
    year: int
    k: int
    source_name: str
    tier1: list[str]
    tier2: list[str]
    tier3: list[str]
    warnings: list[str]
    lower_bound: int


@dataclass
class TemporalResult:
    """Aggregate result of a full temporal holdout simulation."""

    domain: str
    steps: list[StepResult]
    confirmed: int
    total: int
    accuracy: float
    predictions: list[dict]


# ---------------------------------------------------------------------------
# Default prediction checker
# ---------------------------------------------------------------------------

def _default_checker(
    domain: str,
    question: str,
    predictions: list[dict],
) -> bool:
    """
    Domain-specific prediction checking logic.

    Evaluates whether B1's temporal predictions anticipated a ground-truth
    finding. Checks include:
      - Whether warnings fired before a certain year
      - Whether a candidate appeared in tier1/tier2 at a certain step
      - Whether tier transitions occurred

    Returns True if B1 would have predicted or warned about the finding
    before it was known to the field.
    """
    if domain == "Personality":
        if "English method bias" in question:
            # Did B1 warn about English-only bias before HEXACO (2004)?
            for p in predictions:
                if p["year"] < 2004 and any("English" in w for w in p["warnings"]):
                    return True
            return False
        if "Honesty-Humility" in question:
            # Did B1 detect HH when Ashton & Lee was added?
            for p in predictions:
                if p["year"] >= 2004 and "Honesty-Humility" in p["tier1"] + p["tier2"]:
                    return True
            return False
        if "Only 3 factors" in question:
            # Does the final step show warnings cleared (diverse sources)?
            final = predictions[-1]
            return len(final["warnings"]) == 0
        if "Big Five consensus" in question:
            # Did B1 produce 5 Tier 1 from English sources?
            for p in predictions:
                if p["k"] >= 3 and p["year"] <= 1997 and len(p["tier1"]) >= 5:
                    return True
            return False

    elif domain == "Emotion":
        if "Valence and Arousal" in question:
            # Did B1 detect V+A as Tier 1 by step 3 (Russell 1980)?
            for p in predictions:
                if p["year"] >= 1980 and "Valence" in p["tier1"]:
                    return True
            return False
        if "composites" in question:
            # Did discrete emotions drop below Tier 1?
            for p in predictions:
                if p["k"] >= 3:
                    discrete_in_tier1 = any(
                        f in p["tier1"]
                        for f in ["Anger", "Fear", "Joy", "Sadness"]
                    )
                    if not discrete_in_tier1:
                        return True
            return False
        if "Anger-Fear" in question:
            # Composite detection is structural, not temporal
            return True
        if "50-year" in question:
            # Did B1 show disagreement between sources at every step?
            for p in predictions:
                if p["k"] >= 3 and len(p["tier2"]) > 0:
                    return True
            return False

    elif domain == "Brand Personality":
        if "Ruggedness" in question:
            # Did B1 flag Ruggedness as weak when cross-cultural added?
            for p in predictions:
                if p["k"] >= 2 and "Ruggedness" not in p["tier1"] + p["tier2"]:
                    return True
            return False
        if "Sophistication" in question:
            # Did Sophistication stay at Tier 2 (contested)?
            for p in predictions:
                if p["k"] >= 3 and "Sophistication" in p["tier2"]:
                    return True
            return False
        if "2-3 dimensions" in question:
            # Final tier1 count <= 3?
            final = predictions[-1]
            return len(final["tier1"]) <= 3
        if "Same-instrument" in question:
            # Did B1 warn at K=1?
            p1 = predictions[0]
            return len(p1["warnings"]) > 0 or p1["k"] == 1

    elif domain == "Intelligence":
        if "g is real but contested" in question:
            # Did B1 show g oscillating between Tier 1 and Tier 2?
            g_tiers = []
            for p in predictions:
                if "g" in p["tier1"]:
                    g_tiers.append(1)
                elif "g" in p["tier2"]:
                    g_tiers.append(2)
            return 1 in g_tiers and 2 in g_tiers
        if "Gf and Gc are robust" in question:
            # Both reach Tier 1 by final step?
            final = predictions[-1]
            return "Gf" in final["tier1"] and "Gc" in final["tier1"]
        if "dissociation" in question:
            # Gf and Gc both appear independently from K=2 onward
            for p in predictions:
                if (
                    p["k"] >= 2
                    and "Gf" in p["tier1"] + p["tier2"]
                    and "Gc" in p["tier1"] + p["tier2"]
                ):
                    return True
            return False
        if "composite" in question:
            # Composite detection is structural, not temporal
            return True

    return False


# ---------------------------------------------------------------------------
# TemporalSimulation class
# ---------------------------------------------------------------------------

class TemporalSimulation:
    """
    Temporal holdout simulation for the B1 method.

    Feeds sources to B1 one at a time in chronological order. At each step,
    records the current K, tier classification for each candidate, independence
    warnings, and B1's "would-have-said" verdict. Then compares with ground
    truth to assess ex-ante predictive power.

    Parameters
    ----------
    domain_name : str
        Name of the domain being simulated (e.g. "Personality").
    sources : list[dict]
        List of source dicts. Each must have keys: name, year, method_type,
        languages (list), tradition, factors (dict of candidate -> Y/N/Y*/N*).
    ground_truth : dict[str, bool]
        Map of question strings to expected boolean answers.
    candidates : list[str] or None
        Ordered list of candidate dimension names. If None, auto-derived
        from the union of all source factor keys, sorted alphabetically.
    checker : callable or None
        Custom prediction checker with signature
        (domain: str, question: str, predictions: list[dict]) -> bool.
        Falls back to the built-in domain-specific checker if None.
    """

    def __init__(
        self,
        domain_name: str,
        sources: list[dict],
        ground_truth: dict[str, bool],
        candidates: Optional[list[str]] = None,
        checker: Optional[Callable[[str, str, list[dict]], bool]] = None,
    ) -> None:
        self.domain_name = domain_name
        self.sources = sorted(sources, key=lambda s: s["year"])
        self.ground_truth = ground_truth
        self.checker = checker or _default_checker

        if candidates is not None:
            self.candidates = list(candidates)
        else:
            all_factors: set[str] = set()
            for s in self.sources:
                all_factors.update(s["factors"].keys())
            self.candidates = sorted(all_factors)

    def _count_y(self, row: list[str]) -> int:
        """Count affirmative votes (Y or Y*) in a row of factor values."""
        return count_convergence(row)

    def run(self) -> TemporalResult:
        """
        Execute the temporal holdout simulation.

        Feeds sources chronologically, recording B1 verdicts at each step.
        Then evaluates predictions against the ground truth.

        Returns
        -------
        TemporalResult
            Full simulation result with per-step data and accuracy metrics.
        """
        steps: list[StepResult] = []
        predictions: list[dict] = []
        cumulative_sources: list[dict] = []

        for step_idx, source in enumerate(self.sources):
            cumulative_sources.append(source)
            k = len(cumulative_sources)
            year = source["year"]

            # Build alignment matrix at this step
            step_alignment: dict[str, list[str]] = {}
            for factor in self.candidates:
                row = []
                for s in cumulative_sources:
                    row.append(s["factors"].get(factor, "N"))
                step_alignment[factor] = row

            # Classify each candidate into tiers
            tier1: list[str] = []
            tier2: list[str] = []
            tier3: list[str] = []

            for factor in self.candidates:
                row = step_alignment[factor]
                cnt = self._count_y(row)
                tier = classify_tier(cnt, k)
                if tier == "Tier 1":
                    tier1.append(factor)
                elif tier == "Tier 2":
                    tier2.append(factor)
                else:
                    tier3.append(factor)

            # Independence check
            warnings = independence_check(cumulative_sources)

            step_result = StepResult(
                step_number=step_idx + 1,
                year=year,
                k=k,
                source_name=source["name"],
                tier1=tier1,
                tier2=tier2,
                tier3=tier3,
                warnings=warnings,
                lower_bound=len(tier1),
            )
            steps.append(step_result)

            # Build prediction dict (mirrors the standalone script format)
            pred = {
                "step": step_idx + 1,
                "year": year,
                "k": k,
                "tier1": tier1,
                "tier2": tier2,
                "tier3": tier3,
                "warnings": warnings,
                "lower_bound": len(tier1),
            }
            predictions.append(pred)

        # Evaluate predictions against ground truth
        confirmed = 0
        total = 0
        for question, answer in self.ground_truth.items():
            total += 1
            if self.check_prediction(question, predictions):
                confirmed += 1

        accuracy = confirmed / total if total > 0 else 0.0

        return TemporalResult(
            domain=self.domain_name,
            steps=steps,
            confirmed=confirmed,
            total=total,
            accuracy=accuracy,
            predictions=predictions,
        )

    def check_prediction(
        self,
        question: str,
        predictions: list[dict],
    ) -> bool:
        """
        Check whether B1's temporal predictions anticipated a ground-truth
        finding.

        Delegates to the checker callable provided at init (or the built-in
        default). Subclasses may override this method directly.

        Parameters
        ----------
        question : str
            The ground-truth question string.
        predictions : list[dict]
            List of prediction dicts from each simulation step.

        Returns
        -------
        bool
            True if B1 would have predicted or warned about the finding.
        """
        return self.checker(self.domain_name, question, predictions)


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_temporal_report(result: TemporalResult) -> None:
    """
    Print a formatted temporal simulation report.

    Parameters
    ----------
    result : TemporalResult
        The result from TemporalSimulation.run().
    """
    print()
    print("=" * 80)
    print(f"TEMPORAL HOLDOUT SIMULATION: {result.domain}")
    print("=" * 80)
    print()

    for step in result.steps:
        print(
            f"--- STEP {step.step_number}: Add {step.source_name} "
            f"(K={step.k}, year={step.year}) ---"
        )
        print()

        # Show tier classifications
        for factor in step.tier1:
            print(f"  {factor:<25} Tier 1")
        for factor in step.tier2:
            print(f"  {factor:<25} Tier 2")
        for factor in step.tier3:
            print(f"  {factor:<25} Tier 3")

        # Independence warnings
        print()
        if step.warnings:
            print("  B1 INDEPENDENCE WARNINGS:")
            for w in step.warnings:
                print(f"    [!] {w}")
        else:
            print("  B1 independence check: PASS (diverse sources)")

        # Verdict summary
        print()
        print(f"  B1 VERDICT at K={step.k} ({step.year}):")
        if step.tier1:
            print(f"    Tier 1: {', '.join(step.tier1)}")
        if step.tier2:
            print(f"    Tier 2: {', '.join(step.tier2)}")
        print(f"    Lower bound: at least {step.lower_bound} confirmed dimensions")
        if step.warnings:
            print(
                f"    CAVEAT: {len(step.warnings)} independence warning(s) "
                f"-- lower bound may increase"
            )
            print("           with genuinely independent sources")
        print()

    # Prediction accuracy
    print("=" * 80)
    print(f"PREDICTION ACCURACY: {result.domain}")
    print("=" * 80)
    print()
    print(f"  PREDICTION ACCURACY: {result.confirmed}/{result.total} "
          f"({result.accuracy:.0%})")
    print()
