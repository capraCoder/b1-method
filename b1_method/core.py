"""
core.py -- Main engine for the B1-method package.

Implements convergent tier classification of candidate basis vectors
given K independent source assessments, plus independence verification
of the source panel itself.

All functions are pure Python (stdlib only: math, dataclasses, csv).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Primitive functions
# ---------------------------------------------------------------------------

def count_convergence(row: List[str]) -> int:
    """Count how many sources confirm a candidate as present.

    Parameters
    ----------
    row : list of str
        Each element is one of: "Y", "Y*", "N", "N*".
        Y and Y* count as present (convergent).
        N and N* count as absent (non-convergent).

    Returns
    -------
    int
        Number of sources that confirm the candidate.
    """
    return sum(1 for v in row if v.strip().upper() in ("Y", "Y*"))


def classify_tier(count: int, k: int) -> str:
    """Classify a candidate into Tier 1, 2, or 3.

    Thresholds (using math.ceil for exact boundary placement):
        Tier 1: count >= ceil(2K / 3)   -- strong convergence
        Tier 2: count >= ceil(K / 3)    -- partial convergence
        Tier 3: below Tier 2 threshold  -- weak / non-convergent

    Parameters
    ----------
    count : int
        Number of confirming sources (from count_convergence).
    k : int
        Total number of independent sources (1 <= k).

    Returns
    -------
    str
        "Tier 1", "Tier 2", or "Tier 3".
    """
    if k < 1:
        raise ValueError(f"K must be >= 1, got {k}")

    tier1_threshold = math.ceil(2 * k / 3)
    tier2_threshold = math.ceil(k / 3)

    if count >= tier1_threshold:
        return "Tier 1"
    if count >= tier2_threshold:
        return "Tier 2"
    return "Tier 3"


def independence_check(sources: List[Dict]) -> List[str]:
    """Verify methodological independence of the source panel.

    Checks three diversity dimensions:
        1. method_type  -- at least 2 distinct methods recommended
        2. languages    -- at least 2 distinct languages across all sources
        3. tradition    -- at least 2 distinct intellectual traditions

    Parameters
    ----------
    sources : list of dict
        Each dict may contain optional keys:
            - method_type : str   (e.g. "lexical", "questionnaire", "indigenous")
            - languages   : list of str  (e.g. ["English", "German"])
            - tradition   : str   (e.g. "Western", "East Asian")

    Returns
    -------
    list of str
        Warning messages for each failed diversity check.
        Empty list if all checks pass.
    """
    if not sources:
        return ["No source metadata provided; independence cannot be verified."]

    warnings = []

    # --- method_type diversity ---
    methods = {s.get("method_type") for s in sources if s.get("method_type")}
    if len(methods) < 2:
        warnings.append(
            f"Low method diversity: only {len(methods)} distinct method type(s) "
            f"found ({', '.join(sorted(methods)) or 'none'}). "
            f"At least 2 recommended."
        )

    # --- language diversity ---
    all_languages = set()
    for s in sources:
        langs = s.get("languages")
        if isinstance(langs, list):
            all_languages.update(langs)
        elif isinstance(langs, str):
            all_languages.add(langs)
    if len(all_languages) < 2:
        warnings.append(
            f"Low language diversity: only {len(all_languages)} distinct language(s) "
            f"found ({', '.join(sorted(all_languages)) or 'none'}). "
            f"At least 2 recommended."
        )

    # --- tradition diversity ---
    traditions = {s.get("tradition") for s in sources if s.get("tradition")}
    if len(traditions) < 2:
        warnings.append(
            f"Low tradition diversity: only {len(traditions)} distinct tradition(s) "
            f"found ({', '.join(sorted(traditions)) or 'none'}). "
            f"At least 2 recommended."
        )

    return warnings


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class B1Result:
    """Container for a complete B1-method analysis result.

    Attributes
    ----------
    domain : str
        Domain label (e.g. "personality", "colour", "emotion").
    k : int
        Number of independent sources.
    tiers : dict
        Mapping of candidate name -> {"count": int, "tier": str, "alignment": list}.
    warnings : list of str
        Independence-check warnings (empty if all clear).
    lower_bound : int
        Number of Tier-1 candidates (the convergent lower bound on dimensionality).
    tier1_candidates : list of str
        Candidates classified as Tier 1.
    tier2_candidates : list of str
        Candidates classified as Tier 2.
    tier3_candidates : list of str
        Candidates classified as Tier 3.
    """

    domain: str = ""
    k: int = 0
    tiers: Dict[str, Dict] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    lower_bound: int = 0
    tier1_candidates: List[str] = field(default_factory=list)
    tier2_candidates: List[str] = field(default_factory=list)
    tier3_candidates: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to a plain dict for JSON serialisation."""
        return {
            "domain": self.domain,
            "k": self.k,
            "lower_bound": self.lower_bound,
            "tier1_candidates": self.tier1_candidates,
            "tier2_candidates": self.tier2_candidates,
            "tier3_candidates": self.tier3_candidates,
            "tiers": self.tiers,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Main analysis class
# ---------------------------------------------------------------------------

class B1Analysis:
    """Run a B1-method tier-classification analysis.

    Parameters
    ----------
    alignment : dict
        Mapping of candidate name -> list of alignment strings.
        Each list has K elements, one per source (Y, Y*, N, N*).
    sources : list of dict, optional
        Source metadata for independence checking.
        Each dict may have keys: name, method_type, languages, tradition.
    domain : str
        Domain label for reporting.

    Examples
    --------
    >>> alignment = {
    ...     "Extraversion": ["Y", "Y", "Y*"],
    ...     "Agreeableness": ["Y", "Y*", "N"],
    ...     "Openness": ["N", "N", "N"],
    ... }
    >>> result = B1Analysis(alignment, domain="personality").run()
    >>> result.tier1_candidates
    ['Extraversion']
    """

    def __init__(
        self,
        alignment: Dict[str, List[str]],
        sources: Optional[List[Dict]] = None,
        domain: str = "",
    ):
        self.alignment = alignment
        self.sources = sources
        self.domain = domain

        # Derive K from alignment data
        lengths = [len(v) for v in alignment.values()]
        if not lengths:
            raise ValueError("Alignment dict is empty; nothing to analyse.")
        if len(set(lengths)) != 1:
            raise ValueError(
                f"Inconsistent alignment lengths across candidates: {set(lengths)}. "
                f"Every candidate must have exactly K entries."
            )
        self.k = lengths[0]

    def run(self) -> B1Result:
        """Execute the B1 analysis pipeline.

        Steps:
            1. Count convergence for each candidate.
            2. Classify each candidate into Tier 1/2/3.
            3. Run independence check on sources (if provided).
            4. Assemble and return B1Result.

        Returns
        -------
        B1Result
        """
        tiers = {}
        tier1, tier2, tier3 = [], [], []

        for candidate, row in self.alignment.items():
            count = count_convergence(row)
            tier = classify_tier(count, self.k)
            tiers[candidate] = {
                "count": count,
                "tier": tier,
                "alignment": list(row),
            }

            if tier == "Tier 1":
                tier1.append(candidate)
            elif tier == "Tier 2":
                tier2.append(candidate)
            else:
                tier3.append(candidate)

        # Independence verification
        if self.sources is not None:
            warnings = independence_check(self.sources)
        else:
            warnings = ["No source metadata provided; independence cannot be verified."]

        return B1Result(
            domain=self.domain,
            k=self.k,
            tiers=tiers,
            warnings=warnings,
            lower_bound=len(tier1),
            tier1_candidates=tier1,
            tier2_candidates=tier2,
            tier3_candidates=tier3,
        )

    @classmethod
    def from_csv(cls, path: str, sources: Optional[List[Dict]] = None, domain: str = "") -> "B1Analysis":
        """Load alignment data from a CSV file.

        CSV format (first column = candidate, remaining columns = source verdicts):

            candidate,S1_Name,S2_Name,S3_Name
            Extraversion,Y,Y,Y*
            Agreeableness,Y,Y*,N

        Source metadata can be supplied separately via the `sources` parameter
        or loaded from a companion file via the io module.

        Parameters
        ----------
        path : str
            Path to alignment CSV file.
        sources : list of dict, optional
            Source metadata dicts.
        domain : str
            Domain label.

        Returns
        -------
        B1Analysis
        """
        # Lazy import to avoid circular dependency and keep io module optional
        from b1_method.io import load_alignment_csv

        alignment, _source_names = load_alignment_csv(path)
        return cls(alignment=alignment, sources=sources, domain=domain)

    @staticmethod
    def print_report(result: B1Result) -> None:
        """Print a formatted tier-classification report to stdout.

        Parameters
        ----------
        result : B1Result
            The analysis result to display.
        """
        header = f"B1 Analysis: {result.domain}" if result.domain else "B1 Analysis"
        print(f"\n{'=' * 60}")
        print(f"  {header}")
        print(f"  K = {result.k} sources | Lower bound = {result.lower_bound}")
        print(f"{'=' * 60}\n")

        # Tier thresholds for display
        t1_thresh = math.ceil(2 * result.k / 3)
        t2_thresh = math.ceil(result.k / 3)
        print(f"  Thresholds: Tier 1 >= {t1_thresh} | Tier 2 >= {t2_thresh} | Tier 3 < {t2_thresh}\n")

        # Column widths
        max_name = max((len(c) for c in result.tiers), default=9)
        col_w = max(max_name, 12)

        # Header row
        print(f"  {'Candidate':<{col_w}}  {'Count':>5}  {'Tier':>4}  Alignment")
        print(f"  {'-' * col_w}  {'-' * 5}  {'-' * 4}  {'-' * (result.k * 4)}")

        # Sort by tier (ascending), then by count (descending)
        sorted_candidates = sorted(
            result.tiers.items(),
            key=lambda item: (item[1]["tier"], -item[1]["count"], item[0]),
        )

        for candidate, info in sorted_candidates:
            alignment_str = " ".join(f"{v:>2}" for v in info["alignment"])
            print(
                f"  {candidate:<{col_w}}  {info['count']:>5}  {info['tier']:>4}  {alignment_str}"
            )

        # Summary
        print(f"\n  Tier 1 ({len(result.tier1_candidates)}): "
              f"{', '.join(result.tier1_candidates) or '(none)'}")
        print(f"  Tier 2 ({len(result.tier2_candidates)}): "
              f"{', '.join(result.tier2_candidates) or '(none)'}")
        print(f"  Tier 3 ({len(result.tier3_candidates)}): "
              f"{', '.join(result.tier3_candidates) or '(none)'}")

        # Warnings
        if result.warnings:
            print(f"\n  Warnings:")
            for w in result.warnings:
                print(f"    - {w}")

        print(f"\n{'=' * 60}\n")
