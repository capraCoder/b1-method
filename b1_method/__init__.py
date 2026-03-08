"""
b1-method: Domain-Independent Convergent Derivation of Canonical Basis Vectors
===============================================================================

Given K independent sources proposing competing dimensional structures for the
same domain, B1 produces a tier-classified, independence-verified, composite-
checked basis — a lower bound on the manifold's topological complexity.

Usage:
    from b1_method import B1Analysis

    analysis = B1Analysis.from_csv("sources.csv")
    result = analysis.run()
    result.print_report()

CLI:
    b1-method run sources.csv
    b1-method temporal sources.csv
"""

__version__ = "0.2.0"

from b1_method.core import B1Analysis, B1Result
from b1_method.temporal import TemporalSimulation

__all__ = ["B1Analysis", "B1Result", "TemporalSimulation"]
