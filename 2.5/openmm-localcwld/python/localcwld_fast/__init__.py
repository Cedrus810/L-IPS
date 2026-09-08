"""Opt-in, analytic float64 CPU evaluator; not an OpenMM Force or GPU backend."""

from .evaluator import LocalCWLDFastEvaluator

__all__ = ["LocalCWLDFastEvaluator"]
