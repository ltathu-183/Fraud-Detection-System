"""Canonical fraud pipeline public API.

Legacy experiments remain available through their explicit module paths but are
not imported into the supported package surface.
"""

from .ieee_cis_pipeline import FraudDetectionPipeline

__all__ = ["FraudDetectionPipeline"]
