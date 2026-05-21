"""ttcc-rl: training pipeline for TTCC retention-curve prediction.

Re-export the canonical curve parser so callers can do
``from ttcc_rl import parse_curve``.
"""
from ttcc_rl.parser import parse_curve  # noqa: F401
