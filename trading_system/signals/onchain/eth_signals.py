"""
ETH on-chain composite signal. Reuses BTC formulas + ETH-specific extras.
"""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from .btc_signals import compute_btc_onchain_signal


def compute_eth_onchain_signal(metrics_df: pd.DataFrame
                                ) -> Tuple[float, Dict[str, float]]:
    """Same composite as BTC for now; reserved for staking/gas additions."""
    return compute_btc_onchain_signal(metrics_df)
