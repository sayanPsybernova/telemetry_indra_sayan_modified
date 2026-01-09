"""
CSV utilities for telemetry data loading.
"""
from typing import Dict, List

import pandas as pd


def normalize_telemetry_columns(
    df: pd.DataFrame,
    required_cols: List[str],
    alias_map: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Normalize and validate telemetry CSV columns.

    - Strips column names
    - Renames alias columns to required names
    - Ensures required columns exist
    """
    df.columns = [col.strip() for col in df.columns]
    normalized = {col.lower(): col for col in df.columns}

    rename_map: Dict[str, str] = {}
    found_cols = set(normalized.keys())

    for required in required_cols:
        if required in found_cols:
            continue
        for alias in alias_map.get(required, []):
            if alias in normalized:
                rename_map[normalized[alias]] = required
                found_cols.add(required)
                break

    missing = [col for col in required_cols if col not in found_cols]
    if missing:
        available = ", ".join(df.columns)
        raise ValueError(
            f"Missing required columns: {', '.join(missing)}. "
            f"Found columns: {available}"
        )

    for required in required_cols:
        original = normalized.get(required)
        if original and original != required:
            rename_map[original] = required

    if rename_map:
        df = df.rename(columns=rename_map)

    return df
