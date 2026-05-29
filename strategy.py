import numpy as np
import pandas as pd


FAST_WINDOW = 20
SLOW_WINDOW = 50
POSITION_SIZE = 100


def gen_position(df: pd.DataFrame) -> pd.DataFrame:
    """Generate MA crossover signals and long-only stock positions."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    result = df.copy()
    close_col = "Close" if "Close" in result.columns else "close"
    if close_col not in result.columns:
        raise ValueError("df must contain a Close or close column")

    close = pd.to_numeric(result[close_col], errors="coerce")
    result["ma_fast"] = close.rolling(FAST_WINDOW, min_periods=FAST_WINDOW).mean()
    result["ma_slow"] = close.rolling(SLOW_WINDOW, min_periods=SLOW_WINDOW).mean()

    has_signal = result["ma_fast"].notna() & result["ma_slow"].notna()
    long_state = (has_signal & (result["ma_fast"] > result["ma_slow"])).astype(int)

    transition = long_state.diff().fillna(0)
    result["signal"] = np.select(
        [transition > 0, transition < 0],
        [1, -1],
        default=0,
    ).astype(int)
    result["position"] = (long_state * POSITION_SIZE).astype(int)

    return result
