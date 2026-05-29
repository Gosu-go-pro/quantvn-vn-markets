import numpy as np
import pandas as pd


FAST_WINDOW = 20
SLOW_WINDOW = 50
POSITION_SIZE = 100
REQUIRED_COLUMNS = ["Date", "time", "Open", "High", "Low", "Close", "volume"]


def _find_column(df: pd.DataFrame, names):
    columns_by_lower = {str(col).lower(): col for col in df.columns}
    for name in names:
        if name in df.columns:
            return name
        match = columns_by_lower.get(name.lower())
        if match is not None:
            return match
    return None


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    datetime_col = _find_column(result, ["Datetime", "datetime", "timestamp"])
    date_col = _find_column(result, ["Date", "date"])
    time_col = _find_column(result, ["time", "Time"])

    if date_col is None and datetime_col is not None:
        dt = pd.to_datetime(result[datetime_col], errors="coerce")
        result["Date"] = dt.dt.strftime("%Y-%m-%d")
        result["time"] = dt.dt.strftime("%H:%M:%S").fillna("00:00:00")
    elif date_col is not None:
        dt = pd.to_datetime(result[date_col], errors="coerce")
        result["Date"] = dt.dt.strftime("%Y-%m-%d")
        if time_col is not None:
            result["time"] = result[time_col].astype(str)
        else:
            result["time"] = dt.dt.strftime("%H:%M:%S").fillna("00:00:00")

    for target, aliases in {
        "Open": ["Open", "open", "o"],
        "High": ["High", "high", "h"],
        "Low": ["Low", "low", "l"],
        "Close": ["Close", "close", "c"],
        "volume": ["volume", "Volume", "vol", "Vol", "v"],
    }.items():
        source = _find_column(result, aliases)
        if source is not None:
            result[target] = pd.to_numeric(result[source], errors="coerce")

    if "Close" in result.columns:
        for price_col in ["Open", "High", "Low"]:
            if price_col not in result.columns:
                result[price_col] = result["Close"]
    if "volume" not in result.columns:
        result["volume"] = 0

    missing = [col for col in REQUIRED_COLUMNS if col not in result.columns]
    if missing:
        raise ValueError(
            "Data khong dung format. Ban can truyen vao cac cot: "
            + ", ".join(REQUIRED_COLUMNS)
        )

    return result


def gen_position(df: pd.DataFrame) -> pd.DataFrame:
    """Generate MA crossover signals and long-only stock positions."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    result = _normalize_ohlcv(df)

    close = pd.to_numeric(result["Close"], errors="coerce")
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
