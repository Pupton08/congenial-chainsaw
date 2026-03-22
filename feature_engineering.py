"""
Stage 2 — Feature Engineering & Normalisation Pipeline
Trading Algorithm Blueprint

Reads raw CSVs from data/raw/, computes all features specified in the blueprint,
and saves normalised feature arrays to data/processed/ as .parquet files.

Features computed:
  2.2  OHLC-derived: log return, body ratio, wick ratios, range z-score,
                     volume z-score, volume delta, gap
  2.3  Technical:    ATR(7/14/28), ATR ratio, RSI(14/28), MACD histogram,
                     ROC(5/10/20/60), Bollinger Band position,
                     SMA20/50 distance, rolling correlation to index,
                     high-low range percentile
  2.4  Calendar:     day-of-week sin/cos, month sin/cos, hour sin/cos (intraday),
                     forex session flags, quarter-end flag
  1.5  Regime label: trend regime + volatility regime (6 classes)

Usage:
    python feature_engineering.py
"""

import os
import glob
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ─── Configuration ────────────────────────────────────────────────────────────

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
LOG_DIR       = "logs"
LOOKBACK      = 60      # bars per training sample
ROLLING_WIN   = 20      # for z-score normalisation of volume/range
Z_WIN         = 60      # rolling z-score window (blueprint 2.5)
Z_CLIP        = 3.0     # clip z-scores to [-3, +3]

# Reference tickers used for rolling correlation feature
INDEX_REFS = {
    "1D":  "SP500_GSPC_1D",
    "1W":  "SP500_GSPC_1W",
    "1H":  "SP500_GSPC_1H",
    "4H":  "SP500_GSPC_4H",
}

# ─── Logging ──────────────────────────────────────────────────────────────────

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "feature_engineering.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ─── Low-level indicators ─────────────────────────────────────────────────────

def atr(high, low, close, period):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def rsi(close, period):
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def rolling_zscore(series, window):
    m = series.rolling(window, min_periods=window // 2).mean()
    s = series.rolling(window, min_periods=window // 2).std()
    return (series - m) / (s + 1e-10)

# ─── Feature builders ─────────────────────────────────────────────────────────

def compute_ohlc_features(df):
    """Section 2.2 — OHLC-derived features."""
    eps = 1e-10
    hl  = df["High"] - df["Low"] + eps

    feats = pd.DataFrame(index=df.index)
    feats["log_return"]       = np.log(df["Close"] / df["Close"].shift(1) + eps)
    feats["body_ratio"]       = (df["Close"] - df["Open"]) / hl
    feats["upper_wick_ratio"] = (df["High"] - df[["Open","Close"]].max(axis=1)) / hl
    feats["lower_wick_ratio"] = (df[["Open","Close"]].min(axis=1) - df["Low"]) / hl
    feats["range_zscore"]     = rolling_zscore(hl, ROLLING_WIN)
    vol_mean = df["Volume"].rolling(ROLLING_WIN).mean()
    vol_std  = df["Volume"].rolling(ROLLING_WIN).std()
    feats["volume_zscore"]    = (df["Volume"] - vol_mean) / (vol_std + eps)
    feats["volume_delta"]     = df["Volume"] / (df["Volume"].shift(1) + eps) - 1
    feats["gap"]              = (df["Open"] - df["Close"].shift(1)) / (df["Close"].shift(1) + eps)
    return feats


def compute_technical_features(df):
    """Section 2.3 — Technical indicator features."""
    eps   = 1e-10
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    feats = pd.DataFrame(index=df.index)

    # Volatility
    atr7  = atr(high, low, close, 7)
    atr14 = atr(high, low, close, 14)
    atr28 = atr(high, low, close, 28)
    feats["atr7_norm"]   = atr7  / (close + eps)
    feats["atr14_norm"]  = atr14 / (close + eps)
    feats["atr28_norm"]  = atr28 / (close + eps)
    feats["atr_ratio"]   = atr7  / (atr28 + eps)

    # Momentum
    feats["rsi14"] = rsi(close, 14) / 100
    feats["rsi28"] = rsi(close, 28) / 100

    ema12   = ema(close, 12)
    ema26   = ema(close, 26)
    macd    = ema12 - ema26
    signal  = ema(macd, 9)
    feats["macd_hist"] = (macd - signal) / (atr14 + eps)

    for period in [5, 10, 20, 60]:
        feats[f"roc_{period}"] = np.log(close / (close.shift(period) + eps))

    # Mean reversion
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_range = bb_upper - bb_lower + eps
    feats["bb_position"]   = (close - bb_lower) / bb_range
    feats["dist_sma20"]    = (close - sma20) / (atr14 + eps)
    feats["dist_sma50"]    = (close - close.rolling(50).mean()) / (atr14 + eps)

    # High-low range percentile
    bar_range = high - low
    feats["range_percentile"] = bar_range.rolling(60).rank(pct=True)

    return feats


def compute_calendar_features(df, resolution):
    """Section 2.4 — Calendar and session features."""
    feats = pd.DataFrame(index=df.index)
    idx   = df.index

    # Day of week (0=Mon … 4=Fri)
    dow = idx.dayofweek
    feats["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    feats["dow_cos"] = np.cos(2 * np.pi * dow / 5)

    # Month of year
    month = idx.month
    feats["month_sin"] = np.sin(2 * np.pi * month / 12)
    feats["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Hour of day (intraday only)
    if resolution in ("1H", "4H"):
        hour = idx.hour
        feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)

        # Forex session flags (UTC hours)
        feats["session_asian"]   = ((hour >= 0)  & (hour < 8)).astype(float)
        feats["session_london"]  = ((hour >= 8)  & (hour < 16)).astype(float)
        feats["session_newyork"] = ((hour >= 13) & (hour < 21)).astype(float)
        feats["session_overlap"] = ((hour >= 13) & (hour < 16)).astype(float)

    # Quarter-end flag (within 5 trading days of quarter end)
    quarter_ends = pd.tseries.offsets.QuarterEnd()
    feats["quarter_end_flag"] = 0.0
    for i, dt in enumerate(idx):
        try:
            next_qe = dt + quarter_ends
            days_to_qe = (next_qe - dt).days
            if days_to_qe <= 7:   # ~5 trading days
                feats.iloc[i, feats.columns.get_loc("quarter_end_flag")] = 1.0
        except Exception:
            pass

    return feats


def compute_regime_label(df):
    """Section 1.5 — Regime labelling (trend × volatility = 6 classes)."""
    atr14 = atr(df["High"], df["Low"], df["Close"], 14)
    close = df["Close"]

    # Trend regime over LOOKBACK bars
    price_move  = (close - close.shift(LOOKBACK)).abs()
    trend_thresh = 1.5 * atr14 * np.sqrt(LOOKBACK)
    trend_regime = (price_move > trend_thresh).astype(int)   # 1=trending, 0=range

    # Volatility regime
    atr_pct = atr14 / (close + 1e-10)
    vol_regime = pd.cut(
        atr_pct,
        bins=[-np.inf, 0.01, 0.025, np.inf],
        labels=[0, 1, 2]          # 0=low, 1=normal, 2=high
    ).astype(float)

    # Combined: 0-5 (trend × 3 + vol)
    regime = trend_regime * 3 + vol_regime.fillna(1)
    return regime.rename("regime")


def add_index_correlation(df_instrument, df_index, resolution):
    """Rolling 20-bar correlation of instrument log-return to index log-return."""
    if df_index is None:
        return pd.Series(0.0, index=df_instrument.index, name="corr_index")
    eps    = 1e-10
    r_inst = np.log(df_instrument["Close"] / (df_instrument["Close"].shift(1) + eps))
    r_idx  = np.log(df_index["Close"] / (df_index["Close"].shift(1) + eps))
    r_idx  = r_idx.reindex(df_instrument.index).ffill()
    corr   = r_inst.rolling(20).corr(r_idx).rename("corr_index")
    return corr

# ─── Normalisation pipeline (section 2.5) ────────────────────────────────────

def apply_rolling_zscore_pipeline(feature_df, non_zscore_cols):
    """
    Apply rolling z-score (window=Z_WIN) to all features except calendar features
    which are already normalised. Then clip to [-Z_CLIP, +Z_CLIP].
    """
    result = feature_df.copy()
    for col in feature_df.columns:
        if col in non_zscore_cols:
            continue
        result[col] = rolling_zscore(feature_df[col], Z_WIN).clip(-Z_CLIP, Z_CLIP)
    return result

# ─── Window builder ───────────────────────────────────────────────────────────

def build_windows(feature_df, regime_series):
    """
    Slide a LOOKBACK-bar window over the feature DataFrame.
    Returns:
        X      : np.ndarray [n_windows, LOOKBACK, n_features]
        regimes: np.ndarray [n_windows]        (regime label at window end)
        dates  : list of timestamps             (window end date)
    """
    arr     = feature_df.values.astype(np.float32)
    reg_arr = regime_series.values
    n       = len(arr)
    windows, regimes, dates = [], [], []

    for i in range(LOOKBACK, n):
        window = arr[i - LOOKBACK: i]
        if np.isnan(window).any():
            continue
        windows.append(window)
        regimes.append(reg_arr[i])
        dates.append(feature_df.index[i])

    if not windows:
        return None, None, None

    return np.array(windows), np.array(regimes), dates

# ─── Per-file processor ───────────────────────────────────────────────────────

def process_file(filepath, index_dfs):
    fname      = os.path.basename(filepath)
    parts      = fname.replace(".csv", "").split("_")
    resolution = parts[2] if len(parts) >= 3 else "1D"

    log.info("[PROC] %s", fname)

    # Load
    try:
        df = pd.read_csv(filepath, parse_dates=["Datetime"])
        df = df.set_index("Datetime").sort_index()
    except Exception as e:
        log.error("  ✗  Failed to load %s: %s", fname, e)
        return None

    # Need at least LOOKBACK + Z_WIN bars to produce any samples
    if len(df) < LOOKBACK + Z_WIN + 10:
        log.warning("  ✗  Too few rows (%d) in %s — skipping.", len(df), fname)
        return None

    # Drop rows with zero/NaN close
    df = df[df["Close"] > 0].dropna(subset=["Open","High","Low","Close"])

    # Build features
    ohlc_feats = compute_ohlc_features(df)
    tech_feats = compute_technical_features(df)
    cal_feats  = compute_calendar_features(df, resolution)
    regime     = compute_regime_label(df)

    # Index correlation
    idx_df   = index_dfs.get(resolution)
    corr_col = add_index_correlation(df, idx_df, resolution).to_frame()

    # Combine all features
    all_feats = pd.concat([ohlc_feats, tech_feats, corr_col, cal_feats], axis=1)
    all_feats = all_feats.loc[df.index]   # align

    # Calendar cols are already normalised — skip z-scoring them
    cal_cols       = list(cal_feats.columns)
    non_zscore     = cal_cols + ["bb_position", "rsi14", "rsi28",
                                  "range_percentile", "regime"]

    # Apply rolling z-score pipeline
    feat_normalised = apply_rolling_zscore_pipeline(all_feats, non_zscore_cols=set(cal_cols))
    feat_normalised = feat_normalised.dropna()
    regime_aligned  = regime.reindex(feat_normalised.index)

    # Build sample windows
    X, regimes, dates = build_windows(feat_normalised, regime_aligned)
    if X is None:
        log.warning("  ✗  No valid windows produced for %s", fname)
        return None

    log.info("  ✓  %d windows × %d features", X.shape[0], X.shape[2])

    # Regime class distribution check
    unique, counts = np.unique(regimes[~np.isnan(regimes)], return_counts=True)
    total = len(regimes)
    for cls, cnt in zip(unique, counts):
        pct = 100 * cnt / total
        if pct < 5:
            log.warning("  ⚠  Regime class %d is only %.1f%% of samples", int(cls), pct)

    return {
        "X":           X,
        "regimes":     regimes,
        "dates":       dates,
        "feature_cols": list(feat_normalised.columns),
        "source_file":  fname,
        "resolution":   resolution,
    }

# ─── Main ─────────────────────────────────────────────────────────────────────

def load_index_refs():
    """Load S&P 500 (or fallback) DataFrames for correlation feature."""
    index_dfs = {}
    for res, prefix in INDEX_REFS.items():
        pattern = os.path.join(RAW_DIR, f"*GSPC*{res}*.csv")
        matches = glob.glob(pattern)
        if matches:
            try:
                df = pd.read_csv(matches[0], parse_dates=["Datetime"])
                df = df.set_index("Datetime").sort_index()
                index_dfs[res] = df
                log.info("[IDX]  Loaded index ref for %s: %s", res, os.path.basename(matches[0]))
            except Exception as e:
                log.warning("[IDX]  Could not load index ref for %s: %s", res, e)
    return index_dfs


def main():
    log.info("Stage 2 Feature Engineering — Start")
    log.info("Raw data dir  : %s", os.path.abspath(RAW_DIR))
    log.info("Output dir    : %s", os.path.abspath(PROCESSED_DIR))

    csv_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not csv_files:
        log.error("No CSV files found in %s", RAW_DIR)
        return

    log.info("Found %d CSV files", len(csv_files))

    index_dfs = load_index_refs()

    results    = []
    skipped    = []
    nan_checks = []

    for filepath in csv_files:
        result = process_file(filepath, index_dfs)
        if result is None:
            skipped.append(os.path.basename(filepath))
            continue

        # NaN / Inf check
        nan_count = np.isnan(result["X"]).sum()
        inf_count = np.isinf(result["X"]).sum()
        if nan_count > 0 or inf_count > 0:
            nan_checks.append((result["source_file"], nan_count, inf_count))
            log.warning("  ⚠  NaN/Inf in %s: %d NaN, %d Inf",
                        result["source_file"], nan_count, inf_count)

        # Save as parquet (compressed, fast to load for training)
        out_name = result["source_file"].replace(".csv", ".parquet")
        out_path = os.path.join(PROCESSED_DIR, out_name)

        meta_df = pd.DataFrame({
            "date":    result["dates"],
            "regime":  result["regimes"],
        })

        # Save feature array as numpy + metadata as parquet
        np.save(out_path.replace(".parquet", ".npy"), result["X"])
        meta_df.to_parquet(out_path, index=False)

        results.append(result)

    # ── Summary ──────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Feature engineering complete.")
    log.info("  Processed : %d files", len(results))
    log.info("  Skipped   : %d files", len(skipped))

    if skipped:
        log.info("  Skipped list: %s", skipped)

    if nan_checks:
        log.warning("  Files with NaN/Inf values:")
        for fname, n, i in nan_checks:
            log.warning("    %s — NaN: %d  Inf: %d", fname, n, i)
    else:
        log.info("  ✓  No NaN or Inf values detected in any output array.")

    total_windows = sum(r["X"].shape[0] for r in results)
    log.info("  Total training windows: %d", total_windows)

    # Regime class balance across all data
    all_regimes = np.concatenate([r["regimes"] for r in results])
    valid_reg   = all_regimes[~np.isnan(all_regimes)]
    log.info("\nGlobal regime class distribution:")
    for cls in range(6):
        cnt = (valid_reg == cls).sum()
        pct = 100 * cnt / len(valid_reg) if len(valid_reg) > 0 else 0
        flag = " ⚠  BELOW 5%" if pct < 5 else ""
        log.info("  Class %d: %5d samples  (%5.1f%%)%s", cls, cnt, pct, flag)

    if results:
        sample = results[0]
        log.info("\nFeature columns (%d total):", len(sample["feature_cols"]))
        for col in sample["feature_cols"]:
            log.info("  - %s", col)


if __name__ == "__main__":
    main()