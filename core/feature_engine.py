import numpy as np
import pandas as pd

class FeatureEngine:
    """
    Feature Engineering Engine v4.2
    Implements indicators defined in INDICATORS.md:
    - Trend: EMA (TF-specific), Linear Regression, Swing Structure
    - Momentum: RSI (with zone), MACD
    - Volatility: ATR, Bollinger Bands
    - Volume/Flow: Volume spike, CVD proxy
    - Aggregator: Multi-TF Confluence Score
    """
    def __init__(self):
        self._feature_cols = []

    # ==========================================
    # LABELING STRATEGY
    # ==========================================
    def label_trades(self, df: pd.DataFrame, sl_pct: float, tp_pct: float, max_hold: int, direction: str = "LONG") -> pd.Series:
        labels = []
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        n = len(df)

        for i in range(n - max_hold):
            entry = closes[i]
            label = -1
            
            if direction == "LONG":
                sl = entry * (1 - sl_pct)
                tp = entry * (1 + tp_pct)
                for j in range(i + 1, i + max_hold + 1):
                    if lows[j] <= sl:
                        label = 0   # LOSS
                        break
                    if highs[j] >= tp:
                        label = 1   # WIN
                        break
            else:  # SHORT
                sl = entry * (1 + sl_pct)
                tp = entry * (1 - tp_pct)
                for j in range(i + 1, i + max_hold + 1):
                    if highs[j] >= sl:
                        label = 0   # LOSS
                        break
                    if lows[j] <= tp:
                        label = 1   # WIN
                        break
            labels.append(label)

        labels.extend([-1] * max_hold)
        return pd.Series(labels, index=df.index)

    def create_labeled_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Opportunistic
        df["label_long_opp"]  = self.label_trades(df, 0.0012, 0.0020, 24, "LONG")
        df["label_short_opp"] = self.label_trades(df, 0.0012, 0.0020, 24, "SHORT")
        # Aligned
        df["label_long_aln"]  = self.label_trades(df, 0.0015, 0.0040, 72, "LONG")
        df["label_short_aln"] = self.label_trades(df, 0.0015, 0.0040, 72, "SHORT")
        # Conviction
        df["label_long_con"]  = self.label_trades(df, 0.0015, 0.0080, 288, "LONG")
        df["label_short_con"] = self.label_trades(df, 0.0015, 0.0080, 288, "SHORT")
        return df

    # ==========================================
    # FEATURE ENGINEERING
    # ==========================================
    def compute_all_features(self, df: pd.DataFrame, tf_suffix: str = "5m") -> pd.DataFrame:
        df = df.copy()
        
        # ── 1. EMA (TF-specific periods) ──
        p1, p2, p3 = 20, 50, 200
        if tf_suffix in ["1w", "1d"]:
            p1, p2, p3 = 13, 144, 377
        elif tf_suffix in ["4h", "1h"]:
            p1, p2, p3 = 13, 34, 144
        elif tf_suffix == "15m":
            p1, p2, p3 = 8, 21, 55
        elif tf_suffix == "5m":
            p1, p2, p3 = 5, 13, 34

        ema1 = df['close'].ewm(span=p1, adjust=False).mean()
        ema2 = df['close'].ewm(span=p2, adjust=False).mean()
        ema3 = df['close'].ewm(span=p3, adjust=False).mean()

        df[f'price_vs_ema{p1}_{tf_suffix}'] = (df['close'] - ema1) / ema1
        df[f'price_vs_ema{p2}_{tf_suffix}'] = (df['close'] - ema2) / ema2
        df[f'price_vs_ema{p3}_{tf_suffix}'] = (df['close'] - ema3) / ema3
        df[f'ema{p1}_vs_ema{p2}_{tf_suffix}'] = (ema1 - ema2) / ema2
        df[f'ema{p2}_vs_ema{p3}_{tf_suffix}'] = (ema2 - ema3) / ema3
        df[f'ema_align_score_{tf_suffix}'] = (
            (ema1 > ema2).astype(int) +
            (ema2 > ema3).astype(int) +
            (df['close'] > ema1).astype(int)
        )
        df[f'ema_main_slope_{tf_suffix}'] = ema1.diff(5) / 5 / ema1 * 100

        # ── 2. Linear Regression (slope + channel position) ──
        period_lr = 20
        x = np.arange(period_lr)
        x_mean = x.mean()
        divisor = np.sum((x - x_mean) ** 2)

        def calc_slope(y):
            if len(y) < period_lr: return 0.0
            y_mean = y.mean()
            return np.sum((x - x_mean) * (y - y_mean)) / divisor

        def calc_residual(y):
            if len(y) < period_lr: return 0.0
            y_mean = y.mean()
            slope = np.sum((x - x_mean) * (y - y_mean)) / divisor
            intercept = y_mean - slope * x_mean
            y_hat_last = intercept + slope * (period_lr - 1)
            return (y[-1] - y_hat_last) / (y_hat_last + 1e-9) * 100

        def calc_r2(y):
            if len(y) < period_lr: return 0.0
            y_mean = y.mean()
            slope = np.sum((x - x_mean) * (y - y_mean)) / divisor
            intercept = y_mean - slope * x_mean
            y_hat = intercept + slope * x
            sse = np.sum((y - y_hat) ** 2)
            sst = np.sum((y - y_mean) ** 2)
            return 1 - sse / sst if sst > 0 else 0.0

        df[f'lr_slope_{tf_suffix}']    = df['close'].rolling(period_lr).apply(calc_slope, raw=True).fillna(0)
        df[f'lr_residual_{tf_suffix}'] = df['close'].rolling(period_lr).apply(calc_residual, raw=True).fillna(0)
        df[f'lr_r2_{tf_suffix}']       = df['close'].rolling(period_lr).apply(calc_r2, raw=True).fillna(0)
        # Normalize slope by price for cross-asset comparability
        df[f'lr_slope_norm_{tf_suffix}'] = df[f'lr_slope_{tf_suffix}'] / df['close'] * 100

        # ── 3. ATR & Bollinger Bands ──
        df['_atr'] = self._calculate_atr(df, 14)
        df[f'atr_{tf_suffix}']     = df['_atr']
        df[f'atr_pct_{tf_suffix}'] = df['_atr'] / df['close'] * 100

        bb_mid = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_range = (bb_upper - bb_lower).replace(0, 1e-9)

        df[f'bb_width_{tf_suffix}']    = (bb_upper - bb_lower) / bb_mid * 100
        df[f'bb_position_{tf_suffix}'] = (df['close'] - bb_lower) / bb_range  # 0=lower, 1=upper
        df[f'bb_pct_b_{tf_suffix}']    = (df['close'] - bb_lower) / bb_range  # alias for clarity
        df[f'bb_squeeze_{tf_suffix}']  = (df[f'bb_width_{tf_suffix}'] < df[f'bb_width_{tf_suffix}'].rolling(20).quantile(0.2)).astype(int)
        df[f'bb_expansion_{tf_suffix}'] = (df[f'bb_width_{tf_suffix}'] > df[f'bb_width_{tf_suffix}'].rolling(20).quantile(0.8)).astype(int)

        # ── 4. RSI with Zone encoding ──
        rsi = self._calculate_rsi(df['close'], 14)
        df[f'rsi_{tf_suffix}']       = rsi
        df[f'rsi_slope_{tf_suffix}'] = rsi.diff(5) / 5
        df[f'rsi_zone_{tf_suffix}']  = pd.cut(rsi, bins=[-1, 30, 45, 55, 70, 101],
                                               labels=[0, 1, 2, 3, 4]).astype(float).fillna(2)
        # RSI divergence proxy: RSI slope vs price slope diverge
        price_slope = df['close'].diff(5)
        df[f'rsi_div_proxy_{tf_suffix}'] = (np.sign(rsi.diff(5)) != np.sign(price_slope)).astype(int)

        # ── 5. MACD ──
        ema_fast   = df['close'].ewm(span=12, adjust=False).mean()
        ema_slow   = df['close'].ewm(span=26, adjust=False).mean()
        macd_line  = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist  = macd_line - signal_line

        df[f'macd_hist_{tf_suffix}']     = macd_hist
        df[f'macd_hist_norm_{tf_suffix}'] = macd_hist / df['close'] * 100  # normalized
        hist_prev = macd_hist.shift(1)
        df[f'macd_cross_{tf_suffix}'] = 0
        df.loc[(hist_prev < 0) & (macd_hist > 0), f'macd_cross_{tf_suffix}'] = 1
        df.loc[(hist_prev > 0) & (macd_hist < 0), f'macd_cross_{tf_suffix}'] = -1
        df[f'macd_expanding_{tf_suffix}'] = (macd_hist.abs() > macd_hist.abs().shift(1)).astype(int)

        # ── 6. Volume & CVD proxy ──
        vol_ma = df['volume'].rolling(20).mean().replace(0, 1e-9)
        df[f'volume_ratio_{tf_suffix}'] = df['volume'] / vol_ma
        df[f'volume_spike_{tf_suffix}'] = (df[f'volume_ratio_{tf_suffix}'] > 2.0).astype(int)
        # CVD proxy: green candle = buying, red = selling; cumulative delta normalized
        candle_dir = np.sign(df['close'] - df['open'])
        buy_vol    = df['volume'] * ((candle_dir + 1) / 2)
        sell_vol   = df['volume'] * ((1 - candle_dir) / 2)
        delta      = buy_vol - sell_vol
        df[f'cvd_slope_{tf_suffix}'] = delta.rolling(10).sum() / vol_ma  # normalized CVD momentum

        # ── 7. Candle Properties ──
        body      = (df['close'] - df['open']).abs()
        wick_top  = df['high'] - df[['close', 'open']].max(axis=1)
        wick_bot  = df[['close', 'open']].min(axis=1) - df['low']
        total_range = (df['high'] - df['low']).replace(0, 1e-9)

        df[f'body_ratio_{tf_suffix}']       = body / total_range          # 0=doji, 1=full body
        df[f'wick_top_ratio_{tf_suffix}']   = wick_top / total_range      # upper wick dominance
        df[f'wick_bot_ratio_{tf_suffix}']   = wick_bot / total_range      # lower wick dominance
        df[f'candle_direction_{tf_suffix}'] = candle_dir
        df[f'body_vs_atr_{tf_suffix}']      = body / df['_atr'].replace(0, 1e-9)

        # Cleanup internal helpers
        internal = ['_atr']
        df.drop(columns=[c for c in internal if c in df.columns], inplace=True)

        # Track features
        exclude = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
        new_cols = [c for c in df.columns if c not in exclude]
        self._feature_cols.extend([c for c in new_cols if c not in self._feature_cols])

        return df.fillna(0)

    def merge_multi_tf(self, df_5m: pd.DataFrame, df_1h: pd.DataFrame, df_1d: pd.DataFrame) -> pd.DataFrame:
        """Aligns higher timeframe features to the 5M dataset using ffill."""
        for df in [df_5m, df_1h, df_1d]:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        df_1h_feat = self.compute_all_features(df_1h, "1h")
        df_1d_feat = self.compute_all_features(df_1d, "1d")
        df_5m_feat = self.compute_all_features(df_5m, "5m")

        df_1h_feat.set_index('timestamp', inplace=True)
        df_1d_feat.set_index('timestamp', inplace=True)
        df_5m_feat.set_index('timestamp', inplace=True)

        htf_cols_to_drop = ['open', 'high', 'low', 'close', 'volume']
        df_1h_feat.drop(columns=[c for c in htf_cols_to_drop if c in df_1h_feat.columns], inplace=True)
        df_1d_feat.drop(columns=[c for c in htf_cols_to_drop if c in df_1d_feat.columns], inplace=True)

        merged = df_5m_feat.join(df_1h_feat, how='left')
        merged = merged.join(df_1d_feat, how='left')
        merged.ffill(inplace=True)
        merged.reset_index(inplace=True)

        # Multi-TF Confluence Score
        merged['confluence_score'] = (
            np.sign(merged.get('ema_main_slope_5m', 0)) * 0.2 +
            np.sign(merged.get('ema_main_slope_1h', 0)) * 0.4 +
            np.sign(merged.get('ema_main_slope_1d', 0)) * 0.4
        )
        # EMA alignment across TFs
        merged['tf_align_score'] = (
            merged.get('ema_align_score_5m', 0) +
            merged.get('ema_align_score_1h', 0) +
            merged.get('ema_align_score_1d', 0)
        ) / 9.0  # normalize to 0-1

        for col in ['confluence_score', 'tf_align_score']:
            if col not in self._feature_cols:
                self._feature_cols.append(col)

        return merged

    # ── Internal Helpers ──
    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain  = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs    = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        prev_close = df['close'].shift(1)
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - prev_close).abs(),
            (df['low']  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, adjust=False).mean()
