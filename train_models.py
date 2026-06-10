import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch

import config
from data_pipeline.binance_data_fetcher import BinanceDataFetcher
from core.feature_engine import FeatureEngine


def fetch_data_if_needed(days=365, force_refetch=False):
    fetcher = BinanceDataFetcher()
    data_dir = config.DATA_DIR if hasattr(config, 'DATA_DIR') else "data"

    file_5m = os.path.join(data_dir, "raw_5m.csv")
    file_1h = os.path.join(data_dir, "raw_1h.csv")
    file_1d = os.path.join(data_dir, "raw_1d.csv")

    if force_refetch or not (os.path.exists(file_5m) and os.path.exists(file_1h) and os.path.exists(file_1d)):
        print(f"Fetching {days} days of data from Binance...")
        fetcher.fetch_historical_data("1d", days=days)
        fetcher.fetch_historical_data("1h", days=days)
        fetcher.fetch_historical_data("5m", days=days)

    return pd.read_csv(file_5m), pd.read_csv(file_1h), pd.read_csv(file_1d)


def prepare_dataset(df_5m, df_1h, df_1d):
    print("Running Feature Engineering v4.2...")
    fe = FeatureEngine()
    merged_df   = fe.merge_multi_tf(df_5m, df_1h, df_1d)

    print("Applying Forward-Looking Labeling...")
    labeled_df  = fe.create_labeled_dataset(merged_df)
    labeled_df  = labeled_df[labeled_df['label_long_aln'] != -1]

    labeled_df['target'] = 0
    labeled_df.loc[labeled_df['label_long_aln']  == 1, 'target'] = 1
    labeled_df.loc[labeled_df['label_short_aln'] == 1, 'target'] = 2

    feature_cols = fe._feature_cols
    labeled_df.dropna(subset=feature_cols, inplace=True)

    X = labeled_df[feature_cols]
    y = labeled_df['target']
    return X, y, feature_cols


def train_xgboost(X, y, feature_cols):
    n_samples = len(X)
    n_features = len(feature_cols)
    print(f"Training XGBoost ensemble on {n_samples} samples with {n_features} features...")

    # Detect GPU
    use_gpu = torch.cuda.is_available()
    device_str = "cuda" if use_gpu else "cpu"
    tree_method = "hist"
    print(f"XGBoost device: {device_str} ({'GPU accelerated' if use_gpu else 'CPU'})")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    # Shared regularization base
    shared = dict(
        objective="multi:softprob",
        num_class=3,
        tree_method=tree_method,
        device=device_str,
        random_state=42,
        n_estimators=1500,          # high ceiling, early stopping will cut it
        early_stopping_rounds=75,   # stop if no improvement for 75 rounds
        eval_metric="mlogloss",
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
    )

    models_cfg = {
        "conservative": dict(max_depth=4, learning_rate=0.02, reg_alpha=0.1, reg_lambda=2.0),
        "balanced":     dict(max_depth=5, learning_rate=0.04, reg_alpha=0.05, reg_lambda=1.0),
        "aggressive":   dict(max_depth=6, learning_rate=0.06, reg_alpha=0.01, reg_lambda=0.5),
    }

    model_dir = config.MODEL_DIR if hasattr(config, 'MODEL_DIR') else "models"
    os.makedirs(model_dir, exist_ok=True)

    results = {}
    for name, extra_cfg in models_cfg.items():
        print(f"\n--- Training {name.upper()} Model ---")
        params = {**shared, **extra_cfg}
        model  = xgb.XGBClassifier(**params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100           # print every 100 rounds
        )

        best_iter = model.best_iteration
        y_pred = model.predict(X_val)
        print(classification_report(y_val, y_pred, zero_division=0))
        print(f"Best iteration: {best_iter}")

        model_path = os.path.join(model_dir, f"xgb_{name}_v4.json")
        model.save_model(model_path)
        print(f"Saved {name} model to {model_path}")
        results[name] = model

    return results


def main():
    print("=== AI Trading v4.2 ML Training Pipeline ===")

    # 1. Data Fetching
    df_5m, df_1h, df_1d = fetch_data_if_needed(days=365, force_refetch=False)

    # 2. Feature Engineering & Labeling
    X, y, feature_cols = prepare_dataset(df_5m, df_1h, df_1d)

    if len(X) < 1000:
        print("Error: Dataset too small after processing. Cannot train.")
        return

    # 3. Model Training
    train_xgboost(X, y, feature_cols)

    # Save feature column list — used by main.py inference to align features
    _model_dir = config.MODEL_DIR if hasattr(config, "MODEL_DIR") else "models"
    os.makedirs(_model_dir, exist_ok=True)
    feature_cols_path = os.path.join(_model_dir, "feature_columns.json")
    with open(feature_cols_path, "w") as f:
        json.dump(X.columns.tolist(), f, indent=2)
    print(f"Feature columns ({len(X.columns)}) saved to {feature_cols_path}")

    print("\nTraining Pipeline Complete.")


if __name__ == "__main__":
    main()
