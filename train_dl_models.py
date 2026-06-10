import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

import config
from core.feature_engine import FeatureEngine
from core.dl_dataset import SequenceDataset
from core.ml_models import TradingLSTM, TradingCNN1D, TradingTransformer


def train_model(model, train_loader, val_loader, device,
                model_name="model", epochs=40, label_smoothing=0.1):
    print(f"\n--- Training {model_name} on {device} ---")
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)

    # Cosine Annealing with Warm Restarts — much better than ReduceLROnPlateau for financial data
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15   # early stopping patience for DL

    model_dir = config.MODEL_DIR if hasattr(config, 'MODEL_DIR') else "models"
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, f"{model_name}_v4.pt")

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            if isinstance(model, TradingCNN1D):
                batch_x = batch_x.permute(0, 2, 1)   # → (B, F, T)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss    = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            _, pred     = torch.max(outputs.data, 1)
            total      += batch_y.size(0)
            correct    += (pred == batch_y).sum().item()

        scheduler.step(epoch + 1)

        train_loss /= len(train_loader.dataset)
        train_acc   = correct / total

        # ── Validate ──
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                if isinstance(model, TradingCNN1D):
                    batch_x = batch_x.permute(0, 2, 1)

                outputs   = model(batch_x)
                loss      = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                _, pred   = torch.max(outputs.data, 1)
                total    += batch_y.size(0)
                correct  += (pred == batch_y).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc   = correct / total

        lr_current = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1:3d}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {lr_current:.6f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Best model saved (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  -> Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Finished training {model_name}. Best val_loss: {best_val_loss:.4f}")


def main():
    print("=== AI Trading v4.2 Deep Learning Pipeline ===")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("Using device: CPU")

    # 1. Load data
    data_dir = config.DATA_DIR if hasattr(config, 'DATA_DIR') else "data"
    try:
        df_5m = pd.read_csv(os.path.join(data_dir, "raw_5m.csv"))
        df_1h = pd.read_csv(os.path.join(data_dir, "raw_1h.csv"))
        df_1d = pd.read_csv(os.path.join(data_dir, "raw_1d.csv"))
    except FileNotFoundError:
        print("Data not found. Run train_models.py first.")
        return

    # 2. Feature Engineering
    print("Running Feature Engineering v4.2...")
    fe         = FeatureEngine()
    merged_df  = fe.merge_multi_tf(df_5m, df_1h, df_1d)
    labeled_df = fe.create_labeled_dataset(merged_df)

    labeled_df = labeled_df[labeled_df['label_long_aln'] != -1]
    labeled_df['target'] = 0
    labeled_df.loc[labeled_df['label_long_aln']  == 1, 'target'] = 1
    labeled_df.loc[labeled_df['label_short_aln'] == 1, 'target'] = 2

    feature_cols = fe._feature_cols
    labeled_df.dropna(subset=feature_cols, inplace=True)
    labeled_df.sort_values('timestamp', inplace=True)
    labeled_df.reset_index(drop=True, inplace=True)

    # 3. Chronological split (no shuffle to prevent data leakage)
    train_size = int(len(labeled_df) * 0.8)
    df_train   = labeled_df.iloc[:train_size]
    df_val     = labeled_df.iloc[train_size:]

    # 4. Sequence datasets
    seq_len = 60
    print(f"Creating Sequence Datasets (Seq Len: {seq_len})...")
    train_dataset = SequenceDataset(df_train[feature_cols], df_train['target'], seq_len=seq_len, is_train=True)
    val_dataset   = SequenceDataset(df_val[feature_cols],   df_val['target'],   seq_len=seq_len,
                                    scaler=train_dataset.scaler, is_train=False)

    batch_size    = 256
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader    = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    num_features = len(feature_cols)

    # 5. Train all three models
    lstm_model = TradingLSTM(input_size=num_features)
    train_model(lstm_model, train_loader, val_loader, device,
                model_name="trading_lstm", epochs=60, label_smoothing=0.1)

    cnn_model = TradingCNN1D(in_channels=num_features, seq_len=seq_len)
    train_model(cnn_model, train_loader, val_loader, device,
                model_name="trading_cnn1d", epochs=60, label_smoothing=0.1)

    transformer_model = TradingTransformer(input_size=num_features)
    train_model(transformer_model, train_loader, val_loader, device,
                model_name="trading_transformer", epochs=60, label_smoothing=0.1)


if __name__ == "__main__":
    main()
