"""ЛР2, пункт 1: одноразове навчання заданої MLP і збереження прогнозів."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
BATCH_SIZE = 1024
EPOCHS = 12
LR = 1e-3


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Не знайдено {path}. Завантажте оригінальний creditcard.csv з "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
        )
    df = pd.read_csv(path)  
    expected = {"Time", "Class", *FEATURES}
    if set(df.columns) != expected:
        raise ValueError(f"Неправильні поля CSV: відсутні {expected-set(df.columns)}, "
                         f"зайві {set(df.columns)-expected}")
    if len(df) != 284807 or int((df["Class"] == 1).sum()) != 492:
        raise ValueError("Потрібен оригінальний датасет creditcard.csv: 284807 рядків")
    if not df["Class"].isin([0, 1]).all():
        raise ValueError("Class має складатися виключно з 0 та 1")
    if df[FEATURES].isna().any().any() or not np.isfinite(df[FEATURES].to_numpy()).all():
        raise ValueError("Ознаки містять NaN/inf")
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True, help="Шлях до локального creditcard.csv")
    p.add_argument("--output", type=Path, default=Path("results"), help="Каталог артефактів")
    p.add_argument("--overwrite", action="store_true", help="Дозволити заміну попередніх результатів навчання")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "predictions.npz").exists() and not args.overwrite:
        raise FileExistsError("Прогнози вже існують. Для нового навчання явно вкажіть --overwrite")

    df = load_dataset(args.csv)
    labels = df["Class"].to_numpy(dtype=np.int64)
    amounts = df["Amount"].to_numpy(dtype=np.float64, copy=True)  # Сирий Amount 
    features = df[FEATURES].to_numpy(dtype=np.float64, copy=True)
    row_ids = np.arange(len(df), dtype=np.int64)

    train_idx, heldout_idx = train_test_split(
        row_ids, test_size=0.4, stratify=labels, random_state=0
    )
    val_idx, test_idx = train_test_split(
        heldout_idx, test_size=0.5, stratify=labels[heldout_idx], random_state=0
    )
    
    train_idx, val_idx, test_idx = map(np.sort, (train_idx, val_idx, test_idx))
    assert len(set(np.concatenate([train_idx, val_idx, test_idx]).tolist())) == len(df)
    np.savez_compressed(args.output / "split_indices.npz", train=train_idx,
                        validation=val_idx, test=test_idx)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(features[train_idx]).astype(np.float32)
    x_val = scaler.transform(features[val_idx]).astype(np.float32)
    x_test = scaler.transform(features[test_idx]).astype(np.float32)
    np.savez_compressed(args.output / "scaler.npz", mean=scaler.mean_, var=scaler.var_,
                        scale=scaler.scale_, n_samples_seen=scaler.n_samples_seen_)

    torch.manual_seed(0)  
    model = torch.nn.Sequential(
        torch.nn.Linear(29, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)
    ).to(device="cpu", dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  

    xtr = torch.from_numpy(x_train)
    ytr = torch.from_numpy(labels[train_idx].astype(np.float32)).reshape(-1, 1)
    rng = np.random.default_rng(0)
    history = []
    model.train()
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(train_idx))  
        total_loss = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            ids = torch.from_numpy(order[start:start + BATCH_SIZE])
            xb = xtr[ids]
            yb = ytr[ids]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(ids)
        avg_loss = total_loss / len(train_idx)
        history.append({"epoch": epoch, "train_BCE": avg_loss})
        print(f"Епоха {epoch:02d}/{EPOCHS}: train BCE = {avg_loss:.8f}", flush=True)
    pd.DataFrame(history).to_csv(args.output / "training_log.csv", index=False)
    torch.save(model.state_dict(), args.output / "model_weights.pt")

    model.eval()
    with torch.no_grad():
        val_scores = torch.sigmoid(model(torch.from_numpy(x_val))).reshape(-1).numpy().copy()
        test_scores = torch.sigmoid(model(torch.from_numpy(x_test))).reshape(-1).numpy().copy()
    assert val_scores.dtype == np.float32 and test_scores.dtype == np.float32
    np.savez_compressed(
        args.output / "predictions.npz",
        validation_rows=val_idx, validation_labels=labels[val_idx].astype(np.uint8),
        validation_scores=val_scores, validation_amount=amounts[val_idx],
        test_rows=test_idx, test_labels=labels[test_idx].astype(np.uint8),
        test_scores=test_scores, test_amount=amounts[test_idx],
    )
    stats = {key: {"n": int(len(idx)), "fraud": int(labels[idx].sum()),
                   "fraud_rate": float(labels[idx].mean())}
             for key, idx in (("train", train_idx), ("validation", val_idx), ("test", test_idx))}
    metadata = {"dataset": "Credit Card Fraud Detection (ULB)", "num_rows": len(df),
                "num_fraud": int(labels.sum()), "features": FEATURES, "excluded": ["Time", "Class"],
                "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LR,
                "seed_torch": 0, "seed_numpy_rng": 0, "device": "cpu", "dtype": "float32",
                "splits": stats, "final_train_bce": history[-1]["train_BCE"]}
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Модель та необроблені оцінки збережено у", args.output)
    print("Далі: uv run --python 3.12 --with-requirements requirements.txt python analyze.py --artifacts results")


if __name__ == "__main__":
    main()
