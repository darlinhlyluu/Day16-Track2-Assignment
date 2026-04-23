import argparse
import json
import os
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU fallback benchmark (LightGBM on creditcardfraud).")
    parser.add_argument("--data", default="creditcard.csv", help="Path to creditcard.csv")
    parser.add_argument("--out", default="benchmark_result.json", help="Output JSON path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split from train")
    parser.add_argument("--n-estimators", type=int, default=2000)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--num-threads", type=int, default=0, help="0 = LightGBM default")
    parser.add_argument("--repeat-latency", type=int, default=200)
    parser.add_argument("--repeat-throughput", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"Không thấy file dữ liệu: {args.data}. "
            "Hãy tải dataset Kaggle (mlg-ulb/creditcardfraud) và đảm bảo có creditcard.csv trong thư mục hiện tại."
        )

    import numpy as np
    import pandas as pd
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split

    t0 = time.perf_counter()
    df = pd.read_csv(args.data)
    load_seconds = time.perf_counter() - t0

    if "Class" not in df.columns:
        raise ValueError("Dataset không có cột 'Class' (label).")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=args.val_size, random_state=args.seed, stratify=y_train_full
    )

    n_jobs = None if args.num_threads == 0 else args.num_threads
    model = LGBMClassifier(
        objective="binary",
        n_estimators=args.n_estimators,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=args.seed,
        n_jobs=n_jobs,
    )

    t1 = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[
            early_stopping(args.early_stopping_rounds, verbose=False),
            log_evaluation(period=0),
        ],
    )
    train_seconds = time.perf_counter() - t1

    best_iteration = getattr(model, "best_iteration_", None)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    # Inference latency: average over repeats on 1 row
    x1 = X_test.iloc[[0]]
    repeats = max(1, int(args.repeat_latency))
    t2 = time.perf_counter()
    for _ in range(repeats):
        _ = model.predict_proba(x1)
    latency_seconds = (time.perf_counter() - t2) / repeats

    # Throughput: rows/sec for 1000 rows batch (repeat to smooth)
    batch = X_test.iloc[:1000] if len(X_test) >= 1000 else X_test
    rep_tput = max(1, int(args.repeat_throughput))
    t3 = time.perf_counter()
    for _ in range(rep_tput):
        _ = model.predict_proba(batch)
    tput_seconds = (time.perf_counter() - t3) / rep_tput
    throughput_rows_per_sec = float(len(batch) / tput_seconds) if tput_seconds > 0 else float("inf")

    result = {
        "dataset": os.path.basename(args.data),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "load_seconds": load_seconds,
        "train_seconds": train_seconds,
        "best_iteration": int(best_iteration) if best_iteration is not None else None,
        "auc_roc": float(auc),
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "inference_latency_seconds_1_row": float(latency_seconds),
        "inference_throughput_rows_per_sec_batch": float(throughput_rows_per_sec),
        "batch_size": int(len(batch)),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Compact console output for screenshot
    print("=== LightGBM CPU Benchmark (creditcardfraud) ===")
    print(f"load_seconds: {result['load_seconds']:.3f}")
    print(f"train_seconds: {result['train_seconds']:.3f}")
    print(f"best_iteration: {result['best_iteration']}")
    print(f"auc_roc: {result['auc_roc']:.6f}")
    print(f"accuracy: {result['accuracy']:.6f}")
    print(f"f1_score: {result['f1_score']:.6f}")
    print(f"precision: {result['precision']:.6f}")
    print(f"recall: {result['recall']:.6f}")
    print(f"inference_latency_seconds_1_row: {result['inference_latency_seconds_1_row']:.6f}")
    print(
        "inference_throughput_rows_per_sec_batch:"
        f" {result['inference_throughput_rows_per_sec_batch']:.2f} (batch_size={result['batch_size']})"
    )
    print(f"saved: {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

