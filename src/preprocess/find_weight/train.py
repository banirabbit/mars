import numpy as np
import pandas as pd
from scipy.optimize import minimize
import wandb


def load_dataset(train_path, val_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    X_train = train_df[["bm25_score", "vector_score"]].values
    y_train = train_df["label"].values

    X_val = val_df[["bm25_score", "vector_score"]].values
    y_val = val_df["label"].values

    return X_train, y_train, X_val, y_val


def train_with_constraint(
    X_train, y_train, X_val, y_val, project_name="retriever_weight_train"
):
    wandb.init(
        project=project_name,
        name="bm25+vector_fusion",
        mode="offline",
        config={
            "method": "constrained_linear_regression",
            "constraint": "w1 + w2 = 1",
        },
    )

    # 约束：w1 + w2 = 1，且 w1, w2 ∈ [0, 1]
    bounds = [(0, 1), (0, 1)]
    constraints = [{"type": "eq", "fun": lambda w: w[0] + w[1] - 1}]

    def loss_fn(weights):
        preds_train = X_train @ weights
        train_loss = np.mean((preds_train - y_train) ** 2)

        preds_val = X_val @ weights
        val_loss = np.mean((preds_val - y_val) ** 2)

        # 记录 wandb 的 loss 曲线
        wandb.log({"train_loss": train_loss, "val_loss": val_loss})
        return train_loss

    init_weights = np.array([0.5, 0.5])
    result = minimize(loss_fn, init_weights, bounds=bounds, constraints=constraints)

    wandb.finish()

    if result.success:
        w1, w2 = result.x
        print("✅ Optimized weights (sum to 1):")
        print(f"  BM25 weight:   {w1:.4f}")
        print(f"  Vector weight: {w2:.4f}")
        return w1, w2
    else:
        raise RuntimeError("❌ Weight optimization failed.")


if __name__ == "__main__":
    # 修改为你的路径
    train_csv = "data/weight/training_data.csv"
    val_csv = "data/weight/validation_data.csv"

    X_train, y_train, X_val, y_val = load_dataset(train_csv, val_csv)
    train_with_constraint(X_train, y_train, X_val, y_val)
