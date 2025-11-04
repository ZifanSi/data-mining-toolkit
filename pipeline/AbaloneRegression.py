import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import random
# [1] AI prompt: General workflow
# Steps:
#   1. Visualize features vs Age (linear vs polynomial).
#   2. Select cost function (RMSE/MSE/MAE).
#   3. Choose approach (OLS or GD).
#   4. Train on subset, evaluate on held-out set.
CSV_PATH = os.path.join("datasets", "training_data.csv")

PLOTS_DIR_STEP1 = os.path.join("outputs", "part2_plots_step1")
os.makedirs(PLOTS_DIR_STEP1, exist_ok=True)

PRIMARY_COST = "RMSE"
APPROACH = "ols"
MODEL_TYPE = "linear"
POLY_DEGREE = 2

RANDOM_SEED = 42
TEST_SIZE = 0.2

GD_LR = 0.05
GD_EPOCHS = 2000
GD_TOL = 1e-7
GD_STANDARDIZE = True

PLOTS_DIR_STEP4 = os.path.join("outputs", "part2_plots_step4")
os.makedirs(PLOTS_DIR_STEP4, exist_ok=True)

FEATURES = [
    "Length",
    "Diameter",
    "Height",
    "Whole_weight",
    "Shucked_weight",
    "Viscera_weight",
    "Shell_weight",
]
TARGET = "Rings"
AGE_OFFSET = 1.5

# [2] AI prompt: Bug fix — Load + clean dataset
def load_df(path):
    df = pd.read_csv(path)
    drop_cols = [c for c in df.columns if c.strip().lower().startswith("unnamed") or c.strip() == ""]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    need = FEATURES + [TARGET]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")
    df = df.dropna(subset=need).copy()
    for c in need:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=need).copy()
    df["Age"] = df[TARGET] + AGE_OFFSET
    return df

# [3] AI prompt: Split data into train/valid sets by shuffling and ratio
def train_valid_split(X, y, test_size=0.2, seed=42):
    n = X.shape[0]
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    cut = int(round(n * (1.0 - test_size)))
    tr, va = idx[:cut], idx[cut:]
    return X[tr], X[va], y[tr], y[va]

# [4] AI prompt: Compute R² from scratch
def r2_score_np(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0

# [5] AI prompt: Compute regression metrics (RMSE, MSE, MAE, R²)
def compute_metrics(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0
    return {"RMSE": rmse, "MSE": mse, "MAE": mae, "R2": r2}

# [6] AI prompt: Add intercept column of 1s,fix error
def add_intercept(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

# [7] AI prompt: Bug fix — Fit OLS regression with pseudo-inverse, use existing libraries
def fit_ols(X, y):
    XtX = X.T @ X
    Xty = X.T @ y
    return np.linalg.pinv(XtX) @ Xty

# [8] AI prompt: Bug fix — Fit regression via Gradient Descent, error fix
def fit_gd(X, y, lr=0.01, epochs=1000, tol=1e-7):
    n, d = X.shape
    beta = np.zeros((d, 1))
    prev = 1e99
    for it in range(epochs):
        yhat = X @ beta
        grad = (2.0 / n) * (X.T @ (yhat - y))
        beta -= lr * grad
        mse = float(np.mean((yhat - y) ** 2))
        if abs(prev - mse) < tol:
            break
        prev = mse
    return beta

# [9] AI prompt: Standardization — fit (return z, μ, σ)
def standardize_fit(X):
    mu = np.mean(X, axis=0, keepdims=True)
    sd = np.std(X, axis=0, ddof=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd, mu, sd

def standardize_apply(X, mu, sd):
    return (X - mu) / sd

# [10] AI prompt: Bug fix — Build polynomial features up to given degree
def build_poly_features(X_base, names, degree):
    n, p = X_base.shape
    cols = [X_base[:, [j]] for j in range(p)]
    col_names = [names[j] for j in range(p)]

    if degree >= 2:
        def combos_with_repl(p, d, start=0, prefix=()):
            if d == 0:
                yield prefix
            else:
                for j in range(start, p):
                    yield from combos_with_repl(p, d-1, j, prefix + (j,))
        for d in range(2, degree + 1):
            for idxs in combos_with_repl(p, d):
                v = np.ones((n, 1))
                counts = {}
                for j in idxs:
                    v *= X_base[:, [j]]
                    counts[j] = counts.get(j, 0) + 1
                parts = []
                for j in sorted(counts.keys()):
                    c = counts[j]
                    parts.append(f"{names[j]}^{c}" if c > 1 else names[j])
                cols.append(v)
                col_names.append("*".join(parts))

    X_poly = np.hstack(cols) if cols else np.zeros((n, 0))
    return X_poly, col_names


# [10] AI prompt: Bug fix — Build polynomial features up to given degree
def step1_visualize():
    df = load_df(CSV_PATH)
    print(f"step 1: loaded {len(df)} rows; saving plots to {os.path.abspath(PLOTS_DIR_STEP1)}")
    print(f"step 1: Age = Rings + 1.5 (mean Age ≈ {df['Age'].mean():.3f})")

    for feat in FEATURES:
        x = df[feat].to_numpy(dtype=float)
        y = df["Age"].to_numpy(dtype=float)

        coef_lin = np.polyfit(x, y, deg=1)
        y_lin = np.polyval(coef_lin, x)
        r2_lin = r2_score_np(y, y_lin)

        coef_quad = np.polyfit(x, y, deg=2)
        y_quad = np.polyval(coef_quad, x)
        r2_quad = r2_score_np(y, y_quad)
        #[11] AI prompt: Fix Compile Erros
        dR2 = r2_quad - r2_lin
        relation = "likely linear" if dR2 <= 0.02 else "likely polynomial (deg≈2)"

        xs = np.linspace(x.min(), x.max(), 200)
        ys_lin = np.polyval(coef_lin, xs)
        ys_quad = np.polyval(coef_quad, xs)

        #[12] AI prompt: Fix Compile Erros
        plt.figure(figsize=(6, 4))
        sn.scatterplot(x=x, y=y, s=12, alpha=0.6, edgecolor=None)
        plt.plot(xs, ys_lin, linewidth=2, label=f"Linear fit (R²={r2_lin:.3f})")
        plt.plot(xs, ys_quad, linewidth=2, linestyle="--", label=f"Quadratic fit (R²={r2_quad:.3f})")
        plt.xlabel(feat)
        plt.ylabel("Age (years)")
        plt.title(f"{feat} vs Age")
        plt.legend()
        out = os.path.join(PLOTS_DIR_STEP1, f"{feat}_vs_Age.png")
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        plt.close()

        print(f"step 1: {feat:<15s} → {relation}; ΔR²={dR2:+.3f}; plot={out}")

# [13] AI prompt: Step 2 — Select cost metric (report RMSE/MSE/MAE/R², pick primary)
def step2_select_cost_demo(y_true, y_pred):
    m = compute_metrics(y_true, y_pred)
    print(f"step 2: RMSE={m['RMSE']:.4f}, MSE={m['MSE']:.4f}, MAE={m['MAE']:.4f}, R²={m['R2']:.4f}")
    chosen = PRIMARY_COST if PRIMARY_COST in ("RMSE", "MSE", "MAE") else "RMSE"
    print(f"step 2: primary cost = {chosen} → {m[chosen]:.4f}")
    return m

# [14] AI prompt: Step 3+4 — Train (OLS/GD), print β, evaluate on held-out set, plot residuals
def step3_and_4_train_evaluate():
    random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

    df = load_df(CSV_PATH)
    X0 = df[FEATURES].to_numpy(dtype=float)
    y  = df[TARGET].to_numpy(dtype=float).reshape(-1, 1)

    if MODEL_TYPE == "linear":
        Xf, names = X0, FEATURES[:]
    else:
        Xf, names = build_poly_features(X0, FEATURES, POLY_DEGREE)

    X_tr_raw, X_va_raw, y_tr, y_va = train_valid_split(Xf, y, TEST_SIZE, RANDOM_SEED)
    print(f"step 3: model = {MODEL_TYPE if MODEL_TYPE=='linear' else f'polynomial (deg={POLY_DEGREE})'}")

    scaled_flag = False
    if APPROACH == "gd" and GD_STANDARDIZE:
        X_tr_z, mu, sd = standardize_fit(X_tr_raw)
        X_va_z = standardize_apply(X_va_raw, mu, sd)
         #[Bug] AI prompt: Fix Compile Erros
        X_tr = add_intercept(X_tr_z)
        X_va = add_intercept(X_va_z)
        used_names = ["(intercept)"] + [f"z({n})" for n in names]
        scaled_flag = True
        print("step 3: approach = GD (standardized features)")
    else:
        X_tr = add_intercept(X_tr_raw)
        X_va = add_intercept(X_va_raw)
        used_names = ["(intercept)"] + names
        print(f"step 3: approach = {'OLS (closed-form)' if APPROACH=='ols' else 'GD'}")

    if APPROACH == "ols":
        beta = fit_ols(X_tr, y_tr)
    else:
        beta = fit_gd(X_tr, y_tr, lr=GD_LR, epochs=GD_EPOCHS, tol=GD_TOL)

    print("step 3: β (coefficients)")
    for nm, val in zip(used_names, beta.reshape(-1)):
        print(f"  {nm:>20s} : {val:+.6f}")
    if scaled_flag:
        print("step 3: note → β are for standardized (z-score) features.")

    #[Bug] AI prompt: Fix Compile Erros
    y_hat = X_va @ beta
    print(f"step 4: held-out size = {X_va.shape[0]}")
    m = compute_metrics(y_va, y_hat)
     #[Bug] AI prompt: Fix Compile Erros
    print(f"step 4: Validation → RMSE={m['RMSE']:.4f}, MSE={m['MSE']:.4f}, MAE={m['MAE']:.4f}, R²={m['R2']:.4f}")

    _ = step2_select_cost_demo(y_va, y_hat)

    res = (y_va - y_hat).reshape(-1)
    plt.figure(figsize=(6, 4))
    sn.scatterplot(x=y_hat.reshape(-1), y=res, s=12, alpha=0.6, edgecolor=None)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Predicted Rings")
    plt.ylabel("Residual (y - ŷ)")
    plt.title("Residuals vs Predicted (held-out)")
     #[Bug] AI prompt: Fix Compile Erros
    out_png = os.path.join(PLOTS_DIR_STEP4, "residuals.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"step 4: saved residuals plot → {out_png}")

def main():
    print("step 0: setup complete.")
    step1_visualize()
    step3_and_4_train_evaluate()

if __name__ == "__main__":
    main()
