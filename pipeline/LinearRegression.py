import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Paths (os only) ---
OUT_DIR = os.path.join("outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Step 1: load + clean (2018)
FILE = os.path.join("datasets", "gdp-vs-happiness.csv")
YEAR = 2018
YCOL = "Cantril ladder score"
XCOL = "GDP per capita, PPP (constant 2021 international $)"
REGCOL = "World regions according to OWID"

def load(file=FILE, year=YEAR, ycol=YCOL, xcol=XCOL, regcol=REGCOL):
    df = (pd.read_csv(file)
            .loc[lambda d: d["Year"] == year]
            .dropna(subset=[ycol, xcol])
            # [BUG] AI prompt: Fix Compile Erros
            .rename(columns={ycol: "y", xcol: "x", regcol: "Region"}))
    cols = ["Entity", "Code", "Year", "x", "y", "Region"]
    return df[cols].reset_index(drop=True)

# Step 2: Math equations
def standardize(x):
    x = np.asarray(x, dtype=float).ravel()
    mu = x.mean()
    sd = x.std(ddof=0)
    z = (x - mu) / sd
    return z, mu, sd

def predict(beta, x):
    x = np.asarray(x, dtype=float)
    return beta[0] + beta[1] * x

# [1] AI prompt: Calculate mean squared error (MSE).
def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    d = y_true - y_pred
    return float(np.mean(d * d))

# Step 3: Gradient Descent
# [2] AI prompt: Prep for GD — flatten, standardize; short-circuit if no variance.
# [3] AI prompt: Run GD on z; rescale back to original x-units.
def gd_fit(x_raw, y, lr=0.1, epochs=800, tol=0.0):
    x_raw = np.asarray(x_raw, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z, mu, sd = standardize(x_raw)
    if sd == 0:
        return np.array([float(y.mean()), 0.0])

    n = len(z)
    b0, b1 = 0.0, 0.0
    prev = np.inf
    for _ in range(int(epochs)):
        yhat = b0 + b1 * z
        e = yhat - y
        g0 = (2/n) * np.sum(e)
        g1 = (2/n) * np.sum(e * z)
        b0 -= lr * g0
        b1 -= lr * g1
        if tol:
            cur = float(np.mean(e * e))
            if prev - cur < tol:
                break
            prev = cur
    B1 = b1 / sd
    B0 = b0 - b1 * (mu / sd)
    return np.array([B0, B1])

# Step 4: OLS closed-form
# [4] AI prompt: Solve regression via OLS (design matrix with bias column).
def ols_fit(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta

# Step 5: run experiments (5 lrs × 5 epochs = 25 runs)
# [5]/[6] AI prompts: grid lr×epochs; compute MSE; sort by MSE.(twice since different)
def run_experiments(x, y):
    lrs = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1]
    eps = [100, 400, 800, 1200, 2000]
    results = []
    for lr in lrs:
        for ep in eps:
            beta = gd_fit(x, y, lr=lr, epochs=ep)
            loss = mse(y, predict(beta, x))
            results.append({"lr": lr, "epochs": ep, "beta": beta, "mse": loss})
    results.sort(key=lambda r: r["mse"])
    return results

# Step 6: plotting (A) top-6 GD lines; (B) OLS vs best-GD)
# [7] AI prompt: Visualize scatter + lines; compare OLS vs best GD.
def plot_lines(df, results, ols_beta, save=False):
    x = df["x"].to_numpy(float)
    y = df["y"].to_numpy(float)
    xs = np.linspace(x.min(), x.max(), 200)

    top = results[:6]
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.7, label="2018 countries")
    for r in top:
        ys = predict(r["beta"], xs)
        lbl = f"GD lr={r['lr']}, ep={r['epochs']}\nβ′=[{r['beta'][0]:.3f}, {r['beta'][1]:.6f}]"
        plt.plot(xs, ys, linewidth=1.5, label=lbl)
    plt.xlabel("GDP per capita (PPP, const. 2021 $)")
    plt.ylabel("Happiness (Cantril ladder)")
    plt.title("GD regression lines (top 6 by MSE) over 2018 data")
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(OUT_DIR, "part1_gd_lines.png"), dpi=160)

    # [BUG] AI prompt: Fix Compile Erros
    best = top[0]
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.7, label="2018 countries")
    plt.plot(xs, predict(ols_beta, xs), linewidth=2.0,
             label=f"OLS β=[{ols_beta[0]:.3f}, {ols_beta[1]:.6f}]")
    plt.plot(xs, predict(best["beta"], xs), linewidth=2.0,
             label=f"Best GD lr={best['lr']}, ep={best['epochs']}\nβ′=[{best['beta'][0]:.3f}, {best['beta'][1]:.6f}]")
    plt.xlabel("GDP per capita (PPP, const. 2021 $)")
    plt.ylabel("Happiness (Cantril ladder)")
    plt.title("OLS vs Best Gradient Descent")
    plt.legend(fontsize=9, loc="best")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(OUT_DIR, "part1_ols_vs_bestgd.png"), dpi=160)

    plt.show()

# Step 7: main — print all β′ + plots
# [8] AI prompt: Orchestrate the run; print summary; save plots.
def main():
    df = load()
    x = df["x"].to_numpy(float)
    y = df["y"].to_numpy(float)

    beta_ols = ols_fit(x, y)
    ols_mse = mse(y, predict(beta_ols, x))

    results = run_experiments(x, y)

    print("\nAll GD runs (β′ in original x units):")
    for r in results:
        b0, b1 = r["beta"]
        print(f"lr={r['lr']:<7}  ep={r['epochs']:<5}  β′=[{b0:.6f}, {b1:.9f}]  MSE={r['mse']:.6f}")

    best = results[0]
    print("\nOLS vs Best-GD:")
    print(f"OLS    β=[{beta_ols[0]:.6f}, {beta_ols[1]:.9f}]  MSE={ols_mse:.6f}")
    # [BUG] AI prompt: Fix Compile Erros
    print(f"BestGD lr={best['lr']}, ep={best['epochs']}, "
          f"β′=[{best['beta'][0]:.6f}, {best['beta'][1]:.9f}], "
          f"MSE={best['mse']:.6f}")

    plot_lines(df, results, beta_ols, save=True)

if __name__ == "__main__":
    main()
