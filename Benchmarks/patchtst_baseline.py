# Benchmarks/patchtst_baseline.py
import argparse, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

# --- import PatchTST from the cloned repo ---
HERE = Path(__file__).resolve().parent
PATCHTST_SUP = HERE / "patchtst" / "PatchTST_supervised"
sys.path.append(str(PATCHTST_SUP))
from models.PatchTST import Model as PatchTST  # official implementation

# ---------- helpers ----------
def load_long_df(path):
    df = pd.read_csv(path, parse_dates=['ds'])
    need = {'unique_id','ds','y'}
    if not need.issubset(df.columns):
        raise ValueError(f"CSV must have columns {need}, got {df.columns.tolist()}")
    return df.sort_values(['unique_id','ds']).reset_index(drop=True)

def infer_freq(df):
    # infer from the first series
    uid0 = df['unique_id'].iloc[0]
    s = df[df['unique_id'] == uid0]['ds'].sort_values()
    freq = pd.infer_freq(s)
    if freq is None:
        raise ValueError("Could not infer frequency. Ensure regular timestamps per series.")
    return freq
from pandas.tseries.offsets import BusinessDay

def step_from_freq(freq: str):
    unit = ''.join(ch for ch in freq if not ch.isdigit())
    num = ''.join(ch for ch in freq if ch.isdigit())
    num = int(num) if num else 1

    if unit == 'M':                    # monthly (e.g., 'M', '2M')
        return pd.DateOffset(months=num)
    if unit == 'W':                    # weekly (e.g., 'W', '2W')
        return pd.DateOffset(weeks=num)
    if unit == 'B':                    # business day
        return BusinessDay(n=num)
    # fall back: hours/minutes/seconds/days, etc.
    return pd.Timedelta(num, unit)


def default_forecast_date(df, L, freq):
    step = step_from_freq(freq)
    return df['ds'].min() + L * step

def make_cutoffs(start_ts, end_ts, freq):
    step = step_from_freq(freq)
    t = pd.Timestamp(start_ts)
    out = []
    while t <= end_ts:
        out.append(t)
        t = t + step
    return out

def build_windows_tensor(df, L, H, ids):
    # Build (N, 1, L) -> (N, H) across all series (channel-independent)
    X, Y = [], []
    for uid in ids:
        y = df[df['unique_id'] == uid].sort_values('ds')['y'].to_numpy(dtype=np.float32)
        if len(y) < L + H:
            continue
        for t in range(len(y) - (L + H) + 1):
            X.append(y[t:t+L])
            Y.append(y[t+L:t+L+H])
    if not X:
        return None, None
    X = np.stack(X)[:, None, :]   # (N, 1, L)
    Y = np.stack(Y)               # (N, H)
    return torch.from_numpy(X), torch.from_numpy(Y)

def train_patchtst(train_X, train_Y, L, H, d_model, e_layers, n_heads, dropout, lr, epochs, device):
    model = PatchTST(
        c_in=1, seq_len=L, pred_len=H,
        d_model=d_model, n_heads=n_heads, e_layers=e_layers,
        d_ff=4*d_model, dropout=dropout, bias=True, individual=False
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    ds = torch.utils.data.TensorDataset(train_X, train_Y)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, drop_last=False)

    model.train()
    for ep in range(epochs):
        total = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            yhat = model(xb)   # (B, H)
            loss = loss_fn(yhat, yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        # print(f"[PatchTST] epoch {ep+1}/{epochs} MSE={total/len(ds):.5f}")
    return model

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="CSV with unique_id,ds,y")
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--L", type=int, required=True, help="look-back length")
    ap.add_argument("--H", type=int, required=True, help="forecast horizon")
    ap.add_argument("--forecast_date", type=str, default="")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--e_layers", type=int, default=3)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = load_long_df(args.dataset)
    freq = infer_freq(df)
    fdate = pd.Timestamp(args.forecast_date) if args.forecast_date else default_forecast_date(df, args.L, freq)
    end_date = df['ds'].max()

    # train once on data up to forecast_date
    train_df = df[df['ds'] <= fdate].copy()
    ids = train_df['unique_id'].unique()
    X, Y = build_windows_tensor(train_df, args.L, args.H, ids)
    if X is None:
        raise ValueError("Not enough history to build any (L,H) windows.")
    X, Y = X.to(device), Y.to(device)

    start = time.time()
    model = train_patchtst(X, Y, args.L, args.H,
                           args.d_model, args.e_layers, args.n_heads,
                           args.dropout, args.lr, args.epochs, device)
    model.eval()

    # rolling cutoffs
    cutoffs = make_cutoffs(fdate, end_date, freq)

    # forecast for each series at each cutoff
    rows = []
    with torch.no_grad():
        for tcut in cutoffs:
            sub = df[df['ds'] <= tcut]
            for uid, g in sub.groupby('unique_id'):
                y = g.sort_values('ds')['y'].to_numpy(dtype=np.float32)
                if len(y) < args.L:
                    continue
                hist = torch.from_numpy(y[-args.L:][None, None, :]).to(device)  # (1,1,L)
                pred = model(hist).detach().cpu().numpy().reshape(-1)           # (H,)
                row = {"unique_id": uid, "ds": tcut}
                for i in range(args.H):
                    row[i+1] = float(pred[i])
                rows.append(row)

    out = pd.DataFrame(rows).sort_values(['unique_id','ds'])
    out.to_csv(f"{args.save_dir}/mean_preds.csv", index=False)
    print(f"[PatchTST] saved {args.save_dir}/mean_preds.csv  (elapsed {time.time()-start:.1f}s)")

if __name__ == "__main__":
    main()

