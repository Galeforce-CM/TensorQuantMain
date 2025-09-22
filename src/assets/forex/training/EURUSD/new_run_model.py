#!/usr/bin/env python3
"""
EURUSD NN Benchmark — Walk‑Forward Triple‑Barrier

What this script does
---------------------
- Loads OHLC(V) data for EURUSD (CSV or DataFrame hook).
- Builds compact, robust features (returns, vol, RSI, MACD, %K, ATR,...).
- Creates triple‑barrier labels using a forward horizon with TP/SL multiples.
- Generates sliding windows for a sequence NN (1D‑CNN by default).
- Runs strict walk‑forward evaluation: train on last X bars, test on next Y bars, repeat.
- Benchmarks NN vs a simple Logistic Regression baseline.
- Converts class probabilities to LONG/SHORT/FLAT with an EV threshold τ and costs.
- Computes fold‑level metrics and consolidated PnL + equity curve.
- Saves artifacts: metrics CSV, trades CSV, equity CSV, and best model per fold.

Quick start
-----------
python eurusd_nn_benchmark.py \
  --csv data/eurusd_minute.csv \
  --time-col time --price-col Close --high-col High --low-col Low --vol-col Volume \
  --horizon 60 --tp-mult 2.0 --sl-mult 1.5 \
  --window 128 --train-bars 50000 --test-bars 5000 \
  --tx-cost-bps 0.8 --tau 0.02 --scale-mode robust \
  --outdir outputs_eurusd/

Notes
-----
- CSV must include time, open/high/low/close (case‑insensitive configurable via args).
- Horizons/window measured in bars of the input frequency.
- Costs are round‑turn bps on notional PnL; adjust to your broker spread/commissions.
- Set CUDA_VISIBLE_DEVICES for GPU or run on CPU.

"""
from __future__ import annotations
import argparse
import os
import math
import json
import warnings
from dataclasses import dataclass
from typing import Tuple, List, Dict, Iterable, Optional

import numpy as np
import pandas as pd

# sklearn
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# TensorFlow / Keras
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, GlobalAveragePooling1D, Dropout, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------- Utils ---------------------------

def ensure_dir(p: str) -> None:
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)


def zscore(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    mu = np.nanmean(x)
    sd = np.nanstd(x) + eps
    return (x - mu) / sd


# ------------------------ Feature set ------------------------

def compute_features(df: pd.DataFrame,
                     price_col: str,
                     high_col: str,
                     low_col: str,
                     vol_col: Optional[str] = None) -> pd.DataFrame:
    """Compact, robust features that work across FX data.
    Assumes df is sorted by time ascending.
    """
    px = df[price_col].astype(float).values
    high = df[high_col].astype(float).values
    low = df[low_col].astype(float).values
    n = len(df)

    out = pd.DataFrame(index=df.index)

    # log returns and their rolling stats
    logret = np.zeros(n)
    logret[1:] = np.diff(np.log(px))
    out['ret'] = logret

    for w in (5, 14, 30, 60):
        out[f'vol_{w}'] = pd.Series(logret).rolling(w).std().values
        out[f'ma_{w}'] = pd.Series(logret).rolling(w).mean().values

    # ATR (Wilder) in price space
    tr = np.empty(n)
    tr[0] = np.nan
    prev_close = px[:-1]
    tr[1:] = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close)))
    atr14 = pd.Series(tr).ewm(alpha=1/14.0, adjust=False).mean().values
    out['atr14'] = atr14

    # RSI (14)
    diff = np.diff(px, prepend=px[0])
    up = np.where(diff > 0, diff, 0.0)
    dn = np.where(diff < 0, -diff, 0.0)
    roll_up = pd.Series(up).ewm(alpha=1/14.0, adjust=False).mean().values
    roll_dn = pd.Series(dn).ewm(alpha=1/14.0, adjust=False).mean().values
    rs = (roll_up + 1e-12) / (roll_dn + 1e-12)
    out['rsi'] = 100.0 - (100.0 / (1.0 + rs))

    # MACD (12, 26, 9) on price
    ema12 = pd.Series(px).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(px).ewm(span=26, adjust=False).mean().values
    macd = ema12 - ema26
    signal = pd.Series(macd).ewm(span=9, adjust=False).mean().values
    out['macd'] = macd
    out['macd_sig'] = signal
    out['macd_hist'] = macd - signal

    # Stoch %K, %D (14)
    hh = pd.Series(high).rolling(14).max().values
    ll = pd.Series(low).rolling(14).min().values
    denom = (hh - ll)
    denom[denom == 0] = np.nan
    stoch_k = 100.0 * (px - ll) / denom
    out['stoch_k'] = pd.Series(stoch_k).fillna(method='bfill').values
    out['stoch_d'] = pd.Series(out['stoch_k']).rolling(3).mean().fillna(method='bfill').values

    if vol_col and vol_col in df.columns:
        vol = df[vol_col].astype(float).values
        out['vol_ema'] = pd.Series(vol).ewm(span=20, adjust=False).mean().values
        out['vol_chg'] = pd.Series(np.diff(vol, prepend=vol[0]) / (vol + 1e-9)).values

    out = out.replace([np.inf, -np.inf], np.nan).fillna(method='bfill').fillna(method='ffill')
    return out


# ------------------ Triple‑Barrier labeling ------------------

@dataclass
class BarrierOutcome:
    label: int  # -1, 0, +1
    hit_idx: int  # relative index where first barrier hit (1..h) or h if none
    ret_realized: float  # realized simple return measured from entry to hit (signed)


def triple_barrier_labels(px: np.ndarray,
                          horizon: int,
                          tp_mult: float,
                          sl_mult: float,
                          ref_vol: np.ndarray | None = None,
                          use_atr: np.ndarray | None = None) -> List[BarrierOutcome]:
    """Vectorized-ish triple-barrier using either volatility or ATR as barrier size.
    Return per-bar simple returns (not in pips), appropriate for benchmarking.
    """
    n = len(px)
    outcomes: List[BarrierOutcome] = [BarrierOutcome(0, 0, 0.0) for _ in range(n)]

    # distance scale per bar
    if use_atr is not None:
        scale = use_atr
    else:
        scale = ref_vol if ref_vol is not None else pd.Series(np.log(px)).diff().rolling(30).std().values
        scale = pd.Series(scale).fillna(method='bfill').fillna(method='ffill').values

    for i in range(n - horizon):
        p0 = px[i]
        up = p0 * (tp_mult * (scale[i] if scale[i] > 0 else 1e-5))  # used for return magnitude only
        dn = p0 * (sl_mult * (scale[i] if scale[i] > 0 else 1e-5))

        # Convert scale (roughly in log‑ret units if ref_vol) to price deltas using small-angle approx
        # For small r, price target ~ p0 * (1 + r). We'll clamp to avoid extremes.
        tp_price = p0 * (1.0 + np.clip(tp_mult * scale[i], 1e-5, 0.10))
        sl_price = p0 * (1.0 - np.clip(sl_mult * scale[i], 1e-5, 0.10))

        up_hit = None
        dn_hit = None
        for h in range(1, horizon + 1):
            p = px[i + h]
            if up_hit is None and p >= tp_price:
                up_hit = h
            if dn_hit is None and p <= sl_price:
                dn_hit = h
            if up_hit is not None or dn_hit is not None:
                break
        if up_hit is not None and (dn_hit is None or up_hit <= dn_hit):
            outcomes[i] = BarrierOutcome(+1, up_hit, (tp_price - p0) / p0)
        elif dn_hit is not None and (up_hit is None or dn_hit <= up_hit):
            outcomes[i] = BarrierOutcome(-1, dn_hit, (sl_price - p0) / p0)
        else:
            # Neutral: no barrier hit within horizon → mark 0 and realized return to last bar
            pT = px[i + horizon]
            outcomes[i] = BarrierOutcome(0, horizon, (pT - p0) / p0)

    return outcomes


# ---------------------- Windowing (NN) -----------------------

def make_windows(X: np.ndarray, y: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build rolling windows (N, window, F) aligned so that the window ends at t-1
    and the label belongs to t (entry at window end).
    """
    n, f = X.shape
    W = []
    Y = []
    for t in range(window, n):
        W.append(X[t - window:t, :])
        Y.append(y[t])
    return np.asarray(W), np.asarray(Y)


# ------------------------ Model (1D‑CNN) --------------------

def build_cnn(input_shape: Tuple[int, int], n_classes: int = 3) -> tf.keras.Model:
    m = Sequential([
        Input(shape=input_shape),
        Conv1D(32, 5, padding='causal'), BatchNormalization(), ReLU(),
        Conv1D(32, 5, padding='causal'), BatchNormalization(), ReLU(),
        Conv1D(64, 3, padding='causal'), BatchNormalization(), ReLU(),
        GlobalAveragePooling1D(),
        Dropout(0.25),
        Dense(64), ReLU(),
        Dropout(0.25),
        Dense(n_classes, activation='softmax')
    ])
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=CategoricalCrossentropy(label_smoothing=0.05),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name='acc')]
    )
    return m


# ----------------------- Walk‑forward CV ---------------------

def walk_forward_ranges(n: int, train_bars: int, test_bars: int, step: Optional[int] = None) -> List[Tuple[int, int, int, int]]:
    if step is None:
        step = test_bars
    ranges = []
    start = 0
    while True:
        train_start = max(0, start)
        train_end = train_start + train_bars
        test_end = train_end + test_bars
        if test_end > n:
            break
        ranges.append((train_start, train_end, train_end, test_end))
        start += step
    return ranges


# -------------------- Prob → trade decisions -----------------

def probs_to_signal(p_up: float, p_dn: float, tau: float) -> int:
    edge = p_up - p_dn
    if edge > tau:
        return +1  # long
    elif edge < -tau:
        return -1  # short
    else:
        return 0


# ---------------------------- PnL ----------------------------

def backtest_barrier_pnl(signals: np.ndarray,
                         outcomes: List[BarrierOutcome],
                         tx_cost_bps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Map signals to realized returns using precomputed barrier outcomes.
    - If signal matches label (+1/-1), take realized ret. If opposite, take -|ret|.
    - If flat, return 0.
    - Apply round‑turn costs in bps to non‑flat trades.
    Returns per‑trade returns array and equity curve (cumprod on 1+r).
    """
    rets = []
    for s, out in zip(signals, outcomes):
        if s == 0:
            rets.append(0.0)
            continue
        gross = out.ret_realized if s == out.label else -abs(out.ret_realized)
        cost = abs(gross) * (tx_cost_bps / 10000.0)
        rets.append(gross - cost)
    rets = np.asarray(rets, dtype=float)
    eq = np.cumprod(1.0 + np.nan_to_num(rets))
    return rets, eq


# ------------------------- Main runner -----------------------

def run(args: argparse.Namespace):
    ensure_dir(args.outdir)

    # -------- Load data --------
    df = pd.read_csv(args.csv)
    # Normalize column names
    colmap = {
        'time': args.time_col,
        'close': args.price_col,
        'high': args.high_col,
        'low': args.low_col,
        'volume': args.vol_col
    }
    for k, v in list(colmap.items()):
        if v not in df.columns:
            # try case‑insensitive
            cand = {c.lower(): c for c in df.columns}.get(v.lower())
            if cand:
                colmap[k] = cand
            else:
                if k == 'volume':
                    colmap[k] = None
                else:
                    raise ValueError(f"Column '{v}' not found in CSV")

    df = df.sort_values(colmap['time'])

    # -------- Features & labels --------
    feats = compute_features(df, price_col=colmap['close'], high_col=colmap['high'], low_col=colmap['low'], vol_col=colmap['volume'])
    price = df[colmap['close']].astype(float).values

    # Reference vol for barrier sizes: use rolling logret std (vol_30)
    ref_vol = feats['vol_30'].values
    atr = feats['atr14'].values

    outcomes = triple_barrier_labels(price,
                                     horizon=args.horizon,
                                     tp_mult=args.tp_mult,
                                     sl_mult=args.sl_mult,
                                     ref_vol=ref_vol if args.barrier_basis == 'vol' else None,
                                     use_atr=atr if args.barrier_basis == 'atr' else None)

    labels = np.array([o.label for o in outcomes])
    realized = np.array([o.ret_realized for o in outcomes])

    # Discard initial NaNs
    valid = np.isfinite(feats).all(axis=1) & np.isfinite(labels)
    feats = feats[valid]
    labels = labels[valid]
    price = price[valid]
    outcomes = [outcomes[i] for i, ok in enumerate(valid) if ok]

    # Map labels {-1,0,1} → {0,1,2}
    label_map = { -1:0, 0:1, 1:2 }
    y_int = np.array([label_map[int(y)] for y in labels])

    # Scaling
    scaler = RobustScaler() if args.scale_mode == 'robust' else StandardScaler()
    X = scaler.fit_transform(feats.values)

    # Windows
    Xw, Yw = make_windows(X, y_int, window=args.window)

    n_classes = 3
    onehot = tf.keras.utils.to_categorical

    # Walk‑forward ranges operate on windowed arrays
    n = len(Xw)
    ranges = walk_forward_ranges(n, args.train_bars, args.test_bars, step=args.step)

    # Logs
    metrics_rows = []
    trades_rows = []
    equity_all = []

    print(f"Total windows: {n}, folds: {len(ranges)}")

    for fold, (tr0, tr1, te0, te1) in enumerate(ranges, start=1):
        X_tr, Y_tr = Xw[tr0:tr1], Yw[tr0:tr1]
        X_te, Y_te = Xw[te0:te1], Yw[te0:te1]

        # Baseline (LogReg) on last‑timestep features only
        X_tr_last = X_tr[:, -1, :]
        X_te_last = X_te[:, -1, :]
        lr = LogisticRegression(max_iter=1000, multi_class='multinomial')
        lr.fit(X_tr_last, Y_tr)
        proba_lr = lr.predict_proba(X_te_last)

        print(f'proba_lr {proba_lr}')

        # NN
        model = build_cnn(input_shape=(args.window, X_tr.shape[-1]), n_classes=n_classes)
        best_path = os.path.join(args.outdir, f"best_fold{fold}.keras")
        cb = [
            EarlyStopping(monitor='val_acc', mode='max', patience=10, restore_best_weights=True),
            ModelCheckpoint(best_path, monitor='val_acc', mode='max', save_best_only=True)
        ]
        hist = model.fit(
            X_tr, onehot(Y_tr, n_classes),
            validation_split=0.2,
            epochs=args.epochs,
            batch_size=args.batch,
            callbacks=cb,
            verbose=0
        )
        proba_nn = model.predict(X_te, verbose=0)

        # Prob → signals
        p_up_nn = proba_nn[:, 2]
        p_dn_nn = proba_nn[:, 0]
        p_up_lr = proba_lr[:, 2]
        p_dn_lr = proba_lr[:, 0]

        sig_nn = np.array([probs_to_signal(u, d, args.tau) for u, d in zip(p_up_nn, p_dn_nn)], dtype=int)
        sig_lr = np.array([probs_to_signal(u, d, args.tau) for u, d in zip(p_up_lr, p_dn_lr)], dtype=int)

        # Align outcomes to test windows (entry at window end → index te_start+idx+window in original series)
        # Because we made windows, the outcome at sample t corresponds to original index t+window
        outcomes_te = outcomes[te0 + args.window: te1 + args.window]

        # Backtest
        rets_nn, eq_nn = backtest_barrier_pnl(sig_nn, outcomes_te, args.tx_cost_bps)
        rets_lr, eq_lr = backtest_barrier_pnl(sig_lr, outcomes_te, args.tx_cost_bps)

        # Fold metrics
        def metrics_block(name: str, y_true: np.ndarray, proba: np.ndarray, sig: np.ndarray, rets: np.ndarray, eq: np.ndarray) -> Dict:
            y_pred = np.argmax(proba, axis=1)
            rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            out = {
                'model': name,
                'fold': fold,
                'n_tr': len(X_tr),
                'n_te': len(X_te),
                'acc': rep['accuracy'],
                'prec_dn': rep['0']['precision'], 'rec_dn': rep['0']['recall'],
                'prec_neu': rep['1']['precision'], 'rec_neu': rep['1']['recall'],
                'prec_up': rep['2']['precision'], 'rec_up': rep['2']['recall'],
                'trades': int(np.count_nonzero(sig)),
                'avg_ret': float(np.nanmean(rets)) if len(rets) else 0.0,
                'sum_ret': float(np.nansum(rets)),
                'sharpe': float(np.nanmean(rets) / (np.nanstd(rets) + 1e-12) * math.sqrt(252*24*12)) if np.nanstd(rets) > 0 else 0.0,
                'max_dd': float(max_drawdown(eq)),
                'end_equity': float(eq[-1]) if len(eq) else 1.0
            }
            return out

        m_nn = metrics_block('cnn', Y_te, proba_nn, sig_nn, rets_nn, eq_nn)
        m_lr = metrics_block('logreg', Y_te, proba_lr, sig_lr, rets_lr, eq_lr)
        metrics_rows.extend([m_nn, m_lr])

        # Trades log (per test sample)
        for i, (u, d, s, r) in enumerate(zip(p_up_nn, p_dn_nn, sig_nn, rets_nn)):
            trades_rows.append({
                'fold': fold,
                'idx': te0 + i,
                'p_up': float(u), 'p_dn': float(d), 'signal': int(s), 'ret': float(r)
            })

        # Equity join
        equity_all.append(pd.DataFrame({
            'fold': fold,
            'step': np.arange(len(eq_nn)),
            'eq_cnn': eq_nn,
            'eq_lr': eq_lr
        }))

    metrics_df = pd.DataFrame(metrics_rows)
    trades_df = pd.DataFrame(trades_rows)
    equity_df = pd.concat(equity_all, ignore_index=True) if equity_all else pd.DataFrame()

    metrics_path = os.path.join(args.outdir, 'metrics.csv')
    trades_path = os.path.join(args.outdir, 'trades.csv')
    equity_path = os.path.join(args.outdir, 'equity.csv')

    metrics_df.to_csv(metrics_path, index=False)
    trades_df.to_csv(trades_path, index=False)
    equity_df.to_csv(equity_path, index=False)

    # Print compact benchmark table
    if not metrics_df.empty:
        print("\n===== Fold metrics (mean by model) =====")
        print(metrics_df.groupby('model')[['acc','prec_dn','rec_dn','prec_neu','rec_neu','prec_up','rec_up','trades','avg_ret','sum_ret','sharpe','max_dd','end_equity']].mean().round(4))

    print(f"\nArtifacts saved to: {args.outdir}")


def max_drawdown(eq: np.ndarray) -> float:
    if eq is None or len(eq) == 0:
        return 0.0
    peak = -np.inf
    mdd = 0.0
    for v in eq:
        peak = max(peak, v)
        dd = (peak - v) / (peak + 1e-12)
        if dd > mdd:
            mdd = dd
    return float(mdd)


# ---------------------------- CLI ----------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EURUSD NN Benchmark — Walk‑Forward Triple‑Barrier")
    p.add_argument('--csv', type=str, required=True, help='Path to OHLCV CSV')
    p.add_argument('--time-col', type=str, default='time')
    p.add_argument('--price-col', type=str, default='Close')
    p.add_argument('--high-col', type=str, default='High')
    p.add_argument('--low-col', type=str, default='Low')
    p.add_argument('--vol-col', type=str, default='Volume')

    p.add_argument('--horizon', type=int, default=60, help='Forward lookahead bars for barriers')
    p.add_argument('--tp-mult', type=float, default=2.0, help='Take‑profit multiple of ref scale')
    p.add_argument('--sl-mult', type=float, default=1.5, help='Stop‑loss multiple of ref scale')
    p.add_argument('--barrier-basis', choices=['vol','atr'], default='vol')

    p.add_argument('--window', type=int, default=128, help='Sequence window length for NN')
    p.add_argument('--train-bars', type=int, default=50, help='Bars in each training slice (post‑window)')
    p.add_argument('--test-bars', type=int, default=50, help='Bars in each test slice (post‑window)')
    p.add_argument('--step', type=int, default=None, help='Advance step between folds (default=test-bars)')

    p.add_argument('--scale-mode', choices=['standard','robust'], default='robust')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch', type=int, default=512)

    p.add_argument('--tau', type=float, default=0.02, help='EV threshold on (p_up - p_dn) to trigger a trade')
    p.add_argument('--tx-cost-bps', type=float, default=0.8, help='Round‑turn cost in bps, applied per trade')

    p.add_argument('--outdir', type=str, default='outputs_eurusd')
    return p


if __name__ == '__main__':
    args = build_argparser().parse_args()
    run(args)
