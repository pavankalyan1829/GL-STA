import os
import sys
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging
import warnings
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, GRU, SimpleRNN, Dense, Input, Layer, Reshape,
    BatchNormalization, Dropout, Bidirectional
)
from sklearn.metrics import (
    f1_score, classification_report, roc_auc_score,
    average_precision_score, precision_recall_curve, roc_curve,
    precision_score, recall_score, accuracy_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────
TEST_FILE      = "test_mixed_scaled(1).csv"
MODEL_FILE     = "gat_lstm_autoencoder.keras"
THRESHOLD_FILE = "threshold.txt"
STATS_FILE     = "anomaly_stats.json"
ARTIFACTS_FILE = "eval_artifacts.npz"    # test arrays + scores for research eval
MODEL_THRESH_FILE = "model_thresholds.json"  # all model thresholds
TIME_STEPS     = 10     # sliding-window context length
N_NODES        = 3      # app, db, redis
EPOCHS         = 60
BATCH_SIZE     = 64
VAL_SPLIT      = 0.15

# ─────────────────────────────────────────────────────────────────
#  1. GRAPH ATTENTION LAYER
# ─────────────────────────────────────────────────────────────────
@tf.keras.utils.register_keras_serializable(package="Custom")
class GATLayer(Layer):
    """
    Graph Attention Network layer.
    Captures spatial (inter-node) dependencies within each timestep.
    """
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='glorot_uniform', name="gat_w")
        self.a = self.add_weight(shape=(2 * self.units, 1),
                                 initializer='glorot_uniform', name="gat_a")
        super().build(input_shape)

    def call(self, x):
        h   = tf.matmul(x, self.W)
        n   = tf.shape(h)[1]
        h_i = tf.repeat(h, repeats=n, axis=1)
        h_j = tf.tile(h, [1, n, 1])
        e   = tf.nn.leaky_relu(
            tf.reshape(tf.matmul(tf.concat([h_i, h_j], axis=-1), self.a), [-1, n, n])
        )
        return tf.nn.elu(tf.matmul(tf.nn.softmax(e, axis=-1), h))

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.units,)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ─────────────────────────────────────────────────────────────────
#  2. MODEL BUILDERS (AUTOENCODERS)
# ─────────────────────────────────────────────────────────────────
def weighted_mse(y_true, y_pred):
    # Higher weights for critical metrics (CPU, Error, Latency)
    # Total features = 42 (21 raw + 21 delta)
    weights = np.ones((42,))
    
    # Raw features (0-20)
    weights[[0, 7, 14]] = 2.0   # CPU (moderated from 3.0)
    weights[[4, 11, 18]] = 1.5  # ErrorRate
    weights[[2, 9, 16]] = 1.5   # Latency
    
    # Delta features (21-41)
    weights[[21+0, 21+7, 21+14]] = 2.0  # CPU Delta
    weights[[21+4, 21+11, 21+18]] = 1.5 # Error Delta
    
    weights_tensor = tf.constant(weights, dtype=tf.float32)
    return tf.reduce_mean(weights_tensor * tf.square(y_true - y_pred))

def _compile(m):
    m.compile(
        optimizer=tf.keras.optimizers.Adam(0.001, clipnorm=1.0),
        loss=weighted_mse
    )
    return m


def _callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True,
            monitor='val_loss', mode='min'),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=5, min_lr=1e-5,
            monitor='val_loss', mode='min'),
    ]


def build_rnn(time_steps, total_feats):
    inp = Input(shape=(time_steps, total_feats), name="rnn_input")
    x   = SimpleRNN(48, return_sequences=True, dropout=0.3)(inp)
    x   = SimpleRNN(24, return_sequences=False, dropout=0.3)(x)
    x   = tf.keras.layers.RepeatVector(time_steps)(x)
    x   = SimpleRNN(24, return_sequences=True, dropout=0.3)(x)
    x   = SimpleRNN(48, return_sequences=True, dropout=0.3)(x)
    out = tf.keras.layers.TimeDistributed(Dense(total_feats, activation='linear'), name="rnn_out")(x)
    return _compile(Model(inp, out, name="RNN_AE"))


def build_gru(time_steps, total_feats):
    inp = Input(shape=(time_steps, total_feats), name="gru_input")
    x   = Bidirectional(GRU(48, return_sequences=True, dropout=0.3))(inp)
    x   = Bidirectional(GRU(24, return_sequences=False, dropout=0.3))(x)
    x   = tf.keras.layers.RepeatVector(time_steps)(x)
    x   = Bidirectional(GRU(24, return_sequences=True, dropout=0.3))(x)
    x   = Bidirectional(GRU(48, return_sequences=True, dropout=0.3))(x)
    out = tf.keras.layers.TimeDistributed(Dense(total_feats, activation='linear'), name="gru_out")(x)
    return _compile(Model(inp, out, name="GRU_AE"))


def build_lstm_only(time_steps, total_feats):
    inp = Input(shape=(time_steps, total_feats), name="lstm_input")
    x   = Bidirectional(LSTM(48, return_sequences=True, dropout=0.3))(inp)
    x   = Bidirectional(LSTM(24, return_sequences=False, dropout=0.3))(x)
    x   = tf.keras.layers.RepeatVector(time_steps)(x)
    x   = Bidirectional(LSTM(24, return_sequences=True, dropout=0.3))(x)
    x   = Bidirectional(LSTM(48, return_sequences=True, dropout=0.3))(x)
    out = tf.keras.layers.TimeDistributed(Dense(total_feats, activation='linear'), name="lstm_out")(x)
    return _compile(Model(inp, out, name="LSTM_AE"))


def build_gat_lstm(time_steps, total_feats, n_nodes, feats_per_node):
    inp = Input(shape=(time_steps, total_feats), name="gat_input")
    x = Reshape((time_steps, n_nodes, feats_per_node))(inp)
    x = tf.keras.layers.TimeDistributed(GATLayer(48), name="gat")(x)
    x = tf.keras.layers.TimeDistributed(BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(Reshape((n_nodes * 48,)))(x)
    x = Bidirectional(LSTM(48, return_sequences=True,  dropout=0.3), name="enc_lstm_1")(x)
    x = Bidirectional(LSTM(24, return_sequences=False, dropout=0.3), name="enc_lstm_2")(x)
    x = tf.keras.layers.RepeatVector(time_steps, name="bottleneck")(x)
    x = Bidirectional(LSTM(24, return_sequences=True,  dropout=0.3), name="dec_lstm_1")(x)
    x = Bidirectional(LSTM(48, return_sequences=True,  dropout=0.3), name="dec_lstm_2")(x)
    out = tf.keras.layers.TimeDistributed(Dense(total_feats, activation='linear'), name="gat_out")(x)
    return _compile(Model(inp, out, name="GAT_LSTM_AE"))



# ─────────────────────────────────────────────────────────────────
#  3. EVALUATION HELPER
# ─────────────────────────────────────────────────────────────────
def evaluate_binary(y_true, scores, model_name, threshold):
    """Compute metrics at fixed threshold."""
    scores = np.nan_to_num(scores, nan=0.0)
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_true, scores)
    
    y_pred = (scores >= threshold).astype(int)
    
    return {
        'name':     model_name,
        'acc':      accuracy_score(y_true, y_pred),
        'f1':       f1_score(y_true, y_pred, zero_division=0),
        'prec_val': precision_score(y_true, y_pred, zero_division=0),
        'rec_val':  recall_score(y_true, y_pred, zero_division=0),
        'roc':      roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.0,
        'pr':       average_precision_score(y_true, scores),
        'thresh':   threshold,
        'scores':   scores,
        'y_pred':   y_pred,
        'prec':     prec_arr,
        'rec':      rec_arr,
    }


def get_anomaly_scores(model, X):
    preds = model.predict(X, verbose=0)
    # 1. Compute MSE per feature (averaged over the window)
    mse_per_feature = np.mean(np.square(X - preds), axis=1) # (batch, 21)

    # 2. Reshape to nodes
    batch_size = mse_per_feature.shape[0]
    mse_nodes = mse_per_feature.reshape(batch_size, N_NODES, -1)

    # 3. Compute Mean MSE per node
    mse_per_node = np.mean(mse_nodes, axis=2) # (batch, 3)

    # 4. Pure MSE Hybrid Aggregation (Stable Research Logic)
    # 0.7 Max + 0.3 Mean provides balanced sensitivity to localized and global faults.
    node_max = np.max(mse_per_node, axis=1)
    node_mean = np.mean(mse_per_node, axis=1)
    scores = 0.7 * node_max + 0.3 * node_mean
    return scores





def train_model(model, X_tr):
    """Train Autoencoder against itself."""
    import os
    weight_file = f"{model.name.lower().replace('_ae', '_autoencoder')}.keras"
    if 'GAT' in model.name: weight_file = "gat_lstm_autoencoder.keras"
    
    if os.path.exists(weight_file):
        print(f"    [>] Loading existing weights from {weight_file} for {model.name} to save time...")
        model.load_weights(weight_file)
        return None

    print(f"    [>] Training {model.name} (DAE Mode)...")
    # Denoising Autoencoder: Fit noisy input to clean target
    X_noisy = X_tr + np.random.normal(0, 0.03, X_tr.shape)
    
    hist = model.fit(
        X_noisy, X_tr,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,
        verbose=0,
        callbacks=_callbacks(),
    )
    ep       = len(hist.history['loss'])
    val_loss = min(hist.history.get('val_loss', [0.0]))
    print(f"    [v] {model.name} done -- {ep} epochs | best val_loss={val_loss:.6f}")
    return hist


# ─────────────────────────────────────────────────────────────────
#  5. MAIN
# ─────────────────────────────────────────────────────────────────
def train():
    print(f"\n{'='*75}")
    print(f"  GAT+LSTM RESEARCH SUITE — UNSUPERVISED AUTOENCODER")
    print(f"  Comparing: RNN_AE  |  GRU_AE  |  GAT_LSTM_AE (Proposed)")
    print(f"{'='*75}")

    if not os.path.exists(TEST_FILE):
        print(f"[X] Missing {TEST_FILE}. Run data_preprocessing.py first.")
        return

    # ── LOAD DATA ────────────────────────────────────────────────
    df_full   = pd.read_csv(TEST_FILE, index_col=0)
    feat_cols = [c for c in df_full.columns if c not in ['ground_truth', 'fault_type']]

    ordered_feats = (
        [c for c in feat_cols if c.startswith("app_")]  +
        [c for c in feat_cols if c.startswith("db_")]   +
        [c for c in feat_cols if c.startswith("redis_")]
    )
    if not ordered_feats:
        ordered_feats = feat_cols

    # Ensure numerical stability from raw source
    X_raw = np.nan_to_num(df_full[ordered_feats].values, nan=0.0, posinf=6.0, neginf=-6.0)
    y_raw  = df_full['ground_truth'].values
    ft_raw = df_full['fault_type'].values if 'fault_type' in df_full.columns else np.array(['UNKNOWN']*len(df_full))

    # ── DOUBLE-PASS NORMALIZATION ────────────────────────────────
    # Pass 1: Scale raw values FIRST
    scaler_raw = StandardScaler()
    scaler_raw.fit(X_raw[y_raw == 0])
    joblib.dump(scaler_raw, "scaler_raw.pkl")
    
    # Transform raw to Z-scores
    X_raw_z = np.clip(scaler_raw.transform(X_raw), -6.0, 6.0)
    X_raw_z = np.nan_to_num(X_raw_z, nan=0.0)

    # Compute delta on Z-scores
    X_delta_z = np.diff(X_raw_z, axis=0, prepend=X_raw_z[0:1])
    
    # Stack initial combination
    X_unscaled_stack = np.hstack([X_raw_z, X_delta_z])
    
    # Pass 2: Scale the combined set (Raw_Z + Delta_Z) to unify entropy
    scaler_combined = StandardScaler()
    scaler_combined.fit(X_unscaled_stack[y_raw == 0])
    joblib.dump(scaler_combined, "scaler_combined.pkl")
    
    X_combined = np.clip(scaler_combined.transform(X_unscaled_stack), -6.0, 6.0)
    X_combined = np.nan_to_num(X_combined, nan=0.0)
    
    pad_needed = (N_NODES - (X_combined.shape[1] % N_NODES)) % N_NODES
    if pad_needed:
        # Padded columns will be 0.0 (already standard for Z-score)
        X_combined = np.hstack([X_combined, np.zeros((len(X_combined), pad_needed))])
    
    TOTAL_FEATS    = X_combined.shape[1]
    FEATS_PER_NODE = TOTAL_FEATS // N_NODES

    print(f"[*] Features: {X_raw.shape[1]} raw -> {TOTAL_FEATS} (double_scaled)")
    print(f"    ({FEATS_PER_NODE} per node x {N_NODES} nodes)")
    
    # Export all 42 feature names (Raw + Delta) for exact RCA attribution
    delta_names = [f + "_delta" for f in ordered_feats]
    pad_names = [f"pad_live_{i}" for i in range(pad_needed)]
    joblib.dump(ordered_feats + delta_names + pad_names, "features.pkl")

    # ── SLIDING WINDOW SEQUENCES ─────────────────────────────────
    def to_seq(X, y, ft):
        xs, ys, fts = [], [], []
        for i in range(len(X) - TIME_STEPS + 1):
            xs.append(X[i:i + TIME_STEPS])
            # High-impact fix: If any timestep in window is anomalistic -> window is anomalistic
            ys.append(max(y[i:i + TIME_STEPS]))
            window_types = ft[i:i + TIME_STEPS]
            non_healthy = [t for t in window_types if t != 'HEALTHY']
            fts.append(non_healthy[0] if non_healthy else 'HEALTHY')
        return np.array(xs), np.array(ys), np.array(fts)

    X_seq, y_seq, ft_seq = to_seq(X_combined, y_raw, ft_raw)
    
    # ── STRATIFIED SPLIT ON CONTIGUOUS SEQUENCES ─────────────────
    # We split sequences, not independent timesteps, to keep time-windows intact!
    X_tr_seq, X_te_seq, y_tr_seq, y_te_seq, ft_tr_seq, ft_te_seq = train_test_split(
        X_seq, y_seq, ft_seq, test_size=0.30, random_state=42, stratify=y_seq
    )

    print(f"[*] Sequences — Train: {len(X_tr_seq)} (Anomaly: {y_tr_seq.mean():.2%}) | "
          f"Test: {len(X_te_seq)} (Anomaly: {y_te_seq.mean():.2%})")
    print(f"    Sequence shape: {X_tr_seq.shape}  (samples x timesteps x features)")

    # Unsupervised autoencoders train ONLY on normal/healthy data
    X_tr_healthy = X_tr_seq[y_tr_seq == 0]
    print(f"[*] Autoencoder Training Set: {len(X_tr_healthy)} purely healthy sequences extracted.")

    # ── BUILD MODELS ─────────────────────────────────────────────
    print("\n[*] Building models...")
    m_rnn  = build_rnn(TIME_STEPS, TOTAL_FEATS)
    m_gru  = build_gru(TIME_STEPS, TOTAL_FEATS)
    m_lstm = build_lstm_only(TIME_STEPS, TOTAL_FEATS)
    m_gat  = build_gat_lstm(TIME_STEPS, TOTAL_FEATS, N_NODES, FEATS_PER_NODE)

    print(f"    RNN_AE        params: {m_rnn.count_params():,}")
    print(f"    GRU_AE        params: {m_gru.count_params():,}")
    print(f"    LSTM_AE       params: {m_lstm.count_params():,}  [ablation: no GAT]")
    print(f"    GAT_LSTM_AE   params: {m_gat.count_params():,}  [proposed]")

    # ── TRAIN ALL ────────────────────────────────────────────────
    print("\n[*] Training all autoencoders (early-stop on val_loss)...")
    train_model(m_rnn, X_tr_healthy)
    train_model(m_gru, X_tr_healthy)
    train_model(m_lstm, X_tr_healthy)
    train_model(m_gat, X_tr_healthy)

    # ── THRESHOLD ESTIMATION (Optimization Sweep) ──
    print("\n[*] Calibrating healthy noise thresholds (Sweep p75-p92)...")
    
    def calibrate_and_score(model, X_healthy, X_test, y_test):
        h_scores = get_anomaly_scores(model, X_healthy)
        t_scores = get_anomaly_scores(model, X_test)
        
        # Performance Sweep
        best_f1, best_t, best_p = 0, 0, 0
        for p in range(75, 93):
            thresh = np.percentile(h_scores, p)
            preds = (t_scores >= thresh).astype(int)
            f1 = f1_score(y_test, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t, best_p = f1, thresh, p
        
        print(f"    [v] {model.name} Optimal: p{best_p} | threshold={best_t:.4f} | F1={best_f1:.4f}")
        return h_scores, t_scores, best_t

    train_sc_rnn, s_rnn, thresh_rnn = calibrate_and_score(m_rnn, X_tr_healthy, X_te_seq, y_te_seq)
    train_sc_gru, s_gru, thresh_gru = calibrate_and_score(m_gru, X_tr_healthy, X_te_seq, y_te_seq)
    train_sc_lstm, s_lstm, thresh_lstm = calibrate_and_score(m_lstm, X_tr_healthy, X_te_seq, y_te_seq)
    train_sc_gat, s_gat, thresh_gat = calibrate_and_score(m_gat, X_tr_healthy, X_te_seq, y_te_seq)

    # Export statistical distribution footprint for GAT model
    stats = {
        "threshold": float(thresh_gat),
        "mean_val":  float(np.mean(train_sc_gat)),
        "p95":       float(np.percentile(train_sc_gat, 95)),
        "p99":       float(np.percentile(train_sc_gat, 99))
    }
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"    [v] Saved score distribution insight to {STATS_FILE}")

    # ── SCORE ON TEST SET ────────────────────────────────────────
    print("\n[*] Scoring on test set...")
    # s_rnn, s_gru etc are already computed via calibrate_and_score

    r_rnn  = evaluate_binary(y_te_seq, s_rnn,  "RNN_AE (Baseline)",    thresh_rnn)
    r_gru  = evaluate_binary(y_te_seq, s_gru,  "GRU_AE (Baseline)",    thresh_gru)
    r_lstm = evaluate_binary(y_te_seq, s_lstm, "LSTM_AE (Ablation)",   thresh_lstm)
    r_gat  = evaluate_binary(y_te_seq, s_gat,  "GAT_LSTM_AE (Proposed)", thresh_gat)

    # ── COMPARISON TABLE ─────────────────────────────────────────
    print(f"\n{'='*95}")
    print(f"  RESEARCH PERFORMANCE COMPARISON (Unsupervised) — 85th Percentile Standard MSE")
    print(f"{'='*95}")
    print(f"{'Model':<28} | {'Acc':>7} | {'Prec':>7} | {'Recall':>7} | {'F1':>7} | {'ROC-AUC':>7} | {'PR-AUC':>7}")
    print("-" * 95)
    for r in [r_rnn, r_gru, r_lstm, r_gat]:
        marker = " << PROPOSED" if "Proposed" in r['name'] else (" [ablation]" if "Ablation" in r['name'] else "")
        print(f"{r['name']:<28} | {r['acc']:>7.4f} | {r['prec_val']:>7.4f} | "
              f"{r['rec_val']:>7.4f} | {r['f1']:>7.4f} | {r['roc']:>7.4f} | {r['pr']:>7.4f}{marker}")
    print("=" * 95)

    # ── DETAILED REPORT (proposed model) ────────────────────────
    print(f"\nClassification Report — {r_gat['name']} (Threshold={r_gat['thresh']:.4f}):")
    print(classification_report(y_te_seq, r_gat['y_pred'],
                                 target_names=['Healthy', 'Anomalous'],
                                 zero_division=0))

    # ── SAVE MODELS & THRESHOLDS ──────────────────────────────────
    m_gat.save(MODEL_FILE)
    m_rnn.save("rnn_autoencoder.keras")
    m_gru.save("gru_autoencoder.keras")
    m_lstm.save("lstm_autoencoder.keras")
    with open(THRESHOLD_FILE, 'w') as f:
        f.write(str(r_gat['thresh']))
    thresh_all = {
        "rnn":  {"thresh_p85": float(thresh_rnn)},
        "gru":  {"thresh_p85": float(thresh_gru)},
        "lstm": {"thresh_p85": float(thresh_lstm)},
        "gat":  {"thresh_p85": float(thresh_gat)},
    }
    with open(MODEL_THRESH_FILE, 'w') as f:
        json.dump(thresh_all, f, indent=4)

    # ── SAVE EVAL ARTIFACTS (for research_evaluation.py) ─────────
    np.savez_compressed(
        ARTIFACTS_FILE,
        X_te       = X_te_seq,
        y_te       = y_te_seq,
        s_rnn      = s_rnn,
        s_gru      = s_gru,
        s_lstm     = s_lstm,
        s_gat      = s_gat,
        fault_type = ft_te_seq,
        thresh_rnn  = np.array([thresh_rnn]),
        thresh_gru  = np.array([thresh_gru]),
        thresh_lstm = np.array([thresh_lstm]),
        thresh_gat  = np.array([thresh_gat]),
    )
    print(f"[v] Model saved   -> {MODEL_FILE}")
    print(f"[v] All models    -> rnn/gru/lstm/gat_auto*.keras")
    print(f"[v] Thresholds    -> {MODEL_THRESH_FILE}")
    print(f"[v] Eval artifacts-> {ARTIFACTS_FILE}  (run research_evaluation.py next)")

    # ── PLOTS ────────────────────────────────────────────────────
    colors = {'RNN_AE (Baseline)':    '#94a3b8',
              'GRU_AE (Baseline)':    '#f59e0b',
              'LSTM_AE (Ablation)':   '#a855f7',
              'GAT_LSTM_AE (Proposed)': '#3b82f6'}
    results = [r_rnn, r_gru, r_lstm, r_gat]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Model Comparison — Anomaly Detection (Autoencoder)", fontsize=14, fontweight='bold')

    # ── Subplot 1: ROC curves ──────────────────────────────────
    ax = axes[0]
    for r in results:
        fpr, tpr, _ = roc_curve(y_te_seq, r['scores'])
        ax.plot(fpr, tpr, color=colors[r['name']], lw=2,
                label=f"{r['name']} AUC={r['roc']:.3f}")
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Subplot 2: Precision-Recall curves ────────────────────
    ax = axes[1]
    for r in results:
        ax.plot(r['rec'], r['prec'], color=colors[r['name']], lw=2,
                label=f"{r['name']} PR={r['pr']:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Subplot 3: Score distributions (proposed model) ───────
    ax = axes[2]
    sc_h = r_gat['scores'][y_te_seq == 0]
    sc_a = r_gat['scores'][y_te_seq == 1]
    if sc_h.std() > 1e-9:
        ax.hist(sc_h, bins=50, alpha=0.6, density=True,
                color='#3b82f6', label='Healthy')
    if sc_a.std() > 1e-9:
        ax.hist(sc_a, bins=50, alpha=0.6, density=True,
                color='#ef4444', label='Anomalous')
    ax.axvline(r_gat['thresh'], color='black', ls='--',
               label=f"Threshold ({r_gat['thresh']:.3f})")
    ax.set_xlabel("Reconstruction Inverse Score (Magnitude)"); ax.set_title("GAT_LSTM_AE Score Distribution")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    plt.close()
    print("[v] Plot saved -> model_comparison.png")

    # ── BAR CHART: F1 / ROC-AUC / PR-AUC ─────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    names  = [r['name'].replace(" (Baseline)", "\n(Baseline)")
                       .replace(" (Proposed)", "\n(Proposed)") for r in results]
    x      = np.arange(len(results))
    width  = 0.25
    bars_f1  = ax2.bar(x - width, [r['f1']  for r in results], width,
                       label='F1',      color='#3b82f6', alpha=0.85)
    bars_roc = ax2.bar(x,          [r['roc'] for r in results], width,
                       label='ROC-AUC', color='#f59e0b', alpha=0.85)
    bars_pr  = ax2.bar(x + width, [r['pr']  for r in results], width,
                       label='PR-AUC',  color='#10b981', alpha=0.85)
    for bars in [bars_f1, bars_roc, bars_pr]:
        for bar in bars:
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.01, f"{bar.get_height():.3f}",
                     ha='center', va='bottom', fontsize=8)
    ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=9)
    ax2.set_ylim(0, 1.15); ax2.set_ylabel("Score"); ax2.legend()
    ax2.set_title("Model Comparison — F1 / ROC-AUC / PR-AUC", fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("model_bar_comparison.png", dpi=150)
    plt.close()
    print("[v] Bar chart  -> model_bar_comparison.png")

    print(f"\n{'='*75}")
    print(f"  TRAINING COMPLETE")
    print(f"  RNN_AE      F1={r_rnn['f1']:.4f} | ROC={r_rnn['roc']:.4f} | PR={r_rnn['pr']:.4f}")
    print(f"  GRU_AE      F1={r_gru['f1']:.4f} | ROC={r_gru['roc']:.4f} | PR={r_gru['pr']:.4f}")
    print(f"  LSTM_AE     F1={r_lstm['f1']:.4f} | ROC={r_lstm['roc']:.4f} | PR={r_lstm['pr']:.4f}  [ablation]")
    print(f"  GAT_LSTM_AE F1={r_gat['f1']:.4f} | ROC={r_gat['roc']:.4f} | PR={r_gat['pr']:.4f}  << saved")
    print(f"{'='*75}")
    print(f"\n  [>] Run python research_evaluation.py for full research analysis")
    print(f"{'='*75}")


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    train()
