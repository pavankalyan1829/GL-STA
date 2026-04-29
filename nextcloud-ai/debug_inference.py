"""
Debug script: runs ONE inference cycle and prints exactly what
data reaches the scaler and model, so we can identify the constant-output bug.
"""
import os, sys, warnings, logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import joblib
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from datetime import datetime, timedelta, UTC
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package="Custom")
class GATLayer(Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs); self.units = units
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.units), initializer='glorot_uniform', name="gat_w")
        self.a = self.add_weight(shape=(2*self.units, 1), initializer='glorot_uniform', name="gat_a")
        super().build(input_shape)
    def call(self, x):
        h = tf.matmul(x, self.W); n = tf.shape(h)[1]
        h_i = tf.repeat(h, n, axis=1); h_j = tf.tile(h, [1,n,1])
        e = tf.nn.leaky_relu(tf.reshape(tf.matmul(tf.concat([h_i,h_j],axis=-1),self.a),[-1,n,n]))
        return tf.nn.elu(tf.matmul(tf.nn.softmax(e,axis=-1),h))
    def compute_output_shape(self, s): return tuple(s[:-1])+(self.units,)
    def get_config(self):
        c = super().get_config(); c['units'] = self.units; return c
    @classmethod
    def from_config(cls, c): return cls(**c)

PROMETHEUS_URL = "http://localhost:9090"
WINDOW_SIZE    = "10s"
TIME_STEPS     = 10

RAW_QUERIES = {
    "app_cpu":    'rate(container_cpu_usage_seconds_total{name=~".*app.*"}[30s]) * 100',
    "app_mem":    'container_memory_usage_bytes{name=~".*app.*"}',
    "app_latency":'max_over_time(probe_duration_seconds{job="http_latency"}[30s]) or vector(0)',
    "db_cpu":     'rate(container_cpu_usage_seconds_total{name=~".*db.*"}[30s]) * 100',
    "db_mem":     'container_memory_usage_bytes{name=~".*db.*"}',
    "db_threads": 'mysql_global_status_threads_connected or vector(1)',
    "redis_cpu":  'rate(container_cpu_usage_seconds_total{name=~".*redis.*"}[30s]) * 100',
    "redis_mem":  'container_memory_usage_bytes{name=~".*redis.*"}',
}

def q(query):
    try:
        r = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': query}, timeout=5).json()
        if r['data']['result']:
            return float(r['data']['result'][0]['value'][1])
        return None
    except Exception as e:
        return f"ERR: {e}"

print("=== INSTANT PROMETHEUS CHECK ===")
for name, pq in RAW_QUERIES.items():
    val = q(pq)
    print(f"  {name:<22}: {val}")

print()
print("=== SCALER RANGE CHECK ===")
scaler = joblib.load('scaler.pkl')
feature_cols = joblib.load('features.pkl')
print(f"  Feature cols ({len(feature_cols)}):", feature_cols[:5], "...")
print(f"  Scaler mean  (first 5):", scaler.mean_[:5].round(4))
print(f"  Scaler scale (first 5):", scaler.scale_[:5].round(4))

print()
print("=== MODEL BATCH TEST ===")
model = tf.keras.models.load_model('gat_lstm_classifier.keras',
                                    custom_objects={'GATLayer': GATLayer}, compile=False)

# Test 1: all zeros
zeros = np.zeros((1, TIME_STEPS, len(feature_cols)))
p0 = float(model.predict(zeros, verbose=0).flatten()[0])
print(f"  All-zeros input  -> prob = {p0:.4f}")

# Test 2: all ones
ones = np.ones((1, TIME_STEPS, len(feature_cols)))
p1 = float(model.predict(ones, verbose=0).flatten()[0])
print(f"  All-ones  input  -> prob = {p1:.4f}")

# Test 3: scaled CPU spike (simulate stress-ng at 95%)
cpu_spike = np.zeros((TIME_STEPS, len(feature_cols)))
slot_idx = {c: i for i, c in enumerate(feature_cols)}
cpu_idx = slot_idx.get('app_cpu', 0)
cpu_spike[:, cpu_idx] = (95.0 - scaler.mean_[cpu_idx]) / (scaler.scale_[cpu_idx] + 1e-9)
p_spike = float(model.predict(cpu_spike[None], verbose=0).flatten()[0])
print(f"  95% Static CPU   -> prob = {p_spike:.4f}")

# Test 5: Memory Spike (Simulate 2.0 GB usage)
mem_spike = np.zeros((1, TIME_STEPS, len(feature_cols)))
mem_idx = slot_idx.get('app_mem', 1)
raw_mem = 2.0 * 1024 * 1024 * 1024 # 2GB
mem_spike[:, :, mem_idx] = (raw_mem - scaler.mean_[mem_idx]) / (scaler.scale_[mem_idx] + 1e-9)
p_mem = float(model.predict(mem_spike, verbose=0).flatten()[0])
print(f"  2GB Memory Spike -> prob = {p_mem:.4f} (slot={mem_idx})")
print(f"  Scaler mean for app_mem: {scaler.mean_[mem_idx]:,.0f}")
