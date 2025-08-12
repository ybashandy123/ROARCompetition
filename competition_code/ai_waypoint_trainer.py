# ai_waypoint_trainer.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# -------------------------
# Config
# -------------------------
def make_default_config():
    return {
        "NUM_LAPS": 150,
        "BASELINE_WARMUP_LAPS": 3,           # laps 1-3: warmup/baseline
        "MAX_SHIFT_PER_AXIS_M": 1.0,
        "SEGMENT_SPAN_WAYPOINTS": 8,
        "W_SEG": 0.25,                       # weight on segment improvements
        "W_SEC": 0.65,                       # weight on section improvements
        "W_LAP": 0.10,                       # weight on overall lap time
        "CRASH_PENALTY": 5.0,                # subtracted if crashed
        "SAVE_EVERY": 10,
        "MODEL_PATH": "./ckpts_waypoint_policy",
    }


# -------------------------
# Feature builder
# -------------------------
def _align_to_num_sections(arr, S, fill_value=0.0):
    """Pad or trim a 1D list/array to length S."""
    arr = np.array(arr, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.full((S,), fill_value, dtype=np.float32)
    if arr.size == S:
        return arr.astype(np.float32)
    if arr.size > S:
        return arr[:S].astype(np.float32)
    # pad
    out = np.full((S,), fill_value, dtype=np.float32)
    out[:arr.size] = arr
    return out


def build_features(num_sections, baseline_sections, last_lap_sections):
    """
    Returns features of shape [S, 3] where columns are:
      [0] baseline_norm: baseline section ticks normalized by baseline mean
      [1] last_norm: last-lap section ticks normalized by baseline mean
      [2] delta_norm: (baseline - last) / baseline_mean
    If baseline not available, uses ones; if last not available, zeros.
    """
    S = num_sections

    base = _align_to_num_sections(baseline_sections, S, fill_value=0.0)
    last = _align_to_num_sections(last_lap_sections if last_lap_sections is not None else [], S, fill_value=0.0)

    # Handle missing baseline by substituting ones (so features are stable)
    if np.allclose(base, 0.0):
        base = np.ones_like(base, dtype=np.float32)

    base_mean = np.maximum(np.mean(base), 1e-6).astype(np.float32)

    baseline_norm = (base / base_mean).astype(np.float32)
    last_norm = (last / base_mean).astype(np.float32)
    delta_norm = ((base - last) / base_mean).astype(np.float32)

    feats = np.stack([baseline_norm, last_norm, delta_norm], axis=-1)  # [S, 3]
    return feats.astype(np.float32)


# -------------------------
# Reward
# -------------------------
def compute_reward(
    baseline, section_ticks, segment_ticks, total_ticks, crashed,
    w_seg=0.25, w_sec=0.65, w_lap=0.10, crash_penalty=5.0
):
    """
    Positive reward for being faster than baseline. If baseline is missing,
    we fall back to lap-only signal. All terms are normalized by baseline means.
    """
    # Align
    baseline = np.array(baseline, dtype=np.float32).reshape(-1)
    sec = np.array(section_ticks if len(section_ticks) > 0 else [], dtype=np.float32).reshape(-1)
    seg = np.array(segment_ticks if len(segment_ticks) > 0 else [], dtype=np.float32).reshape(-1)

    # If no baseline sections, use lap-only
    parts = {}
    reward = 0.0

    if baseline.size == 0:
        # no baseline: smaller lap ticks -> larger reward
        parts["lap"] = -float(total_ticks)
        reward += w_lap * parts["lap"]
    else:
        S = baseline.size
        sec = sec if sec.size == S else _align_to_num_sections(sec, S, fill_value=float(total_ticks / max(S, 1)))
        base_mean = float(np.maximum(np.mean(baseline), 1e-6))
        sec_mean = float(np.mean(sec)) if sec.size > 0 else float(total_ticks / max(S, 1))

        # Section improvement (higher is better): (baseline - this) / baseline_mean
        sec_term = float(np.mean((baseline - sec) / base_mean))
        parts["sec"] = sec_term
        reward += w_sec * sec_term

        # Segment term: if given, use same normalization heuristic
        if seg.size > 0:
            # Try to derive a baseline-like mean for segments from sections’ baseline_mean
            seg_term = float(np.mean((np.mean(baseline) - seg) / base_mean))
            parts["seg"] = seg_term
            reward += w_seg * seg_term
        else:
            parts["seg"] = 0.0

        # Lap term: faster than baseline mean per-section -> reward
        lap_term = float((np.mean(baseline) - sec_mean) / base_mean)
        parts["lap"] = lap_term
        reward += w_lap * lap_term

    if crashed:
        parts["crash"] = -float(crash_penalty)
        reward -= float(crash_penalty)
    else:
        parts["crash"] = 0.0

    return float(reward), parts


# -------------------------
# Policy Network
# -------------------------
class PolicyNet(tf.keras.Model):
    """
    Input:  x [B, S, F]
    Output: mu [B, S, 2], log_std [B, S, 2]
    """
    def __init__(self, num_sections: int, hidden: int = 128):
        super().__init__()
        self.num_sections = int(num_sections)
        self.backbone = tf.keras.Sequential([
            layers.Dense(hidden, activation="relu"),
            layers.Dense(hidden, activation="relu"),
        ])
        # Head must yield exactly S * 4 outputs: (mux, muy, logstdx, logstdy) per section
        self.head = layers.Dense(self.num_sections * 4, name="policy_head")

    def call(self, x, training=False):
        # x: [B, S, F]
        B = tf.shape(x)[0]
        S = self.num_sections
        F = tf.shape(x)[2]

        # Flatten per batch to feed MLP
        z = tf.reshape(x, [B, S * F])                 # [B, S*F]
        z = self.backbone(z, training=training)       # [B, hidden]
        raw = self.head(z, training=training)         # [B, S*4]
        raw = tf.reshape(raw, [B, S, 4])              # [B, S, 4]

        mu = raw[..., :2]                             # [B, S, 2]
        log_std = tf.clip_by_value(raw[..., 2:], -5.0, 2.0)  # [B, S, 2]
        return mu, log_std


# -------------------------
# Trainer
# -------------------------
class WaypointRLTrainer:
    def __init__(self, num_sections: int, lr: float = 3e-4, hidden: int = 128, seed: int = 0):
        self.num_sections = int(num_sections)
        tf.keras.utils.set_random_seed(seed)
        self.policy = PolicyNet(num_sections=self.num_sections, hidden=hidden)
        # Build once (helps saving)
        dummy = tf.zeros([1, self.num_sections, 3], dtype=tf.float32)
        _ = self.policy(dummy, training=False)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    @staticmethod
    def _diag_gaussian_logp(actions, mu, log_std):
        """Sum log-prob over all dims [B,S,2] -> [B,1]."""
        var = tf.exp(2.0 * log_std)
        # log N(a | mu, std) = -0.5 * [ (a-mu)^2 / var + log(2π) + 2*log_std ]
        logp_elem = -0.5 * (((actions - mu) ** 2) / var + tf.math.log(2.0 * np.pi) + 2.0 * log_std)
        logp = tf.reduce_sum(logp_elem, axis=[1, 2], keepdims=False)  # [B]
        return logp

    def act(self, features_np):
        """
        features_np: [S, F] -> returns actions [1,S,2], logp [1]
        """
        x = tf.convert_to_tensor(features_np[None, ...], dtype=tf.float32)  # [1,S,F]
        mu, log_std = self.policy(x, training=False)                        # [1,S,2] each
        std = tf.exp(log_std)
        eps = tf.random.normal(tf.shape(mu))
        actions = mu + eps * std                                            # [1,S,2]

        logp = self._diag_gaussian_logp(actions, mu, log_std)               # [1]
        return actions.numpy(), logp.numpy()

    def update(self, features_np, actions_np, logp_np, returns_np):
        """
        Single-step REINFORCE-style update. We recompute logp against the provided actions
        under the *current* policy (this is safer than trusting a stale external logp).

        features_np: [S, F]
        actions_np:  [1, S, 2]
        returns_np:  [1] (scalar return for the trajectory)
        """
        x = tf.convert_to_tensor(features_np[None, ...], dtype=tf.float32)  # [1,S,F]
        a = tf.convert_to_tensor(actions_np, dtype=tf.float32)              # [1,S,2]
        R = tf.convert_to_tensor(returns_np.reshape(-1), dtype=tf.float32)  # [1]

        with tf.GradientTape() as tape:
            mu, log_std = self.policy(x, training=True)                     # [1,S,2]
            logp = self._diag_gaussian_logp(a, mu, log_std)                 # [1]
            # Maximize E[R * logπ(a|s)] -> minimize -R * logp
            loss = -tf.reduce_mean(R * logp)

        grads = tape.gradient(loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))
        return float(loss.numpy())
