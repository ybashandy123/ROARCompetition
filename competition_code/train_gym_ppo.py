# train_gym_ppo.py
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import gymnasium as gym

from roar_gym_env import RoarCarlaGymEnv

tf.keras.utils.set_random_seed(42)

class ActorCritic(tf.keras.Model):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Split rays vs. proprio
        # rays = first N entries, but we don't know N ahead of time here;
        # we'll infer during call by a configured slice in the trainer.
        self.rays_net = tf.keras.Sequential([
            layers.Dense(256, activation="relu"),
            layers.Dense(256, activation="relu"),
        ])
        self.prop_net = tf.keras.Sequential([
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
        ])
        self.fuse = tf.keras.Sequential([
            layers.Dense(256, activation="relu"),
            layers.Dense(256, activation="relu"),
        ])

        # Heads
        self.mu = layers.Dense(act_dim, activation=None)
        self.log_std = tf.Variable(initial_value=tf.ones((act_dim,)) * -0.5, trainable=True, name="log_std")
        self.v = layers.Dense(1, activation=None)

    def call(self, obs, ray_count: int, training=False):
        # obs: [B, D]
        rays = obs[:, :ray_count]
        prop = obs[:, ray_count:]
        zr = self.rays_net(rays, training=training)
        zp = self.prop_net(prop, training=training)
        z = self.fuse(tf.concat([zr, zp], axis=-1), training=training)
        mu = self.mu(z)
        v = self.v(z)
        std = tf.exp(self.log_std)[None, :]
        return mu, std, v


class PPOAgent:
    def __init__(
        self,
        env: gym.Env,
        horizon=2048,
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.2,
        pi_lr=3e-4,
        vf_lr=3e-4,
        train_iters=10,
        minibatch=256,
        ent_coef=0.01,
        vf_coef=0.5,
    ):
        self.env = env
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.ray_count = getattr(env, "num_rays", 32)

        self.model = ActorCritic(obs_dim, act_dim)
        # Build once
        dummy = tf.zeros((1, obs_dim), dtype=tf.float32)
        _ = self.model(dummy, self.ray_count, training=False)
        self.pi_opt = tf.keras.optimizers.Adam(pi_lr)
        self.vf_opt = tf.keras.optimizers.Adam(vf_lr)

        self.horizon = horizon
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.train_iters = train_iters
        self.minibatch = minibatch
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef

    @staticmethod
    def _gaussian_logprob(a, mu, std):
        var = std ** 2
        return -0.5 * tf.reduce_sum(((a - mu) ** 2) / var + tf.math.log(2.0 * np.pi) + 2.0 * tf.math.log(std), axis=-1)

    def sample_rollout(self):
        obs_buf, act_buf, rew_buf, val_buf, logp_buf, done_buf = [], [], [], [], [], []

        obs, info = self.env.reset()
        ep_rew, ep_len = 0.0, 0
        for t in range(self.horizon):
            x = tf.convert_to_tensor(obs[None, :], dtype=tf.float32)
            mu, std, v = self.model(x, self.ray_count, training=False)
            eps = tf.random.normal(tf.shape(mu))
            act = (mu + std * eps).numpy()[0]
            logp = self._gaussian_logprob(tf.convert_to_tensor(act[None, :], tf.float32), mu, std).numpy()[0]

            next_obs, reward, terminated, truncated, info = self.env.step(act)
            done = terminated or truncated

            obs_buf.append(obs.copy())
            act_buf.append(act.copy())
            rew_buf.append(float(reward))
            val_buf.append(float(v.numpy()[0, 0]))
            logp_buf.append(float(logp))
            done_buf.append(float(done))

            ep_rew += reward
            ep_len += 1
            obs = next_obs

            if done:
                obs, info = self.env.reset()
                ep_rew, ep_len = 0.0, 0

        return (
            np.array(obs_buf, dtype=np.float32),
            np.array(act_buf, dtype=np.float32),
            np.array(rew_buf, dtype=np.float32),
            np.array(val_buf, dtype=np.float32),
            np.array(logp_buf, dtype=np.float32),
            np.array(done_buf, dtype=np.float32),
        )

    def compute_gae(self, rews, vals, dones, next_val):
        T = len(rews)
        adv = np.zeros(T, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - dones[t]
            delta = rews[t] + self.gamma * next_val * nonterminal - vals[t]
            lastgaelam = delta + self.gamma * self.lam * nonterminal * lastgaelam
            adv[t] = lastgaelam
            next_val = vals[t]
        ret = adv + vals
        # Normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret

    @tf.function
    def _update_minibatch(self, obs, acts, old_logp, adv, ret):
        with tf.GradientTape(persistent=True) as tape:
            mu, std, v = self.model(obs, self.ray_count, training=True)
            logp = PPOAgent._gaussian_logprob(acts, mu, std)
            ratio = tf.exp(logp - old_logp)
            clipped = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv, clipped * adv))
            entropy = tf.reduce_mean(0.5 * tf.math.log(2.0 * np.pi * np.e) + tf.math.log(std))
            v_loss = tf.reduce_mean((ret - tf.squeeze(v, axis=-1)) ** 2)
            loss = pi_loss - self.ent_coef * entropy + self.vf_coef * v_loss
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.pi_opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return pi_loss, v_loss, entropy

    def train(self, total_steps=200_000):
        steps_done = 0
        while steps_done < total_steps:
            obs, acts, rews, vals, old_logp, dones = self.sample_rollout()
            # Bootstrap value for final state:
            x = tf.convert_to_tensor(obs[-1:,:], tf.float32)
            _, _, next_v = self.model(x, self.ray_count, training=False)
            adv, ret = self.compute_gae(rews, vals, dones, float(next_v.numpy()[0,0]))

            # SGD over minibatches
            idx = np.arange(len(obs))
            for _ in range(self.train_iters):
                np.random.shuffle(idx)
                for start in range(0, len(idx), self.minibatch):
                    mb = idx[start:start+self.minibatch]
                    pi_l, v_l, ent = self._update_minibatch(
                        tf.convert_to_tensor(obs[mb], tf.float32),
                        tf.convert_to_tensor(acts[mb], tf.float32),
                        tf.convert_to_tensor(old_logp[mb], tf.float32),
                        tf.convert_to_tensor(adv[mb], tf.float32),
                        tf.convert_to_tensor(ret[mb], tf.float32),
                    )
            steps_done += len(obs)
            print(f"[{steps_done:>8}] pi_loss={pi_l.numpy():.3f} v_loss={v_l.numpy():.3f} ent={ent.numpy():.3f}")

        # Save weights
        os.makedirs("ckpts_ppo", exist_ok=True)
        self.model.save_weights("ckpts_ppo/ac_ppo")


if __name__ == "__main__":
    # NOTE: start CARLA server before running this.
    env = RoarCarlaGymEnv(num_rays=32, fov_deg=180.0, max_ray_m=80.0, render_mode="none")
    agent = PPOAgent(env, horizon=2048, train_iters=10, minibatch=256, ent_coef=0.01)
    try:
        print('Began training')
        agent.train(total_steps=150_000)  # bump as needed
        print('training complete')
    finally:
        env.close()
