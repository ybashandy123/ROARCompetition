from roar_gym_env import RoarCarlaGymEnv
env = RoarCarlaGymEnv(render_mode="none")
obs, info = env.reset()
print("RESET OK. obs len =", len(obs), "info =", info)
for i in range(10):
    obs, r, term, trunc, info = env.step([0.0, 0.3, 0.0])
    print(f"step {i}: r={r:.3f}, term={term}, trunc={trunc}, info={info}")
env.close()