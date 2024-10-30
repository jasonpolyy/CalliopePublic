import gymnasium as gym

from stable_baselines3 import A2C, SAC
from stable_baselines3.common.callbacks import CheckpointCallback

#env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("Pendulum-v1", render_mode="rgb_array")
modeldir = 'models/SAC'
logdir = 'logs'

model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
  save_freq=1000,
  save_path="./logs/",
  name_prefix="sac_pendulum_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

TIMESTEPS=10000
for i in range(1, 30):
    model.learn(total_timesteps=TIMESTEPS, log_interval=4, tb_log_name='SAC_PENDULUM', reset_num_timesteps=False, callback=[checkpoint_callback])
    model.save(f"{modeldir}/{TIMESTEPS*i}")

# del model # remove to demonstrate saving and loading

# model = SAC.load("sac_pendulum")

# obs, info = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         obs, info = env.reset()