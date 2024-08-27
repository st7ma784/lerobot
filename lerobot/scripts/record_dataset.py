# example.py
import imageio
import gymnasium as gym
import numpy as np
import gym_aloha
import clip
from PIL import Image
model,preprocess=clip.load("ViT-B/32")
env = gym.make("gym_aloha/PromptTask-v0")
observation, info = env.reset()
frames = []
print(env.action_space)
for _ in range(1000):
    action = env.action_space.sample()
    image=env.render()
    #    imageio.imwrite("example.jpg", image)

    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()
    frames.append(image)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
imageio.mimsave("example.mp4", np.stack(frames), fps=25)