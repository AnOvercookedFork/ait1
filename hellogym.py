import gymnasium as gym

def main():
    env = gym.make("LunarLander-v2", render_mode="human")
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)
    observation, info = env.reset(seed=42)
    for _ in range(1000):
        action = env.action_space.sample()  # this is where you would insert your policy
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()

    env.close()

if __name__ == '__main__':
    main()