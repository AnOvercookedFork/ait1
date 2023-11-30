import gymnasium as gym
import dqn
import sys

def main():
    do_stuff()
    # env = gym.make("LunarLander-v2", render_mode="human")
    # train_env = gym.make('LunarLander-v2')
    # agent = dqn.DQN(train_env)
    # agent.train(200)
    # agent.replace_env(env)
    # agent.run_iteration(True)
    # train_env.close()
    # print("Action space:", env.action_space)
    # print("Observation space:", env.observation_space)
    # observation, info = env.reset(seed=42)
    # for _ in range(1000):
    #     action = env.action_space.sample()  # this is where you would insert your policy
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     if terminated or truncated:
    #         observation, info = env.reset()

    # env.close()

def do_stuff():
    env = gym.make("LunarLander-v2", render_mode="human")
    train_env = gym.make('LunarLander-v2')
    agent = dqn.DQN(train_env)
    while True:
        try:
            s = input('Command?')
            try:
                match s:
                    case 'exit':
                        print('exitting')
                        sys.exit(0)
                    case 'train':
                        num = int(input('Train iterations?'))
                        agent.replace_env(train_env)
                        agent.train(num)
                    case 'play':
                        agent.replace_env(env)
                        agent.run_iteration(True)
            except KeyboardInterrupt:
                print('Round interrupted')
                agent.reset_env()
        except KeyboardInterrupt:
            print('Interrupted')
            sys.exit(0)


if __name__ == '__main__':
    main()