import sys
import gc
import gymnasium as gym
import dqn
# import cProfile
# import pstats
# from io import StringIO

def main():
    do_stuff()

def do_stuff():
    env = gym.make("BipedalWalker", render_mode="human")
    train_env = gym.make('BipedalWalker ')
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
                        # pr = cProfile.Profile()
                        # pr.enable()
                        agent.train(num)
                        # pr.disable()
                        # s = StringIO()
                        # sortby = 'time'  # or 'time' to sort by total time in each function
                        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                        # ps.print_stats(10)  # Set the threshold time
                        # print(s.getvalue())
                    case 'play':
                        agent.replace_env(env)
                        agent.run_iteration(True)
                    case 'save':
                        s = input('Path to save to?\n')
                        if s[0:6] == 'models':
                            agent.save_model(s)
                    case 'load':
                        s = input('Path to load from?\n')
                        if s[0:6] == 'models':
                            agent.load_model_from_file(s)
                    case 'gc':
                        gc.collect()
                    case 'exit':
                        break
            except KeyboardInterrupt:
                print('Round interrupted')
                agent.reset_env()
        except KeyboardInterrupt:
            print('Interrupted')
            env.close()
            train_env.close()
            sys.exit(0)
    env.close()
    train_env.close()


if __name__ == '__main__':
    main()