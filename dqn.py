import tensorflow as tf
import numpy as np
import gymnasium as gym
import random
import tqdm
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []

    def size(self):
        return len(self.buffer)

    def add(self, experience):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

class DQN:
    def __init__(self, env: gym.Env):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_dim=env.observation_space.shape[0]),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(env.action_space.n, activation='linear')
        ])
        self.model.compile(optimizer='adam', loss='mse')

        self.buffer = ReplayBuffer(50000)

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.995
        self.samples_per_train = 1
        self.batch_size = 5000
        self.steps = 0
        self.step_penalty = 0

        self.env = env
        self.state = None
        self.done = False
        self.reset_env()
    
    def replace_env(self, env): # ONLY WORKS WITH SAME ENV TYPE
        self.env = env
        self.steps = 0
        self.reset_env()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_env(self):
        self.state, _ = self.env.reset()
        self.done = False

    def run_step(self, inference = False):
        # print(f'state: {self.state}')
        action = None
        if inference or np.random.rand() > self.epsilon:
            # print('predicting...')
            # print(f'state shape: {self.state.shape}')
            preds = self.model.predict(np.reshape(self.state, (1, -1)), verbose=0)[0]
            action = np.argmax(preds)
            if inference:
                print(f'Max Q-val: {np.amax(preds)}')
        else:
            action = self.env.action_space.sample()
        # action = np.argmax(self.model.predict(self.state)[0]) if inference or np.random.rand() > self.epsilon else self.env.action_space.sample()
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        self.buffer.add((self.state, action, reward, next_state, terminated or truncated, self.steps))
        self.state = next_state
        self.done = terminated or truncated

    def run_iteration(self, inference = False):
        for _ in range(0, 5000):
            self.steps += 1
            if not self.done:
                self.run_step(inference)
        self.reset_env()
    
    def train_network(self):
        # print('training...')
        minibatch = self.buffer.sample(self.batch_size)
        states = np.array([np.reshape(state, (1, -1))[0] for state, _, _, _, _, _ in minibatch])
        next_states = np.array([np.reshape(next_state, (1, -1))[0] for _, _, _, next_state, _, _ in minibatch])
        steps = [a for _, _, _, _, _, a in minibatch]
        q_values = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)
        targets = q_values.copy()
        for i, (_, action, reward, _, done, _) in enumerate(minibatch):
            target = reward #- steps[i] * self.step_penalty
            if not done:
                target += self.gamma * np.amax(next_q_values[i])
            targets[i, action] = target
        self.model.fit(states, targets, epochs=5, verbose=0)

    def train(self, iterations = 1000):
        for i in tqdm.tqdm(range(0, iterations)):
            # print(i)
            if self.buffer.size() >= self.batch_size: # nested for branch prediction efficiency
                if (i + 1) % self.samples_per_train == 0:
                    self.train_network()
            self.run_iteration()
            self.decay_epsilon()
