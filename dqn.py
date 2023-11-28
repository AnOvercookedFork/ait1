import tensorflow as tf
import numpy as np
import gymnasium as gym
import random

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
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(env.action_space.n, activation='linear')
        ])
        self.model.compile(optimizer='adam', loss='mse')

        self.buffer = ReplayBuffer(100000)

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.samples_per_train = 10
        self.batch_size = 1000

        self.env = env
        self.state = None
        self.done = False
        self.reset_env()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_env(self):
        self.state = self.env.reset()

    def run_step(self):
        action = np.argmax(self.model.predict(self.state)) if np.random.rand() > self.epsilon else self.env.action_space.sample()
        next_state, reward, terminated, truncated, _, _ = self.env.step(action)
        self.buffer.add((self.state, action, reward, next_state, terminated or truncated))
        self.state = next_state
        self.done = terminated or truncated

    def run_iteration(self):
        while not self.done:
            self.run_step()
    
    def train_network(self):
        minibatch = self.buffer.sample(self.batch_size)
        for state, action, reward, next, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def train(self, iterations = 1000):
        for i in range(1, iterations + 1):
            if self.buffer.size() >= self.batch_size: # nested for branch prediction efficiency
                if i % self.samples_per_train == 0:
                    self.train_network()
