#!/usr/bin/env python3
"""

Kenneth Berry III
04/03/2025
Demo for ECE-6474 Class of Apr 3rd

apr3_trainDPG_DDPG.py

This script trains two agents on the Pendulum-v1 environment concurrently:
1. DDPG: Full implementation with replay buffer, target networks, and exploration noise.
2. DPG: A distinct agent that does not use deep networks. Its actor is a linear policy
   (with weights and bias) and its critic is a simple linear function learned via TD updates.
   
Both agents share the same model subdirectory (generated using a common timestamp).
Logs per-episode metrics to separate log files and saves model checkpoints.
A multiprocessing lock is used to synchronize log file writes.
"""

import os
import time
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from multiprocessing import Process, Lock

# Fix for numpy bool8 error
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_


# Utility functions

def get_timestamp():
    return time.strftime("%Y%m%d-%H%M%S")

def save_checkpoint(model, filename):
    torch.save(model.state_dict(), filename)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


# Neural Network Architectures for DDPG

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_sizes=(256, 256)):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], action_dim),
            nn.Tanh()  # bound actions between -1 and 1
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(256, 256)):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.net(x)

class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.storage = deque(maxlen=max_size)

    def add(self, data):
        self.storage.append(data)

    def sample(self, batch_size):
        batch = random.sample(self.storage, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done).unsqueeze(1)
        )


# DDPG Agent

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, use_replay=True, use_target=True,
                 actor_lr=1e-3, critic_lr=1e-3, tau=0.005, gamma=0.99, noise_std=0.1):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.tau = tau
        self.gamma = gamma
        self.max_action = max_action

        self.use_replay = use_replay
        self.use_target = use_target

        if self.use_replay:
            self.replay_buffer = ReplayBuffer()
        else:
            self.replay_buffer = None  # Uses only the latest transition

        # noise parameter
        self.noise_std = noise_std

    def select_action(self, state, noise=True):
        if isinstance(state, tuple):
            state = state[0]
        state = torch.FloatTensor(np.array(state).reshape(1, -1))
        action = self.actor(state).detach().cpu().numpy().flatten()
        if noise:
            action = action + np.random.normal(0, self.noise_std, size=action.shape)
        return np.clip(action, -self.max_action, self.max_action)

    def train(self, batch_size=64):
        if self.use_replay:
            state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        else:
            state, action, reward, next_state, done = (self.last_state, self.last_action,
                                                       self.last_reward, self.last_next_state,
                                                       self.last_done)

        with torch.no_grad():
            if self.use_target:
                next_action = self.actor_target(next_state)
                target_Q = self.critic_target(next_state, next_action)
            else:
                next_action = self.actor(next_state)
                target_Q = self.critic(next_state, next_action)
            target_Q = reward + ((1 - done) * self.gamma * target_Q).detach()

        current_Q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.use_target:
            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)

        avg_Q = current_Q.mean().item()
        return actor_loss.item(), critic_loss.item(), avg_Q


# Non–NN DPG Agent
# A simple linear policy and linear critic using numpy.

class LinearActor:
    def __init__(self, state_dim, action_dim, max_action):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.W = np.random.randn(action_dim, state_dim) * 0.01  # small random init
        self.b = np.zeros(action_dim)

    def predict(self, state):
        action = np.dot(self.W, state) + self.b
        return np.clip(action, -self.max_action, self.max_action)

    def update(self, state, grad_from_critic, lr):
        self.W += lr * np.outer(grad_from_critic, state)
        self.b += lr * grad_from_critic

class LinearCritic:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.weights = np.zeros(state_dim + action_dim + 1)  # extra term for bias

    def predict(self, state, action):
        feature = np.concatenate([state, action, [1.0]])
        return np.dot(self.weights, feature)

    def update(self, state, action, target, lr):
        feature = np.concatenate([state, action, [1.0]])
        pred = np.dot(self.weights, feature)
        error = target - pred
        self.weights += lr * error * feature
        return error

class DPGAgentNoNN:
    def __init__(self, state_dim, action_dim, max_action, actor_lr=1e-3, critic_lr=1e-3, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.actor = LinearActor(state_dim, action_dim, max_action)
        self.critic = LinearCritic(state_dim, action_dim)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma

    def select_action(self, state):
        return self.actor.predict(state)

    def train_step(self, state, action, reward, next_state, done):
        next_action = self.actor.predict(next_state)
        target = reward
        if not done:
            target += self.gamma * self.critic.predict(next_state, next_action)
        critic_error = self.critic.update(state, action, target, self.critic_lr)
        # Gradient from critic (portion corresponding to action)
        grad_from_critic = self.critic.weights[self.state_dim:self.state_dim + self.action_dim]
        self.actor.update(state, grad_from_critic, self.actor_lr)
        return critic_error


# Training Functions

def train_agent(agent, env, num_episodes=200, batch_size=64, agent_name="agent",
                log_dir="logs", model_dir="models", run_date=None, log_lock=None):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    # Use the common run_date for the model subfolder.
    model_subfolder = os.path.join(model_dir, run_date)
    os.makedirs(model_subfolder, exist_ok=True)
    # Use the common run_date for the log filename.
    log_filename = os.path.join(log_dir, f"{agent_name}_log_{get_timestamp()}.csv")
    if log_lock:
        with log_lock:
            with open(log_filename, "w") as log_file:
                log_file.write("episode,episode_reward,actor_loss,critic_loss,avg_Q\n")
    else:
        with open(log_filename, "w") as log_file:
            log_file.write("episode,episode_reward,actor_loss,critic_loss,avg_Q\n")
    
    episode_rewards = []
    for episode in range(1, num_episodes+1):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            episode_reward += reward

            if agent.use_replay:
                agent.replay_buffer.add((state, action, reward, next_state, float(done)))
            else:
                agent.last_state = torch.FloatTensor(np.array(state).reshape(1, -1))
                agent.last_action = torch.FloatTensor(np.array(action).reshape(1, -1))
                agent.last_reward = torch.FloatTensor([reward])
                agent.last_next_state = torch.FloatTensor(np.array(next_state).reshape(1, -1))
                agent.last_done = torch.FloatTensor([float(done)])

            state = next_state

            if agent.use_replay:
                if len(agent.replay_buffer.storage) >= batch_size:
                    a_loss, c_loss, avg_Q = agent.train(batch_size)
            else:
                a_loss, c_loss, avg_Q = agent.train(batch_size)

        if log_lock:
            with log_lock:
                with open(log_filename, "a") as log_file:
                    log_file.write(f"{episode},{episode_reward:.3f},{a_loss:.6f},{c_loss:.6f},{avg_Q:.6f}\n")
        else:
            with open(log_filename, "a") as log_file:
                log_file.write(f"{episode},{episode_reward:.3f},{a_loss:.6f},{c_loss:.6f},{avg_Q:.6f}\n")
        episode_rewards.append(episode_reward)
        print(f"[{agent_name}] Episode {episode} - Reward: {episode_reward:.3f} | Actor Loss: {a_loss:.6f} | Critic Loss: {c_loss:.6f} | Avg Q: {avg_Q:.6f}")

        if episode % 100 == 0:
            # Save checkpoints periodically.
            if episode == num_episodes+1:
                actor_filename = os.path.join(model_subfolder, f"{agent_name}_actor_Efinal.pt")
                critic_filename = os.path.join(model_subfolder, f"{agent_name}_critic_Efinal.pt")
                save_checkpoint(agent.actor, actor_filename)
                save_checkpoint(agent.critic, critic_filename)
            else:
                actor_filename = os.path.join(model_subfolder, f"{agent_name}_actor_E{episode}.pt")
                critic_filename = os.path.join(model_subfolder, f"{agent_name}_critic_E{episode}.pt")                
                save_checkpoint(agent.actor, actor_filename)
                save_checkpoint(agent.critic, critic_filename)
    
    np.save(os.path.join("logs", f"{agent_name}_rewards.npy"), np.array(episode_rewards))
    return episode_rewards, log_filename

def train_agent_dpg(agent, env, num_episodes=200, log_dir="logs", model_dir="models", run_date=None, log_lock=None):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    # Uses the common run_date here too.
    model_subfolder = os.path.join(model_dir, run_date)
    os.makedirs(model_subfolder, exist_ok=True)
    log_filename = os.path.join(log_dir, f"DPG_log_{run_date}.csv")
    if log_lock:
        with log_lock:
            with open(log_filename, "w") as log_file:
                log_file.write("episode,episode_reward,critic_error,avg_Q\n")
    else:
        with open(log_filename, "w") as log_file:
            log_file.write("episode,episode_reward,critic_error,avg_Q\n")
    
    episode_rewards = []
    for episode in range(1, num_episodes+1):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        episode_reward = 0
        critic_errors = []
        done = False
        while not done:
            action = agent.select_action(np.array(state).flatten())
            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            episode_reward += reward
            critic_error = agent.train_step(np.array(state).flatten(), np.array(action).flatten(),
                                              reward, np.array(next_state).flatten(), done)
            critic_errors.append(critic_error)
            state = next_state
        avg_critic_error = np.mean(critic_errors) if critic_errors else 0.0
        last_action = agent.actor.predict(np.array(state).flatten())
        avg_Q = agent.critic.predict(np.array(state).flatten(), last_action)
        if log_lock:
            with log_lock:
                with open(log_filename, "a") as log_file:
                    log_file.write(f"{episode},{episode_reward:.3f},{avg_critic_error:.6f},{avg_Q:.6f}\n")
        else:
            with open(log_filename, "a") as log_file:
                log_file.write(f"{episode},{episode_reward:.3f},{avg_critic_error:.6f},{avg_Q:.6f}\n")
        episode_rewards.append(episode_reward)
        print(f"[DPG] Episode {episode} - Reward: {episode_reward:.3f} | Avg Critic Error: {avg_critic_error:.6f} | Avg Q: {avg_Q:.6f}")

        if episode % 100 == 0 or episode == num_episodes:
            actor_filename = os.path.join(model_subfolder, f"DPG_actor_E{episode}.npz")
            np.savez(actor_filename, W=agent.actor.W, b=agent.actor.b)
    
    np.save(os.path.join("logs", "dpg_rewards.npy"), np.array(episode_rewards))
    return episode_rewards, log_filename

def train_ddpg_agent(num_episodes=200, batch_size=64, run_date=None, log_lock=None):
    # Use Pendulum-v1
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    ddpg_agent = DDPGAgent(state_dim, action_dim, max_action, use_replay=True, use_target=True, noise_std=0.1)
    ddpg_rewards, _ = train_agent(ddpg_agent, env, num_episodes, batch_size,
                                  agent_name="DDPG", log_dir="logs", model_dir="models", run_date=run_date, log_lock=log_lock)
    np.save(os.path.join("logs", "ddpg_rewards.npy"), np.array(ddpg_rewards))

def train_dpg_agent(num_episodes=200, run_date=None, log_lock=None):
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    dpg_agent = DPGAgentNoNN(state_dim, action_dim, max_action, actor_lr=1e-3, critic_lr=1e-3, gamma=0.99)
    dpg_rewards, _ = train_agent_dpg(dpg_agent, env, num_episodes,
                                     log_dir="logs", model_dir="models", run_date=run_date, log_lock=log_lock)    # Rewards are saved within train_agent_dpg


# Main training routine (using multiprocessing)

def main():
    num_episodes = 2000
    batch_size = 64
    # Generate one common timestamp for this run.
    common_run_date = get_timestamp()

    # Create a shared lock for logging
    log_lock = Lock()
    
    # Create two processes for concurrent training, passing the shared lock and common_run_date
    p1 = Process(target=train_ddpg_agent, args=(num_episodes, batch_size, common_run_date, log_lock))
    p2 = Process(target=train_dpg_agent, args=(num_episodes, common_run_date, log_lock))
    
    p1.start()
    p2.start()
    
    p1.join()
    p2.join()

if __name__ == "__main__":
    main()
