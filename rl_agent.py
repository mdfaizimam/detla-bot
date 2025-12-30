import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from trading_env import CryptoTradingEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [PPO]: %(message)s")
log = logging.getLogger("rl_agent")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Shared featre extractor
        self.base = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        
        # Actor (Policy)
        self.actor = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic (Value)
        self.critic = nn.Linear(64, 1)
        
    def forward(self, state):
        x = self.base(state)
        probs = self.actor(x)
        value = self.critic(x)
        return probs, value

class PPOAgent:
    def __init__(self, state_dim=5, action_dim=3, lr=0.0003, gamma=0.99, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.mse_loss = nn.MSELoss()
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs, _ = self.policy_old(state)
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, memory):
        # Unpack memory
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal: discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        old_states = torch.squeeze(torch.stack(memory.states), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).detach()
        
        # Optimize policy for K epochs
        for _ in range(4):
            probs, state_values = self.policy(old_states)
            dist = torch.distributions.Categorical(probs)
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            state_values = torch.squeeze(state_values)
            
            ratios = torch.exp(logprobs - old_logprobs)
            
            advantages = rewards - state_values.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5*self.mse_loss(state_values, rewards) - 0.01*dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class RLAgent:
    """Wrapper to match previous API"""
    def __init__(self, data, model_path="ppo_agent"):
        self.env = CryptoTradingEnv(data)
        self.agent = PPOAgent(state_dim=self.env.state_dim)
        self.memory = Memory()
        self.model_path = model_path
        
    def train(self, total_timesteps=1000):
        log.info("Starting Custom PPO Training...")
        
        timestep = 0
        update_timestep = 200 # Update every 200 steps
        
        state, _ = self.env.reset()
        
        while timestep < total_timesteps:
            timestep += 1
            
            action, log_prob = self.agent.select_action(state)
            next_state, reward, done, _, _ = self.env.step(action)
            
            self.memory.states.append(torch.FloatTensor(state).unsqueeze(0))
            self.memory.actions.append(torch.tensor(action).unsqueeze(0))
            self.memory.logprobs.append(log_prob.unsqueeze(0))
            self.memory.rewards.append(reward)
            self.memory.is_terminals.append(done)
            
            state = next_state
            
            if timestep % update_timestep == 0:
                self.agent.update(self.memory)
                self.memory.clear()
                log.info(f"Step {timestep}: Policy Updated.")
                
            if done:
                state, _ = self.env.reset()
                
        log.info("Training Complete.")
        
    def backtest(self):
        log.info("Backtesting Custom PPO...")
        state, _ = self.env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = self.agent.select_action(state)
            state, reward, done, _, _ = self.env.step(action)
            total_reward += reward
            
        log.info(f"Backtest Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    import pandas as pd
    import os
    if os.path.exists("fused_data_sample.csv"):
        df = pd.read_csv("fused_data_sample.csv")
        agent = RLAgent(df)
        agent.train(1000)
