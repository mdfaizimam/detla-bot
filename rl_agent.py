# --- detla-bot/rl_agent.py ---
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from trading_env import CryptoTradingEnv

log = logging.getLogger("rl_agent")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ContinuousActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim=1):
        super(ContinuousActorCritic, self).__init__()
        
        # Shared Feature Extractor
        self.base = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.Tanh()
        )
        
        # Actor Head (Mean & Std Dev)
        self.mu_head = nn.Linear(64, action_dim)
        self.log_std_head = nn.Parameter(torch.zeros(1, action_dim)) # Learnable log_std
        
        # Critic Head (Value)
        self.critic = nn.Linear(64, 1)
        
    def forward(self, state):
        x = self.base(state)
        
        # Continuous Action: Mu (Mean) restricted to [-1, 1] via Tanh
        mu = torch.tanh(self.mu_head(x))
        std = torch.exp(self.log_std_head).expand_as(mu)
        
        value = self.critic(x)
        return mu, std, value
    
    def evaluate(self, state, action):
        mu, std, value = self.forward(state)
        dist = torch.distributions.Normal(mu, std)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_logprobs, value, dist_entropy

class PPOAgent:
    def __init__(self, state_dim, lr=0.0003, gamma=0.99, eps_clip=0.2, K_epochs=4):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ContinuousActorCritic(state_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ContinuousActorCritic(state_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.mse_loss = nn.MSELoss()
        
    def select_action(self, state, memory=None):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            mu, std, _ = self.policy_old(state)
        
        # Sample from Normal Distribution
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        action_clipped = torch.clamp(action, -1.0, 1.0) # Enforce bounds
        log_prob = dist.log_prob(action)
        
        if memory:
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(log_prob)
        
        return action_clipped.item(), log_prob

    def update(self, memory):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        if rewards.std() > 0:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        else:
            rewards = rewards - rewards.mean()

        # Convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach()
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.mse_loss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear memory
        memory.clear_memory()

class RLAgent:
    """Wrapper that creates the Environment and Agent"""
    def __init__(self, data, model_path="ppo_agent"):
        # We assume CryptoTradingEnv now handles continuous actions
        self.env = CryptoTradingEnv(data, continuous=True) 
        self.state_dim = self.env.state_dim
        self.agent = PPOAgent(state_dim=self.state_dim)
        self.model_path = model_path
        self.memory = Memory()
        
    def train(self, total_timesteps=10000, update_timestep=2000):
        log.info(f"Starting training for {total_timesteps} timesteps")
        
        timestep = 0
        state = self.env.reset()
        
        # Check if reset returns tuple (gym>=0.26)
        if isinstance(state, tuple):
            state = state[0]
            
        while timestep < total_timesteps:
            timestep += 1
            
            # Select action
            action, _ = self.agent.select_action(state, self.memory)
            
            # Execute action
            step_result = self.env.step(action)
            if len(step_result) == 5:
                next_state, reward, done, truncated, info = step_result
            else:
                next_state, reward, done, info = step_result
                truncated = False
            
            # Save data to memory
            self.memory.rewards.append(reward)
            self.memory.is_terminals.append(done)
            
            state = next_state
            
            # Update if update_timestep is reached
            if timestep % update_timestep == 0:
                self.agent.update(self.memory)
                log.info(f"Step {timestep}/{total_timesteps} - PPO Updated")
                
            if done:
                state = self.env.reset()
                if isinstance(state, tuple):
                    state = state[0]
                    
        log.info("Training complete")
        self.save()

    def backtest(self, df=None):
        log.info("Starting backtest...")
        
        # If new data is provided, update the environment
        if df is not None:
             log.info(f"🔄 Switching environment to validation data ({len(df)} rows)")
             self.env = CryptoTradingEnv(df, continuous=True)
             
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]
            
        done = False
        total_reward = 0
        trades = 0
        
        while not done:
            action, _ = self.agent.select_action(state)
            step_result = self.env.step(action)
            if len(step_result) == 5:
                next_state, reward, done, truncated, info = step_result
            else:
                next_state, reward, done, info = step_result
                truncated = False            
            state = next_state
            total_reward += reward
            trades += 1
            
        results = {
            "Total Reward": total_reward,
            "Total Steps": trades,
            "Final Sortino": total_reward / (trades + 1e-6) # Approximate
        }
        return results

    def save(self):
        torch.save(self.agent.policy.state_dict(), f"{self.model_path}.pth")
        
    def load(self):
        self.agent.policy.load_state_dict(torch.load(f"{self.model_path}.pth"))
        self.agent.policy_old.load_state_dict(self.agent.policy.state_dict())
