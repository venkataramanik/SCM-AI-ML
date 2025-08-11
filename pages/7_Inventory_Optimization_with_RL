import streamlit as st
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Inventory Optimization with RL"
)

st.title("Inventory Optimization with Reinforcement Learning")
st.write("---")

# -- Business Context Section --
st.subheader("Business Context")
st.write("""
Reinforcement Learning offers a dynamic approach to complex supply chain problems. Instead of relying on static rules or historical data, an RL agent learns to make optimal decisions in a dynamic environment, adapting to changing demand, costs, and lead times.
""")

# -- The Concept: Reinforcement Learning --
st.subheader("Reinforcement Learning")
st.write("""
We'll train an **RL agent** to manage an inventory system. The agent's goal is to learn a policy that minimizes total costs by balancing the cost of holding inventory with the cost of stockouts. The agent learns by receiving **rewards** (penalties for costs) and adjusting its behavior over thousands of simulated inventory cycles.
""")

# -- Tools Used Section --
st.subheader("Tools Used")
st.write("""
- **Gymnasium**: The industry-standard toolkit for developing and comparing RL algorithms.
- **Stable Baselines3**: A reliable set of RL algorithms, including **PPO** (Proximal Policy Optimization), which we use to train our agent.
- **Pandas** and **Matplotlib** for data visualization.
""")

# -- RL Environment Setup --
class InventoryEnv(gym.Env):
    def __init__(self, demand_mean, holding_cost, stockout_cost, max_inventory, max_demand):
        super(InventoryEnv, self).__init__()
        self.demand_mean = demand_mean
        self.holding_cost = holding_cost
        self.stockout_cost = stockout_cost
        self.max_inventory = max_inventory
        self.max_demand = max_demand
        
        self.action_space = spaces.Discrete(3)  # Actions: 0 (Hold), 1 (Order 100), 2 (Order 200)
        self.observation_space = spaces.Box(low=np.array([0]), high=np.array([max_inventory]), dtype=np.int32)
        
        self.state = self.max_inventory // 2  # Starting inventory
        self.current_step = 0
        self.total_cost = 0

    def step(self, action):
        order_quantity = action * 100
        demand = int(np.random.poisson(self.demand_mean))
        
        # Update inventory based on order and demand
        self.state += order_quantity
        
        if self.state >= demand:
            cost = self.holding_cost * (self.state - demand)
            self.state -= demand
        else:
            cost = self.holding_cost * self.state + self.stockout_cost * (demand - self.state)
            self.state = 0
            
        reward = -cost
        self.total_cost += cost
        self.current_step += 1
        
        terminated = self.current_step >= 365
        truncated = False
        info = {}
        
        return np.array([self.state], dtype=np.int32), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.max_inventory // 2
        self.current_step = 0
        self.total_cost = 0
        return np.array([self.state], dtype=np.int32), {}
        
# -- Code and Model Demonstration --
@st.cache_resource
def train_ppo_agent(total_timesteps):
    st.write("Training PPO agent...")
    env = InventoryEnv(demand_mean=50, holding_cost=0.5, stockout_cost=5.0, max_inventory=1000, max_demand=100)
    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=None, n_steps=256)
    model.learn(total_timesteps=total_timesteps)
    st.write("Training complete!")
    return model

@st.cache_data
def simulate_policy(model, env, num_episodes=5):
    all_costs = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_costs = []
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_costs.append(-reward) # Reward is negative cost, so we flip it
        all_costs.append(np.cumsum(episode_costs))
    
    return all_costs

@st.cache_data
def simulate_baseline(env, num_episodes=5):
    all_costs = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_costs = []
        terminated = False
        truncated = False
        while not terminated and not truncated:
            # Simple baseline: order 100 units if stock is below 500
            action = 1 if obs[0] < 500 else 0
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_costs.append(-reward)
        all_costs.append(np.cumsum(episode_costs))
    return all_costs

if st.button("Train Agent and Run Simulation"):
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()

st.subheader("RL Agent vs. Simple Baseline")
st.write("Watch how a trained RL agent performs compared to a simple, rule-based inventory policy.")

# Parameters for the environment and training
timesteps = st.slider("Training Timesteps (thousands)", min_value=1, max_value=20, value=5)
total_timesteps = timesteps * 1000

env_params = {
    "demand_mean": 50,
    "holding_cost": 0.5,
    "stockout_cost": 5.0,
    "max_inventory": 1000,
    "max_demand": 100
}

# Training and simulation
with st.spinner("Training RL agent... this may take a moment."):
    trained_model = train_ppo_agent(total_timesteps)

simulated_env = InventoryEnv(**env_params)
rl_costs = simulate_policy(trained_model, simulated_env)
baseline_costs = simulate_baseline(simulated_env)

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))
for costs in baseline_costs:
    ax.plot(costs, color='gray', linestyle='--', alpha=0.5)
ax.plot(baseline_costs[0], color='gray', linestyle='--', label='Baseline Policy')

for costs in rl_costs:
    ax.plot(costs, color='blue', alpha=0.5)
ax.plot(rl_costs[0], color='blue', label='RL Agent Policy')

ax.set_title("Cumulative Inventory Costs over 1 Year")
ax.set_xlabel("Day")
ax.set_ylabel("Cumulative Cost")
ax.legend()
ax.grid(True)
st.pyplot(fig)
