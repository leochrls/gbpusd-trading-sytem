"""
Tests unitaires pour l'environnement RL et l'agent DQN.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from features.pipeline import FEATURE_COLUMNS
from training.rl.agent import DQNAgent, ReplayBuffer
from training.rl.environment import TradingEnv


@pytest.fixture
def sample_df():
    """DataFrame M15 synthetique avec features."""
    n = 500
    np.random.seed(42)
    prices = 1.25 + np.cumsum(np.random.normal(0, 0.0002, n))

    df = pd.DataFrame(
        {col: np.random.normal(0, 1, n) for col in FEATURE_COLUMNS},
        index=pd.date_range('2022-01-01', periods=n, freq='15min'),
    )
    df['close'] = prices
    df['rsi_14'] = np.clip(50 + np.random.normal(0, 15, n), 0, 100)
    df['open'] = prices + np.random.normal(0, 0.0001, n)
    df['high'] = prices + abs(np.random.normal(0, 0.0003, n))
    df['low'] = prices - abs(np.random.normal(0, 0.0003, n))
    df['volume'] = np.random.randint(100, 1000, n).astype(float)

    return df


@pytest.fixture
def env(sample_df):
    return TradingEnv(sample_df)


@pytest.fixture
def agent(env):
    return DQNAgent(state_dim=env.n_state, n_actions=3, seed=42)


# === TESTS ENVIRONNEMENT ===

def test_env_reset_returns_correct_shape(env):
    """reset() doit retourner un state de bonne dimension."""
    obs, info = env.reset()
    assert obs.shape == (env.n_state,)
    assert isinstance(info, dict)


def test_env_step_returns_correct_types(env):
    """step() doit retourner les bons types."""
    env.reset()
    obs, reward, terminated, truncated, info = env.step(1)

    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_env_action_space(env):
    """Action space doit avoir 3 actions."""
    assert env.action_space.n == 3


def test_env_observation_space(env):
    """Observation space doit matcher n_state."""
    obs, _ = env.reset()
    assert env.observation_space.shape == (env.n_state,)
    assert obs.shape == env.observation_space.shape


def test_env_terminates(env):
    """L'episode doit se terminer."""
    obs, _ = env.reset()
    done = False
    steps = 0

    while not done and steps < 10000:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1

    assert done, "L'episode ne s'est jamais termine !"


def test_env_balance_positive(env):
    """Balance ne doit pas devenir negative en conditions normales."""
    obs, _ = env.reset()
    done = False

    while not done:
        obs, _, terminated, truncated, _ = env.step(1)  # HOLD
        done = terminated or truncated

    assert env.balance > 0


# === TESTS AGENT ===

def test_agent_select_action_range(agent, env):
    """Actions doivent etre dans {0, 1, 2}."""
    obs, _ = env.reset()
    for _ in range(20):
        action = agent.select_action(obs, training=True)
        assert action in [0, 1, 2]


def test_agent_greedy_deterministic(agent, env):
    """En mode greedy (training=False), meme state -> meme action."""
    obs, _ = env.reset()
    agent.epsilon = 0.0

    actions = [agent.select_action(obs, training=False) for _ in range(5)]
    assert len(set(actions)) == 1, "Greedy non deterministe !"


def test_replay_buffer_capacity(agent):
    """Buffer ne doit pas depasser sa capacite."""
    state = np.zeros(agent.state_dim, dtype=np.float32)

    for _ in range(15_000):
        agent.memory.push(state, 1, 0.0, state, False)

    assert len(agent.memory) <= agent.memory.capacity


def test_agent_optimize_returns_loss(agent, env):
    """optimize() doit retourner une loss apres remplissage buffer."""
    obs, _ = env.reset()

    # Remplit buffer
    for _ in range(100):
        action = agent.select_action(obs, training=True)
        next_obs, reward, done, _, _ = env.step(action)
        agent.store_transition(obs, action, reward, next_obs, done)
        obs = next_obs
        if done:
            obs, _ = env.reset()

    loss = agent.optimize()
    assert loss is not None
    assert np.isfinite(loss)


def test_agent_save_load(agent, tmp_path):
    """Save/load doit preserver les poids."""
    path = str(tmp_path / "test_agent.pth")
    agent.save(path, episode=10)

    loaded = DQNAgent.load(path)

    # Compare poids
    for p1, p2 in zip(
        agent.policy_net.parameters(),
        loaded.policy_net.parameters(),
    ):
        assert torch.allclose(p1, p2), "Poids differents apres load !"


def test_vram_under_limit(agent):
    """Verifie que le modele ne depasse pas 6GB VRAM."""
    if agent.device.type != 'cuda':
        pytest.skip("Test GPU uniquement")

    allocated_gb = torch.cuda.memory_allocated() / 1e9
    assert allocated_gb < 6.0, f"VRAM trop elevee : {allocated_gb:.2f}GB"
