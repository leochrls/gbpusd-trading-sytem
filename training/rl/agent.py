"""
Agent DQN leger pour RTX Ada 1000.
Architecture [128, 64] max, batch_size=32, buffer=10k.
"""

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger


# ============================================================
# RESEAU DE NEURONES
# ============================================================

class DQNNetwork(nn.Module):
    """
    Reseau DQN leger.
    Architecture fixe [input -> 128 -> 64 -> 3 actions].
    Optimise pour RTX Ada 1000 (6GB VRAM).
    """

    def __init__(self, state_dim: int, n_actions: int = 3) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, n_actions),
        )

        # Initialisation Xavier
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# REPLAY BUFFER
# ============================================================

@dataclass
class Transition:
    """Une transition (s, a, r, s', done)."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Buffer d'experience pour DQN.
    Taille limitee a 10 000 pour RTX Ada 1000.
    """

    def __init__(self, capacity: int = 10_000) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)
        self.capacity = capacity

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        transitions = random.sample(list(self.buffer), batch_size)
        states = np.array([t.state for t in transitions], dtype=np.float32)
        actions = np.array([t.action for t in transitions], dtype=np.int64)
        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        next_states = np.array([t.next_state for t in transitions], dtype=np.float32)
        dones = np.array([t.done for t in transitions], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def is_ready(self) -> bool:
        """Buffer pret quand il contient au moins batch_size elements."""
        return len(self.buffer) >= 64


# ============================================================
# AGENT DQN
# ============================================================

class DQNAgent:
    """
    Agent DQN avec target network et epsilon-greedy.

    Optimise RTX Ada 1000 :
    - Reseau [128, 64]
    - batch_size = 32
    - buffer = 10 000
    - CUDA si dispo, CPU sinon

    Attributes:
        state_dim: Dimension du vecteur d'etat.
        n_actions: Nombre d'actions possibles.
        gamma: Facteur de discount.
        epsilon: Taux d'exploration courant.
        batch_size: Taille du batch d'entrainement.
        policy_net: Reseau de politique.
        target_net: Reseau cible.
        memory: Replay buffer.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int = 3,
        gamma: float = 0.99,
        lr: float = 1e-4,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
        buffer_size: int = 10_000,
        target_update_freq: int = 100,
        seed: int = 42,
    ) -> None:
        # Setup device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        logger.info(f"Device : {self.device}")

        if self.device.type == 'cuda':
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU : {torch.cuda.get_device_name(0)} ({vram:.1f}GB)")

        # Parametres
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps_done = 0

        # Seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Reseaux
        self.policy_net = DQNNetwork(state_dim, n_actions).to(self.device)
        self.target_net = DQNNetwork(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimiseur
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=lr,
            weight_decay=1e-5,
        )

        # Buffer
        self.memory = ReplayBuffer(buffer_size)

        # Monitoring
        self.losses: list[float] = []
        self.q_values: list[float] = []

        logger.info(
            f"DQNAgent initialise : "
            f"state_dim={state_dim} | "
            f"batch={batch_size} | "
            f"buffer={buffer_size} | "
            f"gamma={gamma}"
        )

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Selectionne une action (epsilon-greedy en training, greedy en eval).

        Args:
            state: Vecteur d'observation
            training: Si True, utilise epsilon-greedy

        Returns:
            Action (0, 1 ou 2)
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            self.q_values.append(q_values.max().item())
            return int(q_values.argmax().item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Stocke une transition dans le replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def optimize(self) -> Optional[float]:
        """
        Optimise le reseau sur un batch.

        Returns:
            Loss (float) ou None si buffer pas pret
        """
        if len(self.memory) < self.batch_size:
            return None

        # Echantillonnage
        states, actions, rewards, next_states, dones = \
            self.memory.sample(self.batch_size)

        # Conversion tenseurs
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Q-values courants
        current_q = self.policy_net(states_t).gather(
            1, actions_t.unsqueeze(1),
        ).squeeze(1)

        # Q-values cibles (target network)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        # Huber loss (robuste aux outliers)
        loss = nn.SmoothL1Loss()(current_q, target_q)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (stabilite)
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

        self.optimizer.step()

        loss_val = loss.item()
        self.losses.append(loss_val)

        # Mise a jour target network
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            logger.debug(
                f"Target network mis a jour (step {self.steps_done})"
            )

        return loss_val

    def decay_epsilon(self) -> None:
        """Decroissance epsilon apres chaque episode."""
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay,
        )

    def save(self, path: str, episode: int = 0) -> None:
        """Sauvegarde l'agent complet.

        Args:
            path: Chemin du fichier .pth.
            episode: Numero de l'episode courant.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'episode': episode,
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'losses': self.losses[-1000:],
            'state_dim': self.state_dim,
            'n_actions': self.n_actions,
        }, path)
        logger.success(f"Agent sauvegarde : {path} (episode {episode})")

    @classmethod
    def load(cls, path: str, **kwargs) -> 'DQNAgent':
        """Charge un agent sauvegarde.

        Args:
            path: Chemin du fichier .pth.

        Returns:
            Instance DQNAgent chargee.
        """
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        agent = cls(
            state_dim=checkpoint['state_dim'],
            n_actions=checkpoint['n_actions'],
            **kwargs,
        )
        agent.policy_net.load_state_dict(checkpoint['policy_net'])
        agent.target_net.load_state_dict(checkpoint['target_net'])
        agent.optimizer.load_state_dict(checkpoint['optimizer'])
        agent.epsilon = checkpoint['epsilon']
        agent.steps_done = checkpoint['steps_done']

        logger.success(
            f"Agent charge : {path} "
            f"(episode {checkpoint['episode']})"
        )
        return agent

    def get_vram_usage(self) -> str:
        """Retourne l'utilisation VRAM actuelle."""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            return f"VRAM : {allocated:.2f}GB alloue / {reserved:.2f}GB reserve"
        return "CPU mode (pas de VRAM)"
