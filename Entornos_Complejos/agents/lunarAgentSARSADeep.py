import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from agents.agent import Agent
import gymnasium as gym

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Q-values


class SemiGradientSarsaAgent(Agent):

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.99,
        decay_type: str = "linear",
    ):
        super().__init__(
            env=env,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
            decay_type=decay_type,
        )

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.q_network = QNetwork(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate) # alpha * grad loss = alpha * (r + gamma * q(S', A', w) - q(S, A, w)) * grad q(S, A, w)
        self.loss_fn = nn.MSELoss()

        torch.manual_seed(42)

    def get_action(self, obs):
        """
        Epsilon-greedy.
        Devuelve (action, exploring_bool)
        """

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample(), True

        state_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        return int(torch.argmax(q_values).item()), False

    def update(
        self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs,
    ):
        """
        Semi-gradient SARSA:
        w <- w + alpha [ r + gamma Q(s',a',w) - Q(s,a,w) ] * gradient of Q(s,a,w)
        """

        state_tensor = torch.FloatTensor(obs).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_obs).unsqueeze(0)

        # Elegimos next_action usando política actual (SARSA)
        next_action, _ = self.get_action(next_obs)

        # 1. Calculamos q(S, A, w)
        q_values = self.q_network(state_tensor) # Devuelve un tensor con los Q-values para todas las acciones
        q_sa = q_values[0][action] # Extraemos el Q-value específico para la acción tomada. El 0 es porque el tensor tiene forma [1, action_dim].

        # 2. Calculamos el objetivo (Target): R + gamma * q(S', A', w)
        with torch.no_grad():
            reward_tensor = torch.tensor(reward, dtype=torch.float32)
            if terminated:
                target = reward_tensor # w <- w + alpha [ r - Q(s,a,w) ] * gradient de Q(s,a,w) si el episodio terminó en el siguiente estado
            else:
                next_q_values = self.q_network(next_state_tensor)
                next_q_sa = next_q_values[0][next_action]
                target = reward_tensor + self.discount_factor * next_q_sa

        # 3. Calculamos la pérdida y actualizamos pesos (Descenso de gradiente)
        loss = self.loss_fn(q_sa, target) 

        self.optimizer.zero_grad()
        loss.backward() # \text{loss} = (q_{sa} - target)^2 = (Q(s,a,w) - (r + \gamma Q(s',a',w)))^2 => \nabla_w \text{loss} = 2 (Q(s,a,w) - target) \nabla_w Q(s,a,w) = -2 \delta \nabla_w Q(s,a,w)
        self.optimizer.step() # w <- w + alpha * grad loss 

        self.training_error.append(loss.item())

    def get_current_policy(self):
        """
        Devuelve la acción greedy para cada estado.
        Solo válido si el espacio es discreto y enumerable.
        """

        n_states = self.env.observation_space.n
        policy = np.zeros(n_states, dtype=int)

        for state in range(n_states):
            state_tensor = torch.FloatTensor([state]).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            policy[state] = int(torch.argmax(q_values).item())

        return policy

    # def decay_epsilon(self):
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay