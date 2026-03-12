# Importamos las librerías necesarias
import gymnasium as gym  # Biblioteca para entornos de aprendizaje por refuerzo.
import torch  # PyTorch: manejo de tensores y redes neuronales.
import torch.nn as nn  # Módulo para definir modelos de redes neuronales.
import torch.nn.functional as F  # Funciones de activación y utilidades de PyTorch.
import torch.optim as optim
import numpy as np
from agents.agent import Agent

# Código de esta clase sacado de https://github.com/ldaniel-hm/eml_approximate/blob/0730b02e2a973683f400f71d60e92ac24a403d55/SolucionDeepSARSA_CartPole.ipynb#L345
class QNetworkSARSA(nn.Module):
    """
    Red neuronal para aproximar la función Q.

    Parámetros:
      - state_dim (int): Dimensión del estado.
      - action_dim (int): Número de acciones posibles.
      - hidden_dim (int): Número de neuronas en las capas ocultas.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetworkSARSA, self).__init__()
        # Primera capa: de estado a capa oculta de tamaño hidden_dim.
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        # Segunda capa oculta.
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Capa de salida: de hidden_dim a número de acciones.
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """
        Propagación hacia adelante.

        Parámetro:
          - x (Tensor): Estado de entrada con forma [batch_size, state_dim].

        Retorna:
          - Tensor: Valores Q para cada acción, con forma [batch_size, action_dim].
        """
        # Aplicar la primera capa seguida de ReLU.
        x = F.relu(self.fc1(x))
        # Aplicar la segunda capa seguida de ReLU.
        x = F.relu(self.fc2(x))
        # Capa de salida sin activación, para obtener los valores Q.
        return self.fc3(x)  # Q-values


class SemiGradientSarsaDeepAgent(Agent):

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.99,
        decay_type: str = "linear",
        hidden_dim: int = 128,
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

        torch.manual_seed(42)
        self.q_network = QNetworkSARSA(self.state_dim, self.action_dim, hidden_dim=hidden_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate) # alpha * grad loss = alpha * (r + gamma * q(S', A', w) - q(S, A, w)) * grad q(S, A, w)
        self.loss_fn = nn.MSELoss()

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
        next_action: int,
    ):
        """
        Semi-gradient SARSA:
        w <- w + alpha [ r + gamma Q(s',a',w) - Q(s,a,w) ] * gradient of Q(s,a,w)
        """

        state_tensor = torch.FloatTensor(obs).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_obs).unsqueeze(0)

        # Elegimos next_action usando política actual (SARSA)
        # next_action, _ = self.get_action(next_obs)

        # 1. Calculamos q(S, A, w)
        q_values = self.q_network(state_tensor) # Devuelve un tensor con los Q-values para todas las acciones
        q_sa = q_values[0, action].unsqueeze(0) # Extraemos el Q-value específico para la acción tomada. El 0 es porque el tensor tiene forma [1, action_dim]. Quitar un

        # 2. Calculamos el objetivo (Target): R + gamma * q(S', A', w)
        with torch.no_grad():
            reward_tensor = torch.tensor([reward], dtype=torch.float32) # Quitar un
            if terminated:
                target = reward_tensor # w <- w + alpha [ r - Q(s,a,w) ] * gradient de Q(s,a,w) si el episodio terminó en el siguiente estado
            else:
                next_q_values = self.q_network(next_state_tensor)
                next_q_sa = next_q_values[0, next_action].unsqueeze(0) # Quitar un
                target = reward_tensor + self.discount_factor * next_q_sa

        # 3. Calculamos la pérdida y actualizamos pesos (Descenso de gradiente)
        loss = self.loss_fn(q_sa, target) 

        self.optimizer.zero_grad()
        loss.backward() # \text{loss} = (q_{sa} - target)^2 = (Q(s,a,w) - (r + \gamma Q(s',a',w)))^2 => \nabla_w \text{loss} = 2 (Q(s,a,w) - target) \nabla_w Q(s,a,w) = -2 \delta \nabla_w Q(s,a,w)
        self.optimizer.step() # w <- w + alpha * grad loss 

        self.training_error.append(loss.item())

    def get_current_policy(self):
        pass

    def test(self, num_episodes=1000):
        """Test agent performance without learning or exploration."""
        total_rewards = []
        env = self.env
        # Temporarily disable exploration for testing
        old_epsilon = self.epsilon
        self.epsilon = 0.01  # Pure exploitation

        for t in range(num_episodes):
            seed = num_episodes*2 + t
            obs, _ = env.reset(seed=seed)
            episode_reward = 0
            done = False

            while not done:
                action, _ = self.get_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated

            total_rewards.append(episode_reward)

        # Restore original epsilon
        self.epsilon = old_epsilon

        win_rate = np.mean(np.array(total_rewards) > 200)
        average_reward = np.mean(total_rewards)
        standard_deviation = np.std(total_rewards)

        print(f"Test Results over {num_episodes} episodes:")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Average Reward: {average_reward:.3f}")
        print(f"Standard Deviation: {standard_deviation:.3f}")
        return win_rate, average_reward, standard_deviation
    
    # def decay_epsilon(self):
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay