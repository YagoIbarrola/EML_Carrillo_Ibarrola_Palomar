from collections import defaultdict
import gymnasium as gym
import numpy as np
from agents.agent import Agent

class LunarAgentSARSA(Agent):
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.99, # Generalmente 0.99 para LunarLander
        decay_type: str = "linear",
    ):
        super().__init__(
            env=env,
            epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
            decay_type=decay_type,
        )

        # Ajustamos el learning rate por la cantidad de tilings (práctica recomendada)
        # Buscamos n_tilings en la cadena de wrappers
        # n_tilings = getattr(env, "n_tilings", None)
        # if n_tilings is None:
        #     # Si no está en la capa exterior, buscamos en el entorno envuelto
        #     n_tilings = getattr(env.unwrapped, "n_tilings", 8) # 8 por defecto

        # self.lr = learning_rate / n_tilings 
        # self.n_actions = env.action_space.n
        
        # Matriz de pesos W: [total_features, n_actions]
        # if hasattr(env, "total_features"):
        #     total_features = env.total_features
        # elif hasattr(env.unwrapped, "total_features"):
        #     total_features = env.unwrapped.total_features
        # else:
        #     raise AttributeError("No se pudo determinar total_features del entorno")
        self.action_values = np.zeros((*env.bins, env.action_space.n))
        print(self.action_values.shape)
        self.training_error = []
        self.learning_rate = learning_rate

    def get_action(self, obs):
        if np.random.random() < self.epsilon:
            return np.random.randint(4) # Selecciona una acción al azar
        else:
            av = self.action_values[obs]
            action = np.argmax(av)

            return action
    
    def update(
        self,
        obs: list[int],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: list[int],
        next_action: int,
    ):
        """Regla de actualización Semigradiente SARSA."""
        qsa = self.action_values[obs][action]
        next_qsa = self.action_values[next_obs][next_action]
        te = reward + self.discount_factor * next_qsa - qsa
        self.training_error.append(te)
        self.action_values[obs][action] = qsa + self.learning_rate * te

    def get_current_policy(self):
        """
        Return the current policy (e.g., action with highest Q-value for each state).
        Must be implemented by subclasses.
        """
        return # np.argmax(self.w, axis=1)  # Política greedy basada en los pesos actuales

    def test(self, num_episodes=1000):
        """Test agent performance without learning or exploration."""
        total_rewards = []
        env = self.env
        # Temporarily disable exploration for testing
        old_epsilon = self.epsilon
        self.epsilon = 0.0  # Pure exploitation

        for _ in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = self.get_action(obs)
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