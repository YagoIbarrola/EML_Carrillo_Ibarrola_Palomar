from abc import ABC, abstractmethod
import gymnasium as gym
import numpy as np

class Agent(ABC):
    def __init__(
        self,
        env: gym.Env,
        epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
    ):
        """
        Base class for tabular RL agents.

        Args:
            env: The training environment
            learning_rate: Step size for updates (0-1)
            initial_epsilon: Initial exploration rate
            epsilon_decay: Epsilon decrease per episode
            final_epsilon: Minimum exploration rate
            discount_factor: Gamma (0-1)
        """
        self.env = env
        self.discount_factor = discount_factor

        # Exploration parameters
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []

    def get_action(self, obs):
        """
        Epsilon-greedy action selection.
        """
        pass

    @abstractmethod
    def update(
        self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs,
    ):
        """
        Learning rule (algorithm-specific).
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_current_policy(self):
        """
        Return the current policy (e.g., action with highest Q-value for each state).
        Must be implemented by subclasses.
        """
        pass
    
    def decay_epsilon(self):
        """
        Reduce exploration rate after each episode.
        """
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


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
                action, _ = self.get_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated

            total_rewards.append(episode_reward)

        # Restore original epsilon
        self.epsilon = old_epsilon

        win_rate = np.mean(np.array(total_rewards) > 0)
        average_reward = np.mean(total_rewards)
        standard_deviation = np.std(total_rewards)

        print(f"Test Results over {num_episodes} episodes:")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Average Reward: {average_reward:.3f}")
        print(f"Standard Deviation: {standard_deviation:.3f}")
        return win_rate, average_reward, standard_deviation