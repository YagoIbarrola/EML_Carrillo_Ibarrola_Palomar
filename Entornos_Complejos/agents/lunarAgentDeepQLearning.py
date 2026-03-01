import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from agents.agent import Agent
import gymnasium as gym
import random
from collections import deque

class QNetwork(nn.Module):
    """
    Red neuronal para aproximar la función Q.

    Parámetros:
      - state_dim (int): Dimensión del estado.
      - action_dim (int): Número de acciones posibles.
      - hidden_dim (int): Número de neuronas en las capas ocultas.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=64, seed = 42):
        super(QNetwork, self).__init__()
        # Primera capa: de estado a capa oculta de tamaño hidden_dim.
        self.seed = torch.manual_seed(seed)
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
        x = self.fc3(x)
        return x
    

class DqnReplayBuffer:
    def __init__(self, max_capacity):
        # Configuramos el dispositivo de hardware disponible
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Utilizamos deque para manejar el límite de capacidad automáticamente
        self.buffer = deque(maxlen=max_capacity)

    def push(self, transition):
        # Añadimos la nueva transición; si se supera max_capacity, deque borra la más antigua
        self.buffer.append(transition)

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, k = batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).float().to(self.device)

        actions = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        
        


        return states, next_states, actions, rewards, dones
    
    def __len__(self):
        return len(self.buffer)
    
    

class LunarAgentDeepQLearning(Agent):
    def __init__(self, env, state_size, action_size, epsilon, epsilon_decay, final_epsilon, discount_factor, seed):
        super().__init__(
            env=env,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
            decay_type="linear",  
        )
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Hiperparametros de DQN
        self.buffer_size = int(1e5)  # Capacidad de la memoria
        self.batch_size = 100         # Tamaño del lote de entrenamiento
        self.tau = 1e-3              # Parametro para la actualizacion suave de la red objetivo
        self.lr = 5e-4               # Tasa de aprendizaje
        self.update_every = 4        # Frecuencia de entrenamiento en pasos

        # Instanciamos la Red Local y la Red Objetivo
        self.qnetwork_local = QNetwork(state_size, action_size).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(self.device)
        

        # Definimos el optimizador para la red local
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Inicializamos la memoria de repeticion
        self.memory = DqnReplayBuffer(self.buffer_size)
        
        # Inicializamos el contador de pasos temporales
        self.t_step = 0

    def update(self, state, action, reward, done, next_state):
        # Guardamos la experiencia en la memoria
        self.memory.push((state, next_state, action, reward, done))
        
        # Incrementamos el contador de pasos
        self.t_step = (self.t_step + 1) % self.update_every
        
        # Aprendemos solo si ha pasado el intervalo y tenemos suficientes datos
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

    def get_action(self, state, eps=0.0):
        # Convertimos el estado de numpy a tensor
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # Ponemos la red en modo evaluacion temporalmente
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()

        # Seleccion de accion epsilon-greedy
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, next_states, actions, rewards, dones = experiences

        # Obtenemos los valores Q maximos proyectados por la red objetivo para los estados siguientes
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Calculamos los objetivos Q reales (Target TD)
        Q_targets = rewards + (self.discount_factor * Q_targets_next * (1 - dones))

        # Obtenemos las predicciones de la red local para las acciones tomadas
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Calculamos la perdida utilizando el Error Cuadratico Medio (MSE)
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimizamos la perdida actualizando los pesos de la red local
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Actualizamos la red objetivo mezclando suavemente sus pesos
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def soft_update(self, local_model, target_model):
        # Interpolacion lineal: θ_target = τ*θ_local + (1 - τ)*θ_target
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def decay_epsilon(self):
        # Decaimiento lineal de epsilon
        self.epsilon = max(self.final_epsilon, self.epsilon*self.epsilon_decay)

    def test(self, env, n_episodes=5):
        total_wins = 0
        # Creamos una lista para almacenar las puntuaciones de todas las partidas
        all_scores = [] 
        
        for episode in range(1, n_episodes + 1):
            # Eliminamos la semilla para que cada partida de prueba sea un escenario nuevo
            state, _ = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Sin exploracion durante la prueba (politica puramente codiciosa)
                action = self.get_action(state, eps=0.0)  
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                state = next_state
                
            # Guardamos la puntuacion final del episodio completo
            all_scores.append(episode_reward)
            
            # Evaluamos la victoria sobre la recompensa ACUMULADA, fuera del bucle while
            if episode_reward >= 200.0: 
                total_wins += 1
                
            print(f"Episodio {episode}: Recompensa = {episode_reward:.2f}")
            
        # Calculamos la media real sumando todas las partidas
        media_real = sum(all_scores) / n_episodes
        print(f"\nResultados finales: {total_wins}/{n_episodes} victorias.")
        print(f"Recompensa media real: {media_real:.2f}")
        
    def get_current_policy(self):
        pass
            
            