# Importación de módulos o clases
from .agent import Agent
from .taxiAgentQLearning import TaxiAgentQLearning
from .taxiAgentDoubleQLearning import TaxiAgentDoubleQLearning
from .taxiAgentMontecarloOnPolicy import TaxiAgentMontecarloOnPolicy
from .taxiAgentMontecarloOffPolicy import TaxiAgentMontecarloOffPolicy
from .taxiAgentSARSA import TaxiAgentSARSA
from .taxiAgentExpectedSARSA import TaxiAgentExpectedSARSA
from .taxiAgentMontecarloOnPolicyInvDecay import TaxiAgentMontecarloOnPolicyInvDecay
from .lunarAgentSARSASemi import LunarAgentSARSA
from .lunarLanderTileCoding import TileCodingEnv
from .lunarAgentSARSADeep import SemiGradientSarsaDeepAgent, QNetworkSARSA
from .lunarAgentDeepQLearning import LunarAgentDeepQLearning, DqnReplayBuffer, QNetwork

# Lista de módulos o clases públicas
__all__ = [
    'Agent',

    'TaxiAgentQLearning',
    'TaxiAgentDoubleQLearning',

    'TaxiAgentMontecarloOnPolicy',
    'TaxiAgentMontecarloOnPolicyInvDecay',
    'TaxiAgentMontecarloOffPolicy',

    'TaxiAgentSARSA',
    'TaxiAgentExpectedSARSA',


    'LunarAgentSARSA',
    'TileCodingEnv',
    'SemiGradientSarsaDeepAgent',

    'QNetworkSARSA',
    'LunarAgentDeepQLearning',

    'QNetwork',
    'DqnReplayBuffer'
    ]


