# Importación de módulos o clases
from .agent import Agent
from .taxiAgentQLearning import TaxiAgentQLearning
from .taxiAgentDoubleQLearning import TaxiAgentDoubleQLearning
from .taxiAgentMontecarloOnPolicy import TaxiAgentMontecarloOnPolicy
from .taxiAgentMontecarloOffPolicy import TaxiAgentMontecarloOffPolicy
from .taxiAgentSARSA import TaxiAgentSARSA
from .taxiAgentExpectedSARSA import TaxiAgentExpectedSARSA

# Lista de módulos o clases públicas
__all__ = [
    'Agent',
    'TaxiAgentQLearning',
    'TaxiAgentDoubleQLearning',
    'TaxiAgentMontecarloOnPolicy',
    'TaxiAgentMontecarloOffPolicy',
    'TaxiAgentSARSA',
    'TaxiAgentExpectedSARSA'
    ]


