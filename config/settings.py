"""
Configurações do projeto 
"""

# Configurações de dados
SYMBOL = 'PETR4.SA'
START_DATE = '2018-01-01'
END_DATE = '2026-01-01'

# Configurações do modelo
SEQUENCE_LENGTH = 60
INPUT_SIZE = 1
HIDDEN_SIZE = 50
NUM_LAYERS = 2
DROPOUT = 0.2

# Configurações de treinamento
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
PATIENCE = 15

# Configurações de otimização
OPTUNA_TRIALS = 20
OPTUNA_TIMEOUT = 600

# Caminhos
MODELS_DIR = 'models'
