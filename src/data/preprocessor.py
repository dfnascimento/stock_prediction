import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    """Pré-processamento de dados sem visualização"""
    
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def normalize(self, data):
        """Normaliza dados para escala [0, 1]"""
        return self.scaler.fit_transform(data)
    
    def inverse_normalize(self, data):
        """Desnormaliza dados para escala original"""
        return self.scaler.inverse_transform(data)
    
    @staticmethod
    def create_sequences(data, sequence_length):
        """Cria sequências para modelos LSTM"""
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            seq = data[i:(i + sequence_length)]
            target = data[i + sequence_length]
            X.append(seq)
            y.append(target)
            
        return np.array(X), np.array(y)
    
    def prepare_data(self, close_prices, sequence_length):
        """Pipeline completo de preparação de dados"""
        # Normalizar
        scaled_data = self.normalize(close_prices)
        
        # Criar sequências
        X, y = self.create_sequences(scaled_data, sequence_length)
        
        # Estatísticas
        stats = {
            'original_shape': close_prices.shape,
            'scaled_mean': float(scaled_data.mean()),
            'scaled_std': float(scaled_data.std()),
            'num_sequences': len(X),
            'sequence_length': sequence_length
        }
        
        return X, y, self.scaler, stats