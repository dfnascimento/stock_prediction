import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """Modelo LSTM para previsão de séries temporais"""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Camada LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Regularização
        self.dropout = nn.Dropout(dropout)
        
        # Camada de saída
        self.fc = nn.Linear(hidden_size, 1)
        
        # Inicializar pesos
        self._init_weights()
    
    def _init_weights(self):
        """Inicializa os pesos do modelo"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)
    
    def forward(self, x):
        """Forward pass do modelo"""
        # Inicializar estados ocultos
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Última saída da sequência
        lstm_out = lstm_out[:, -1, :]
        
        # Dropout
        lstm_out = self.dropout(lstm_out)
        
        # Saída
        output = self.fc(lstm_out)
        
        return output
    
    def get_model_info(self):
        """Retorna informações sobre o modelo"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'architecture': 'LSTM'
        }