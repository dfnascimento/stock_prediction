import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class OptunaOptimizer:
    """Otimização de hiperparâmetros com Optuna"""
    
    def __init__(self, train_dataset, val_dataset, device, input_size=1):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.input_size = input_size
        
    def create_objective_function(self):
        """Cria função objetivo para Optuna"""
        def objective(trial):
            # Espaço de busca
            hidden_size = trial.suggest_int('hidden_size', 32, 256, step=32)
            num_layers = trial.suggest_int('num_layers', 1, 4)
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            
            # Criar modelo
            from src.models.lstm_model import LSTMModel
            model = LSTMModel(
                input_size=self.input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            ).to(self.device)
            
            # Criar DataLoaders
            train_loader = DataLoader(
                self.train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(
                self.val_dataset, batch_size=batch_size, shuffle=False
            )
            
            # Treinar por algumas épocas
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            model.train()
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(20):  # Épocas reduzidas para otimização
                # Treino
                model.train()
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    optimizer.zero_grad()
                    predictions = model(batch_X)
                    loss = criterion(predictions, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                # Validação
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        predictions = model(batch_X)
                        val_loss += criterion(predictions, batch_y).item()
                
                val_loss /= len(val_loader)
                
                # Reportar para Optuna
                trial.report(val_loss, epoch)
                
                # Early stopping
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= 5:
                    break
            
            return best_val_loss
        
        return objective
    
    def optimize(self, n_trials=20, timeout=600):
        """Executa otimização"""
        print("Iniciando otimização de hiperparâmetros com Optuna...")
        
        study = optuna.create_study(
            direction='minimize',
            study_name='lstm_stock_prediction',
            pruner=optuna.pruners.MedianPruner()
        )
        
        objective = self.create_objective_function()
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        print("Otimização concluída!")
        print(f"Melhor trial: {study.best_trial.number}")
        print(f"Melhor loss: {study.best_value:.6f}")
        print(f"Melhores hiperparâmetros:")
        
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        return study.best_params