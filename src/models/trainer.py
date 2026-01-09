import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from datetime import datetime

class ModelTrainer:
    """Treinamento de modelos"""
    
    def __init__(self, model, train_loader, val_loader, device, 
                 learning_rate, model_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_dir = model_dir
        
        # Criar diretório se não existir
        os.makedirs(model_dir, exist_ok=True)
        
        # Loss e otimizador
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Histórico
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
    def train_epoch(self):
        """Treina uma época"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_X, batch_y in self.train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch_X)
            loss = self.criterion(predictions, batch_y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def validate(self):
        """Valida o modelo"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y in self.val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def train(self, epochs=100, patience=10, verbose=True):
        """Treina o modelo"""
        patience_counter = 0
        
        for epoch in range(epochs):
            # Treinar e validar
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            # Atualizar scheduler
            self.scheduler.step(val_loss)
            
            # Salvar histórico
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Verificar se é o melhor modelo
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Progresso
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Época {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"LR: {self.history['learning_rate'][-1]:.6f}")
            
            # Early stopping
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping na época {epoch+1}")
                break
        
        if verbose:
            print(f"Treinamento concluído!")
            print(f"Melhor val loss: {self.best_val_loss:.6f} (época {self.best_epoch+1})")
        
        return self.history