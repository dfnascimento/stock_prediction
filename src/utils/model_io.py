import os
import torch
import pickle
from datetime import datetime
import glob
from config import settings
import json
from src.models.lstm_model import LSTMModel

def save_model_and_scaler(model, scaler, folder='models'):
    """Salva modelo e scaler"""
    
    # Criar pasta
    os.makedirs(folder, exist_ok=True)
    
    # Timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Salvar modelo
    model_file = f'{folder}/model_{ts}.pth'
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': model.lstm.input_size,
            'hidden_size': model.hidden_size,
            'num_layers': model.num_layers,
            'output_size': model.fc.out_features,
            'dropout': model.dropout
        }
    }, model_file)
    
    # Salvar scaler
    scaler_file = f'{folder}/scaler_{ts}.pkl'
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
        
    
    return model_file, scaler_file

def load_model_and_scaler(folder='models', device='cpu', model_class=LSTMModel):
    """Carrega o último modelo e scaler"""
    
    # Procurar modelos
    model_files = glob.glob(f'{folder}/model_*.pth')
    if not model_files:
        raise FileNotFoundError(f"Nenhum modelo em {folder}")
    
    # Pegar o mais recente
    latest_model = max(model_files)
    
    
    
    # Procurar scaler correspondente (mesmo timestamp)
    base_name = os.path.basename(latest_model).replace('model_', '').replace('.pth', '')
    scaler_file = f'{folder}/scaler_{base_name}.pkl'
    
    if not os.path.exists(scaler_file):
        # Se não encontrar correspondente, pega o scaler mais recente
        scaler_files = glob.glob(f'{folder}/scaler_*.pkl')
        if scaler_files:
            scaler_file = max(scaler_files)
        else:
            raise FileNotFoundError(f"Nenhum scaler em {folder}")
    
    # Carregar
    checkpoint = torch.load(latest_model, map_location=device, weights_only=False)
    
    
      # Verificar se temos configuração completa
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        
        
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config.get('dropout_rate')  # Carregar dropout rate


    
    # Imprimir de forma simples
    print("=" * 50)
    print("CONFIGURAÇÕES DO MODELO CARREGADO")
    print("=" * 50)
    print(f"Input Size:    {input_size}")
    print(f"Hidden Size:   {hidden_size}")
    print(f"Num Layers:    {num_layers}")
    print(f"Dropout:       {dropout}")
    print("=" * 50)
    
    print(input_size)
    if model_class is None:
        raise ValueError("Forneça a classe do modelo (model_class)")
    
    model = model_class(input_size, hidden_size, num_layers, dropout).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
    

    return model, scaler