import os
import torch
import joblib
import numpy as np
from flask import Blueprint, jsonify, request
from flasgger.utils import swag_from
from api.transaction_features import TransactionFeatures
from datetime import datetime
from src.utils.model_io import load_model_and_scaler
from config import settings
from src.models.lstm_model import LSTMModel


predict = Blueprint('predict', __name__ )

# Configurar dispositivo
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

try:
    model, scaler = load_model_and_scaler(settings.MODELS_DIR, 
                                          device, 
                                          LSTMModel)
    
    print(f"Modelo carregado")
except Exception as e:
    print(settings.MODELS_DIR)
    print(f"Erro ao carregar modelo: {e}")
    model = None
  


@predict.route('/', methods=['GET'])
def get_predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Modelo não carregado'}), 500
    
    try:
        # Receber dados
        data = request.json
        prices = data.get('prices', [])
        days = data.get('days', 1)

        model.eval()
        
        # Preparar entrada
        recent_prices = prices[-60:]  # último SEQUENCE_LENGTH dias
        input_array = np.array(recent_prices).reshape(-1, 1)
        
        # Normalizar
        scaled = scaler.transform(input_array)
        input_tensor = torch.FloatTensor(scaled.reshape(1, 60, 1))
        
        # Fazer previsão
        predictions = []
        current_input = input_tensor.clone()  
        
        with torch.no_grad():
            for _ in range(days):
                pred_scaled = model(current_input)
                
                pred_value = pred_scaled[0, 0].item()  
                

                new_sequence = torch.cat([
                    current_input[:, 1:, :],  
                    pred_scaled.unsqueeze(1)   
                ], dim=1)
                
                current_input = new_sequence
        
        # Desnormalizar
        preds_array = np.array(predictions).reshape(-1, 1)
        final_preds = scaler.inverse_transform(preds_array).flatten()
        
        return jsonify({
            'predictions': final_preds.tolist(),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500