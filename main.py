import warnings
import torch
import sys
import os

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

warnings.filterwarnings('ignore')

# Importar módulos
from config.settings import *
from src.data.collector import DataCollector
from src.data.preprocessor import DataPreprocessor
from src.data.splitter import DataSplitter
from src.models.lstm_model import LSTMModel
from src.models.trainer import ModelTrainer
from src.utils.metrics import MetricsCalculator
from src.utils.logger import Logger
from src.utils.model_io import save_model_and_scaler
from src.optimization.optuna_optimization import OptunaOptimizer
from torch.utils.data import TensorDataset

def main():
    # Inicializar logger
    logger = Logger()
    logger.section("INÍCIO DO PIPELINE DE PREVISÃO")
    
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo: {device}")
    
    # 1. COLETA DE DADOS
    logger.section("1. COLETA DE DADOS")
    
    collector = DataCollector(SYMBOL, START_DATE, END_DATE)
    df = collector.download_data(progress=False)
    df_close = collector.get_close_prices()
    
    data_info = collector.get_data_info()
    logger.info(f"Símbolo: {data_info['symbol']}")
    logger.info(f"Período: {data_info['first_date']} a {data_info['last_date']}")
    logger.info(f"Total de dias: {data_info['total_days']}")
    logger.info(f"Preço inicial: R$ {data_info['first_price']:.2f}")
    logger.info(f"Preço final: R$ {data_info['last_price']:.2f}")
    logger.info(f"Variação: {data_info['price_change_pct']:.2f}%")
    
    # 2. PRÉ-PROCESSAMENTO
    logger.section("2. PRÉ-PROCESSAMENTO")
    
    preprocessor = DataPreprocessor()
    close_prices = df_close['Close'].values.reshape(-1, 1)
    
    X, y, scaler, preprocess_stats = preprocessor.prepare_data(
        close_prices, SEQUENCE_LENGTH
    )
    
    logger.info(f"Sequências criadas: {preprocess_stats['num_sequences']}")
    logger.info(f"Comprimento da sequência: {SEQUENCE_LENGTH} dias")
    logger.info(f"Formato de X: {X.shape}")
    logger.info(f"Formato de y: {y.shape}")
    logger.info(f"Dados normalizados - Média: {preprocess_stats['scaled_mean']:.4f}, "
                f"Desvio padrão: {preprocess_stats['scaled_std']:.4f}")
    
    # 3. DIVISÃO DOS DADOS
    logger.section("3. DIVISÃO DOS DADOS")
    
    splits, split_stats = DataSplitter.split_data(
        X, y, TRAIN_RATIO, VAL_RATIO
    )
    
    for split_name, stats in split_stats.items():
        logger.info(f"{split_name.upper():10s}: {stats['samples']:4d} amostras "
                   f"({stats['percentage']:.1f}%)")
    
    dataloaders = DataSplitter.create_dataloaders(splits, BATCH_SIZE)
    
    # 4. TREINAMENTO DO MODELO
    logger.section("4. TREINAMENTO DO MODELO")
    
    model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    
    model_info = model.get_model_info()
    logger.info(f"Arquitetura: {model_info['architecture']}")
    logger.info(f"Tamanho da camada oculta: {model_info['hidden_size']}")
    logger.info(f"Número de camadas: {model_info['num_layers']}")
    logger.info(f"Parâmetros totais: {model_info['total_params']:,}")
    logger.info(f"Parâmetros treináveis: {model_info['trainable_params']:,}")
    
    trainer = ModelTrainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        device=device,
        learning_rate=LEARNING_RATE,
        model_dir=MODELS_DIR
    )
    
    logger.info(f"Iniciando treinamento...")
    logger.info(f"Épocas: {EPOCHS}, Paciência: {PATIENCE}, "
               f"Batch Size: {BATCH_SIZE}, Learning Rate: {LEARNING_RATE}")
    
    history = trainer.train(epochs=EPOCHS, patience=PATIENCE)
    
    logger.info(f"Treinamento concluído!")
    logger.info(f"Melhor val loss: {trainer.best_val_loss:.6f} "
               f"(época {trainer.best_epoch + 1})")
    
    # 5. OTIMIZAÇÃO (OPCIONAL)
    logger.section("5. OTIMIZAÇÃO DE HIPERPARÂMETROS")
    
    try:
        # Criar datasets para otimização
        
        
        X_train_tensor = torch.FloatTensor(splits['train'][0])
        y_train_tensor = torch.FloatTensor(splits['train'][1])
        X_val_tensor = torch.FloatTensor(splits['val'][0])
        y_val_tensor = torch.FloatTensor(splits['val'][1])
        
        train_dataset_opt = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset_opt = TensorDataset(X_val_tensor, y_val_tensor)
        
        optimizer = OptunaOptimizer(train_dataset_opt, val_dataset_opt, device, INPUT_SIZE)
        best_params = optimizer.optimize(n_trials=OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT)
        
        logger.info(f"Melhores hiperparâmetros encontrados:")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value}")
            
            
            
        logger.section("6. RETREINO COM MELHORES HIPERPARÂMETROS")
        
        # Extrair melhores parâmetros
        best_hidden_size = int(best_params.get('hidden_size', HIDDEN_SIZE))
        best_num_layers = int(best_params.get('num_layers', NUM_LAYERS))
        best_dropout = best_params.get('dropout', DROPOUT)
        best_learning_rate = best_params.get('learning_rate', LEARNING_RATE)
        
        # Criar novo modelo com melhores parâmetros
        logger.info(f"Criando novo modelo com melhores parâmetros:")
        logger.info(f"  Hidden Size: {best_hidden_size}")
        logger.info(f"  Num Layers: {best_num_layers}")
        logger.info(f"  Dropout: {best_dropout}")
        
        best_model = LSTMModel(
            INPUT_SIZE, 
            best_hidden_size, 
            best_num_layers, 
            best_dropout
        ).to(device)
        
        # Criar novo trainer com melhores parâmetros
        best_trainer = ModelTrainer(
            model=best_model,
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            device=device,
            learning_rate=best_learning_rate,
            model_dir=MODELS_DIR
        )
        
        # Treinar com melhores parâmetros
        logger.info(f"Iniciando retreino com learning rate: {best_learning_rate}")
        logger.info(f"Épocas: {EPOCHS}, Paciência: {PATIENCE}")
        
        best_history = best_trainer.train(epochs=EPOCHS, patience=PATIENCE)
        
        logger.info(f"Retreino concluído!")
        logger.info(f"Melhor val loss: {best_trainer.best_val_loss:.6f} "
                   f"(época {best_trainer.best_epoch + 1})")
        
        # Atualizar modelo para o melhor modelo treinado
        model = best_model
        trainer = best_trainer
        
    except Exception as e:
        logger.warning(f"Otimização não realizada: {e}")
        best_params = None
    
    # 6. AVALIAÇÃO
    logger.section("6. AVALIAÇÃO DO MODELO")
    
    # Fazer previsões
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in dataloaders['test']:
            batch_X = batch_X.to(device)
            batch_pred = model(batch_X)
            predictions.extend(batch_pred.cpu().numpy())
            targets.extend(batch_y.numpy())
    
    # Desnormalizar
    predictions = scaler.inverse_transform(predictions)
    targets = scaler.inverse_transform(targets)
    
    # Calcular métricas
    metrics = MetricsCalculator.calculate_all_metrics(predictions, targets)
    
    # Gerar relatório
    metrics_report = MetricsCalculator.generate_metrics_report(metrics)
    print("\n" + metrics_report)
    
    # 7. SALVAMENTO SIMPLIFICADO
    logger.section("7. SALVAMENTO DO MODELO")
    model_file, scaler_file = save_model_and_scaler(model, scaler, folder='models')
    
    logger.info(f"Modelo salvo: {model_file}")
    logger.info(f"Scaler salvo: {scaler_file}")
    
    # 7. RESUMO FINAL
    
    logger.info(f"Pipeline executado com sucesso!")
    logger.info(f"Símbolo: {SYMBOL}")
    logger.info(f"Período: {data_info['first_date']} a {data_info['last_date']}")
    logger.info(f"Modelo: LSTM ({model_info['total_params']:,} parâmetros)")
    logger.info(f"MAE: R$ {metrics.get('MAE', 0):.2f}")
    logger.info(f"MAPE: {metrics.get('MAPE', 0):.2f}%")
    logger.info(f"R²: {metrics.get('R2', 0):.4f}")
    


if __name__ == "__main__":
    try:
        main()
        print("\nPipeline executado com sucesso!")
    except Exception as e:
        print(f"\nErro: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)