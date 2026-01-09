import numpy as np
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    mean_absolute_percentage_error,
    r2_score
)

class MetricsCalculator:
    """Calcula métricas de avaliação sem gráficos"""
    
    @staticmethod
    def calculate_all_metrics(predictions, targets, prefix=''):
        """Calcula todas as métricas de avaliação"""
        # Converter para arrays numpy
        preds = np.array(predictions).flatten()
        tgts = np.array(targets).flatten()
        
        # Métricas
        mae = mean_absolute_error(tgts, preds)
        mse = mean_squared_error(tgts, preds)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(tgts, preds) * 100
        r2 = r2_score(tgts, preds)
        
        # Erros absolutos
        abs_errors = np.abs(preds - tgts)
        
        # Métricas adicionais
        max_error = np.max(abs_errors)
        mean_error = np.mean(preds - tgts)
        std_error = np.std(preds - tgts)
        
        # Percentis
        percentiles = [5, 25, 50, 75, 95]
        percentile_values = np.percentile(abs_errors, percentiles)
        percentiles_dict = {f'p{p}': val for p, val in zip(percentiles, percentile_values)}
        
        metrics = {
            f'{prefix}MAE': mae,
            f'{prefix}MSE': mse,
            f'{prefix}RMSE': rmse,
            f'{prefix}MAPE': mape,
            f'{prefix}R2': r2,
            f'{prefix}Max_Error': max_error,
            f'{prefix}Mean_Error': mean_error,
            f'{prefix}Std_Error': std_error,
            **percentiles_dict
        }
        
        return metrics
    

    
    @staticmethod
    def generate_metrics_report(metrics):
        """Gera um relatório textual das métricas"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("RELATÓRIO DE MÉTRICAS")
        report_lines.append("=" * 60)
        
        # Agrupar métricas por tipo
        error_metrics = {k: v for k, v in metrics.items() 
                        if any(x in k for x in ['MAE', 'MSE', 'RMSE', 'MAPE', 'Error'])}
        
        accuracy_metrics = {k: v for k, v in metrics.items() 
                           if any(x in k for x in ['R2', 'Accuracy'])}
        
        # Erros
        report_lines.append("\nERRO:")
        report_lines.append("-" * 40)
        for name, value in error_metrics.items():
            if 'MAPE' in name:
                report_lines.append(f"{name:25s}: {value:8.2f}%")
            elif any(x in name for x in ['MAE', 'MSE', 'RMSE', 'Error']):
                report_lines.append(f"{name:25s}: ${value:8.2f}")
        
        # Acurácia
        report_lines.append("\nACURÁCIA:")
        report_lines.append("-" * 40)
        for name, value in accuracy_metrics.items():
            if 'Accuracy' in name:
                report_lines.append(f"{name:25s}: {value:8.2f}%")
            else:
                report_lines.append(f"{name:25s}: {value:8.4f}")
        
        # Percentis
        report_lines.append("\nPERCENTIS DO ERRO ABSOLUTO:")
        report_lines.append("-" * 40)
        for name, value in metrics.items():
            if name.startswith('p'):
                report_lines.append(f"{name:25s}: ${value:8.2f}")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)