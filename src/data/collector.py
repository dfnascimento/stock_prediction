import yfinance as yf
import pandas as pd
import numpy as np

class DataCollector:
    """Coleta dados históricos de ações sem gráficos"""
    
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        
    def download_data(self, progress=False):
        """Baixa dados históricos do Yahoo Finance"""
        try:
            self.data = yf.download(
                self.symbol,
                start=self.start_date,
                end=self.end_date,
                progress=progress
            )
            
            if self.data.empty:
                raise ValueError("Nenhum dado encontrado para o símbolo fornecido")
                
            return self.data
            
        except Exception as e:
            raise Exception(f"Erro ao baixar dados: {e}")
    
    def get_close_prices(self):
        """Retorna apenas os preços de fechamento"""
        if self.data is None:
            raise ValueError("Dados não baixados. Execute download_data() primeiro")
            
        return self.data[['Close']].copy()
    
    def get_data_info(self):
        """Retorna informações sobre os dados coletados"""
        if self.data is None:
            raise ValueError("Dados não baixados")
            
        df_close = self.get_close_prices()
        close_prices = df_close['Close'].values.reshape(-1, 1)
        
        return {
            'symbol': self.symbol,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'total_days': len(df_close),
            'first_date': df_close.index[0].date(),
            'last_date': df_close.index[-1].date(),
            'first_price': float(close_prices[0][0]),
            'last_price': float(close_prices[-1][0]),
            'price_change_pct': float((close_prices[-1][0] / close_prices[0][0] - 1) * 100)
        }