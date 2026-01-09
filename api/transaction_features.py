import pandas as pd

class TransactionFeatures:
    def __init__(self, data):
        self.Time = data.get('Time')
        self.Amount = data.get('Amount')
        self.V1 = data.get('V1')
        self.V2 = data.get('V2')
        self.V3 = data.get('V3')
        self.V4 = data.get('V4')
        self.V5 = data.get('V5')
        self.V6 = data.get('V6')
        self.V7 = data.get('V7')
        self.V8 = data.get('V8')
        self.V9 = data.get('V9')
        self.V10 = data.get('V10')
        self.V11 = data.get('V11')
        self.V12 = data.get('V12')
        self.V13 = data.get('V13')
        self.V14 = data.get('V14')
        self.V15 = data.get('V15')
        self.V16 = data.get('V16')
        self.V17 = data.get('V17')
        self.V18 = data.get('V18')
        self.V19 = data.get('V19')
        self.V20 = data.get('V20')
        self.V21 = data.get('V21')
        self.V22 = data.get('V22')
        self.V23 = data.get('V23')
        self.V24 = data.get('V24')
        self.V25 = data.get('V25')
        self.V26 = data.get('V26')
        self.V27 = data.get('V27')
        self.V28 = data.get('V28')

    def to_dataframe(self):
        """Converter para DataFrame"""
        transaction_dict = {
            'Time': self.Time,
            'Amount': self.Amount,
            'V1': self.V1, 'V2': self.V2, 'V3': self.V3, 'V4': self.V4, 'V5': self.V5,
            'V6': self.V6, 'V7': self.V7, 'V8': self.V8, 'V9': self.V9, 'V10': self.V10,
            'V11': self.V11, 'V12': self.V12, 'V13': self.V13, 'V14': self.V14, 'V15': self.V15,
            'V16': self.V16, 'V17': self.V17, 'V18': self.V18, 'V19': self.V19, 'V20': self.V20,
            'V21': self.V21, 'V22': self.V22, 'V23': self.V23, 'V24': self.V24, 'V25': self.V25,
            'V26': self.V26, 'V27': self.V27, 'V28': self.V28
        }
        return pd.DataFrame([transaction_dict])
    
    def pre_processing(self, df):
        #Aplicar mesma engenharia de features pré processamento
        df['Hour'] = df['Time'] % 24
        
        df_new = df[['V11', 'V4', 'V18', 'V7', 'V3', 'V16', 'V10', 'V12', 'V14', 'V17', 'Amount', 'Hour']]
        
        print(df_new)
        
        return df_new