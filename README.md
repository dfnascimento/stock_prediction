# 📈 Sistema de Previsão de Preços de Ações com LSTM

Um sistema de aprendizado profundo para previsão de preços de ações usando redes neurais LSTM com otimização automatizada de hiperparâmetros e capacidades de previsão em tempo real.


## ✨ Funcionalidades

- **📊 Coleta de Dados em Tempo Real**: Busca automática de dados de ações do Yahoo Finance
- **🧠 Rede Neural LSTM**: Previsão de séries temporais avançada usando PyTorch
- **⚡ Otimização Automática**: Ajuste de hiperparâmetros com Optuna
- **📈 Múltiplas Métricas**: Avaliação com MAE, RMSE, MAPE, R² score
- **☁️ Pronto para Cloud**: Deploy fácil no Vercel com API REST


```

## 🚀 Começando Rápido

### Pré-requisitos

- Python 3.8+
- Git
- Conta no [GitHub](https://github.com)
- Conta no [Vercel](https://vercel.com) (para deploy)

### Instalação Local

1. **Clone o repositório**
```bash
git clone https://github.com/seu-usuario/stock-prediction.git
cd stock-prediction
```

2. **Crie um ambiente virtual**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Instale as dependências**
```bash
pip install -r requirements.txt
```

4. **Execute o pipeline completo**
```bash
python main.py
```

## 📊 Pipeline de Execução

O sistema segue um pipeline de 7 etapas:

### 1. **Coleta de Dados**
- Busca históricos de ações do Yahoo Finance
- Suporte para múltiplos símbolos (AAPL, TSLA, GOOGL, etc.)
- Intervalos personalizáveis (diário, semanal, mensal)

### 2. **Pré-processamento**
- Normalização Min-Max ou StandardScaler
- Criação de sequências temporais

### 3. **Divisão dos Dados**
- 70% treinamento
- 15% validação
- 15% teste
- Preserva ordem temporal

### 4. **Treinamento do Modelo**
- Arquitetura LSTM personalizável
- Early stopping para evitar overfitting
- Checkpoint automático do melhor modelo

### 5. **Otimização com Optuna** 

- Busca automática dos melhores hiperparâmetros

### 6. **Retreino com Melhores Parâmetros**
- Treinamento final com configuração otimizada
- Validação cruzada
- Salvamento do modelo final

### 7. **Avaliação e Deploy**
- Métricas detalhadas
- Preparação para produção



**Parâmetros Configuráveis:**
- `input_size`: Número de features (padrão: 1)
- `hidden_size`: Neurônios na camada oculta (50-256)
- `num_layers`: Camadas LSTM (1-4)
- `dropout`: Regularização (0.0-0.5)
- `sequence_length`: Janela temporal (30-90 dias)

## 📈 Métricas de Avaliação

| Métrica | Descrição |
|---------|-----------|
| **MAE** | Erro Absoluto Médio |
| **RMSE** | Raiz do Erro Quadrático Médio |
| **MAPE** | Erro Percentual Absoluto Médio | 
| **R²** | Coeficiente de Determinação | 


# 🌐 **API de Previsão - Como Usar**

## **Endpoint Principal**
```
POST (URL)/predict
```

## **Parâmetros Obrigatórios**
| Campo | Tipo | Descrição |
|-------|------|-----------|
| `prices` | array | Lista de preços históricos (mínimo 60 valores) |
| `days` | integer | Número de dias para prever (1-30) |

## **📋 Exemplos JSON**

### **Exemplo Básico**
```json
{
  "prices": [
    100.0, 101.5, 102.3, 103.8, 102.9, 104.2, 105.5, 103.8, 106.1, 107.3,
    108.0, 107.5, 109.2, 110.0, 111.3, 110.8, 112.1, 113.4, 112.9, 114.2,
    115.0, 114.5, 116.2, 117.0, 118.3, 117.8, 119.1, 120.4, 119.9, 121.2,
    122.0, 121.5, 123.2, 124.0, 125.3, 124.8, 126.1, 127.4, 126.9, 128.2,
    129.0, 128.5, 130.2, 131.0, 132.3, 131.8, 133.1, 134.4, 133.9, 135.2,
    136.0, 135.5, 137.2, 138.0, 139.3, 138.8, 140.1, 141.4, 140.9, 142.2
  ],
  "days": 3
}
```

### **Resposta de Sucesso**
```json
{
  "success": true,
  "predictions": [210.15, 211.28],
  "last_price": 209.0,
  "days": 2,
  "timestamp": "2024-01-15T10:30:45.123456"
}
```

## **⚠️ Requisitos Mínimos**
- 60 preços históricos
- Valores numéricos (não strings)
- Dias entre 1 e 30


## 🙏 Agradecimentos

- [PyTorch](https://pytorch.org/) - Framework de deep learning
- [Optuna](https://optuna.org/) - Otimização de hiperparâmetros
- [Yahoo Finance](https://finance.yahoo.com/) - Dados de mercado
- [Vercel](https://vercel.com/) - Plataforma de deploy



**Desenvolvido por Diego de Faria do Nascimento**  
*Última atualização: Janeiro 2026*

> **⚠️ Disclaimer**: Este projeto é para fins educacionais e de pesquisa. Não é uma recomendação de investimento. O mercado de ações é volátil e previsões passadas não garantem resultados futuros.