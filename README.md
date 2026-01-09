# 📈 Sistema de Previsão de Preços de Ações com LSTM

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![Licença](https://img.shields.io/badge/Licen%C3%A7a-MIT-green)](LICENSE)
[![Vercel](https://img.shields.io/badge/Deploy-Vercel-black)](https://vercel.com)

Um sistema de aprendizado profundo para previsão de preços de ações usando redes neurais LSTM com otimização automatizada de hiperparâmetros e capacidades de previsão em tempo real.

## 🌐 Demonstração Online

- **API**: `https://stock-predictor.vercel.app/api`

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

| Métrica | Descrição | Fórmula | Ideal |
|---------|-----------|---------|-------|
| **MAE** | Erro Absoluto Médio | $\frac{1}{n}\sum|y-\hat{y}|$ | Quanto menor |
| **RMSE** | Raiz do Erro Quadrático Médio | $\sqrt{\frac{1}{n}\sum(y-\hat{y})^2}$ | Quanto menor |
| **MAPE** | Erro Percentual Absoluto Médio | $\frac{100\%}{n}\sum|\frac{y-\hat{y}}{y}|$ | < 5% |
| **R²** | Coeficiente de Determinação | $1 - \frac{\sum(y-\hat{y})^2}{\sum(y-\bar{y})^2}$ | ≈ 1 |

## 🌐 Deploy no Vercel


**API Endpoints Disponíveis**
```http
GET    /api/health          # Status do serviço
POST   /api/predict         # Previsão de ações
GET    /api/symbols         # Símbolos disponíveis
GET    /api/history/{symbol}# Histórico de preços
```

### Exemplo de Uso da API

```python
import requests
import json

# Previsão para AAPL nos próximos 7 dias
payload = {
    "symbol": "AAPL",
    "days": 7
}

response = requests.post(
    "https://seu-projeto.vercel.app/api/predict",
    json=payload
)

result = response.json()
print(f"Preço atual: ${result['last_price']:.2f}")
print(f"Previsões: {result['predictions']}")
```



## 🙏 Agradecimentos

- [PyTorch](https://pytorch.org/) - Framework de deep learning
- [Optuna](https://optuna.org/) - Otimização de hiperparâmetros
- [Yahoo Finance](https://finance.yahoo.com/) - Dados de mercado
- [Vercel](https://vercel.com/) - Plataforma de deploy



**Desenvolvido com ❤️ por Diego de Faria do Nascimento**  
*Última atualização: Janeiro 2026*

> **⚠️ Disclaimer**: Este projeto é para fins educacionais e de pesquisa. Não é uma recomendação de investimento. O mercado de ações é volátil e previsões passadas não garantem resultados futuros.