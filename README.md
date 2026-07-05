# 🕵️‍♂️ Detecção de Fraudes em Cartões de Crédito

Este projeto implementa um pipeline completo de Machine Learning para a identificação de fraudes em transações de cartão de crédito. Devido à natureza do problema (fraudes representam cerca de 0,17% dos dados históricos), o foco principal está no tratamento de **dados altamente desbalanceados** e na otimização da métrica de **Recall**, garantindo que o modelo seja capaz de capturar o maior número possível de transações ilícitas.

---

## 🚀 Funcionalidades e Pipeline

O projeto foi construído passando por todas as etapas essenciais de um ciclo profissional de Ciência de Dados:

*   **Coleta e Preparações Iniciais:** Consumo de um dataset real hospedado pelo TensorFlow.
*   **Feature Engineering:** Aplicação de transformação logarítmica (`np.log1p`) para normalizar valores extremos de transações e `StandardScaler` para padronizar as escalas numéricas.
*   **Estratégia de Validação Segura:** Separação de Treino e Teste com estratificação (`stratify`) para manter a proporção exata de fraudes nos conjuntos e evitar vieses.
*   **Tratamento de Desbalanceamento:** Utilização da técnica de oversampling **SMOTE** (Synthetic Minority Over-sampling Technique) para ensinar o modelo a identificar a classe minoritária sem causar *Data Leakage*.
*   **Modelagem Progressiva:** 
    *   **Regressão Logística** (Baseline inicial).
    *   **Random Forest** (Uso de pesos balanceados nas classes).
    *   **XGBoost** (Modelo avançado baseado em *Gradient Boosting* focado em alta performance).
*   **Otimização de Hiperparâmetros:** Uso de `GridSearchCV` para testar sistematicamente e encontrar as melhores combinações de arquitetura (profundidade de árvores e número de estimadores) no XGBoost.
*   **Explicabilidade do Modelo:** Implementação da biblioteca **SHAP** para abrir a "caixa preta" do algoritmo e entender visualmente quais variáveis são mais determinantes para classificar uma transação como fraude.

---

## 🛠️ Tecnologias Utilizadas

*   **Linguagem:** Python (Recomenda-se a versão **3.12** ou similarmente estável, para evitar erros de compilação em dependências complexas de Machine Learning).
*   **Manipulação de Dados:** `pandas`, `numpy`
*   **Machine Learning:** `scikit-learn`, `xgboost`, `imbalanced-learn`
*   **Explicabilidade e Visualização:** `shap`, `matplotlib`

---

## ⚙️ Instalação e Configuração

**1. Pré-requisitos:**
Certifique-se de ter o Python (3.11 ou 3.12) instalado e corretamente configurado nas variáveis de ambiente (`PATH`) do seu sistema.

**2. Instalação das dependências:**
Abra o seu terminal e execute o comando abaixo para instalar todas as bibliotecas necessárias para o pipeline rodar:

```bash
pip install pandas numpy scikit-learn matplotlib imbalanced-learn xgboost shap
