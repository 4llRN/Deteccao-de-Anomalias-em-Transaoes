# 🕵️‍♂️ Detecção de Fraudes em Cartões de Crédito

Este projeto implementa um pipeline completo de Machine Learning para a identificação de fraudes em transações de cartão de crédito. Devido à natureza do problema (fraudes representam cerca de 0,17% dos dados históricos), o foco principal está no tratamento de **dados altamente desbalanceados** e na otimização da métrica de **Recall**, garantindo que o algoritmo seja capaz de capturar o maior número possível de transações ilícitas.

---

## 🚀 Funcionalidades e Pipeline

O projeto foi construído passando por todas as etapas essenciais de um ciclo profissional de Ciência de Dados:

* **Coleta e Preparações Iniciais:** Consumo de um dataset real hospedado pelo TensorFlow.
* **Feature Engineering:** Aplicação de transformação logarítmica (`np.log1p`) para normalizar valores extremos de transações numéricas, otimizando o aprendizado da máquina.
* **Estratégia de Validação Segura:** Separação de Treino e Teste com estratificação (`stratify`) para manter a proporção exata de fraudes nos conjuntos e evitar vieses e *Data Leakage*.
* **Modelagem de Alta Performance:** Utilização do algoritmo **LightGBM**, focado em velocidade de treinamento e eficiência de memória, utilizando pesos balanceados nativos (`class_weight="balanced"`) para penalizar o modelo quando ele erra a classe minoritária.
* **Otimização de Hiperparâmetros:** Uso de `GridSearchCV` para testar sistematicamente e encontrar as melhores combinações de arquitetura (profundidade de árvores e número de estimadores).
* **Explicabilidade do Modelo (XAI):** Implementação da biblioteca **SHAP** para abrir a "caixa preta" do algoritmo e gerar gráficos que provam visualmente (destacando variáveis ocultas como V4 e V14) o motivo de cada transação ser classificada como fraude.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python (Recomenda-se a versão **3.12** para evitar conflitos de DLLs e garantir estabilidade com pacotes matemáticos).
* **Manipulação de Dados:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`, `lightgbm`
* **Explicabilidade e Visualização:** `shap`, `matplotlib`

---

## ⚙️ Instalação e Configuração

**Instalação das dependências:**
Abra o seu terminal e execute o comando abaixo para instalar todas as bibliotecas necessárias para o projeto:

```bash
pip install pandas numpy scikit-learn matplotlib lightgbm shap
