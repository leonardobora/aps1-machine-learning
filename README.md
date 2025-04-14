# Modelo de Previsão de Emissões de N2O no Brasil

**Equipe:**
- Leonardo Bora
- Leticia Cardoso
- Luan Constancio
- Carlos Krueger

Este projeto implementa um modelo de aprendizado de máquina para prever emissões de óxido nitroso (N2O) no Brasil utilizando técnicas clássicas de machine learning. O objetivo é superar um baseline simplista e criar um modelo otimizado que pode auxiliar na compreensão dos fatores que influenciam as emissões.

## Estrutura do Projeto

```
aps1/
├── data/
│   └── br_seeg_emissoes_brasil.csv      # Dataset de emissões
├── src/
│   └── aps1_fixed.ipynb                 # Notebook com solução otimizada
├── outputs/
│   └── n2o_predictions.csv              # Saída com previsões
├── requirements.txt                     # Dependências do projeto
└── README.md                            # Este arquivo
```

## Configuração do Ambiente

1. Clone este repositório
2. Crie um ambiente virtual (recomendado)
3. Instale as dependências:

```bash
pip install -r requirements.txt
```

## Executando o Projeto

Abra o notebook `src/aps1_fixed.ipynb` em um ambiente Jupyter:

```bash
jupyter notebook src/aps1_fixed.ipynb
```

O notebook está organizado em seções sequenciais:

1. Compreensão e Preparação dos Dados
2. Análise Exploratória (EDA)
3. Tratamento de Dados e Engenharia de Features
4. Preparação para Modelagem
5. Modelagem e Avaliação
6. Cross-Validation e Validação Final
7. Conclusão

## Metodologia

O projeto segue estas etapas principais:

- **Análise Exploratória:** Compreensão das características dos dados, identificação de padrões e outliers.
- **Tratamento de Dados:** Limpeza, imputação de valores ausentes e tratamento de outliers.
- **Engenharia de Features:** Criação de novas variáveis para melhorar o poder preditivo.
- **Modelagem:** Implementação de diversos algoritmos (Regressão Linear, Árvores de Decisão, Random Forest, Gradient Boosting).
- **Avaliação:** Comparação dos modelos usando métricas como RMSE, MAE e R².
- **Validação:** Validação cruzada para garantir robustez e generalização.

## Resultados

O modelo Random Forest otimizado apresentou os melhores resultados, superando significativamente o baseline de Regressão Linear. As previsões finais são salvas em `outputs/n2o_predictions.csv`.

## Requisitos

- Python 3.8 ou superior
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- Jupyter Notebook

Para instalar todas as dependências, execute:
```bash
pip install -r requirements.txt
```