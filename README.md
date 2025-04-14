# Modelo de Previsão de Emissões de N2O no Brasil

**Equipe:**
- Leonardo Bora
- Leticia Cardoso
- Luan Constancio
- Carlos Krueger

Este projeto implementa um modelo de aprendizado de máquina para prever emissões de óxido nitroso (N2O) no Brasil utilizando técnicas avançadas de machine learning. O objetivo é superar um baseline simplista e criar um modelo otimizado que pode auxiliar na compreensão dos fatores que influenciam as emissões de gases de efeito estufa no Brasil.

## Estrutura do Projeto

```
aps1-machine-learning/
├── data/
│   └── br_seeg_emissoes_brasil.csv      # Dataset de emissões
├── src/
│   ├── aps1_final_improved.ipynb        # Solução final melhorada (recomendada)
│   ├── aps1_fixed.ipynb                 # Versão inicial do modelo
│   └── improved_model.py                # Módulo Python com funções modulares
├── outputs/
│   ├── analysis/                        # Análises detalhadas dos resultados
│   ├── improved_model/                  # Resultados do modelo melhorado
│   └── n2o_predictions.csv              # Saída com previsões
├── requirements.txt                     # Dependências do projeto
└── README.md                            # Este arquivo
```

## Recursos Implementados

- **Transformação Logarítmica**: Aplicamos transformação logarítmica ao target para melhor capturar a ampla faixa de valores de emissão
- **Modelos Específicos por Setor**: Utilizamos XGBoost para o setor agropecuário e Gradient Boosting para os demais setores
- **Engenharia de Features Avançada**: Implementamos características temporais, interações e features específicas de domínio
- **Detecção de Outliers por Grupo**: Tratamento personalizado de outliers por setor para preservar a variação natural dos dados
- **Validação Cruzada Temporal**: Avaliação robusta considerando a natureza temporal dos dados

## Configuração do Ambiente

1. Clone este repositório
2. Crie um ambiente virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```
3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Executando o Projeto

Abra o notebook `src/aps1_final_improved.ipynb` em um ambiente Jupyter:

```bash
jupyter notebook src/aps1_final_improved.ipynb
```

O notebook está organizado em seções sequenciais:

1. Carregamento e Preparação dos Dados
2. Tratamento de Dados
3. Engenharia de Features Avançada
4. Preparação para Modelagem com Transformação Logarítmica
5. Treinamento de Modelos Específicos por Setor
6. Combinação dos Modelos Setoriais e Avaliação Final
7. Salvar Previsões Finais
8. Comparação com o Modelo Original
9. Conclusão e Resumo das Melhorias

## Resultados

O modelo melhorado apresentou ganhos significativos em todas as métricas:

| Métrica | Modelo Original | Modelo Melhorado | Melhoria |
|---------|----------------|------------------|----------|
| RMSE    | 8.125,52       | 1.799,56         | -77,85%  |
| MAE     | 1.386,42       | 286,31           | -79,35%  |
| R²      | 0,32           | 0,95             | +0,63    |

A principal melhoria foi a capacidade de prever corretamente a ampla faixa de valores de emissão, superando a limitação do modelo original que subestimava severamente as altas emissões.

## Requisitos

- Python 3.8 ou superior
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook

Para instalar todas as dependências, execute:
```bash
pip install -r requirements.txt
```