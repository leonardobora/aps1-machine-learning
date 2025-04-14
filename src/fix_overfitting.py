"""
Script para corrigir problemas de overfitting no modelo de previsão de emissões de N2O
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
from pathlib import Path

# Configurações de visualização
plt.style.use('ggplot')
pd.set_option('display.max_columns', 50)
warnings.filterwarnings('ignore')

# 1. Carregamento de dados
print("\n=== CARREGANDO E FILTRANDO DADOS ===")
data_path = "../data/br_seeg_emissoes_brasil.csv"

try:
    data = pd.read_csv(data_path, encoding='utf-8')
    print(f"Dados carregados: {data.shape[0]} linhas x {data.shape[1]} colunas")
except FileNotFoundError:
    # Tente um caminho alternativo se o primeiro falhar
    data_path = "data/br_seeg_emissoes_brasil.csv"
    data = pd.read_csv(data_path, encoding='utf-8')
    print(f"Dados carregados de caminho alternativo: {data.shape[0]} linhas x {data.shape[1]} colunas")

# Filtrar dados de N2O
n2o_data = data[data['gas'].str.contains('N2O', case=False, na=False)]
print(f"Registros de N2O filtrados: {n2o_data.shape[0]} linhas")

# 2. Tratamento de valores ausentes
print("\n=== TRATAMENTO DE VALORES AUSENTES ===")
n2o_data_processed = n2o_data.copy()

# Preencher valores ausentes em colunas categóricas
cat_columns = n2o_data_processed.select_dtypes(include=['object']).columns
for col in cat_columns:
    if n2o_data_processed[col].isnull().sum() > 0:
        n2o_data_processed[col].fillna('Desconhecido', inplace=True)

# Preencher valores ausentes em 'emissao' com 0
if 'emissao' in n2o_data_processed.columns and n2o_data_processed['emissao'].isnull().sum() > 0:
    n2o_data_processed['emissao'].fillna(0, inplace=True)

print("Valores ausentes tratados")

# 3. Criação de features (SEM usar o valor alvo como feature)
print("\n=== ENGENHARIA DE FEATURES (CORRIGIDA) ===")
n2o_data_enhanced = n2o_data_processed.copy()

# Features temporais
n2o_data_enhanced['decada'] = (n2o_data_enhanced['ano'] // 10) * 10
n2o_data_enhanced['ano_normalizado'] = (n2o_data_enhanced['ano'] - n2o_data_enhanced['ano'].min()) / (n2o_data_enhanced['ano'].max() - n2o_data_enhanced['ano'].min())

# Features categóricas binárias importantes
n2o_data_enhanced['is_agropecuaria'] = (n2o_data_enhanced['nivel_1'] == 'Agropecuária').astype(int)
n2o_data_enhanced['is_energia'] = (n2o_data_enhanced['nivel_1'] == 'Energia').astype(int)
n2o_data_enhanced['is_mudanca_uso_terra'] = (n2o_data_enhanced['nivel_1'] == 'Mudança de Uso da Terra e Floresta').astype(int)

# Combinações hierárquicas
n2o_data_enhanced['nivel_1_2'] = n2o_data_enhanced['nivel_1'] + '_' + n2o_data_enhanced['nivel_2']
n2o_data_enhanced['nivel_2_3'] = n2o_data_enhanced['nivel_2'] + '_' + n2o_data_enhanced['nivel_3']

# Tipo de emissão como feature numérica
if 'tipo_emissao' in n2o_data_enhanced.columns:
    emission_type_map = {'Emissão': 1, 'Remoção': -1}
    n2o_data_enhanced['emission_factor'] = n2o_data_enhanced['tipo_emissao'].map(emission_type_map).fillna(0)

# IMPORTANTE: NÃO usar 'log_emissao' ou qualquer transformação do alvo como feature!

# Interações entre variáveis (limitadas para evitar dimensionalidade excessiva)
for nivel in ['nivel_1', 'nivel_2']:
    dummies = pd.get_dummies(n2o_data_enhanced[nivel], prefix=nivel)
    for col in list(dummies.columns)[:5]:  # Limitar a 5 dummies por nível
        n2o_data_enhanced[f'{col}_por_ano'] = dummies[col] * n2o_data_enhanced['ano_normalizado']

print(f"Features criadas. Dimensões dos dados: {n2o_data_enhanced.shape[0]} linhas x {n2o_data_enhanced.shape[1]} colunas")

# 4. Preparação para modelagem
print("\n=== PREPARAÇÃO PARA MODELAGEM ===")

# Selecionar features (SEM incluir qualquer derivação da variável alvo)
features = [
    'ano', 'nivel_1', 'nivel_2', 'nivel_3', 'nivel_4', 'nivel_5', 'nivel_6',
    'decada', 'ano_normalizado', 'is_agropecuaria', 'is_energia', 'is_mudanca_uso_terra',
    'nivel_1_2', 'nivel_2_3', 'emission_factor'
]

# Adicionar colunas de interação
interaction_cols = [col for col in n2o_data_enhanced.columns if '_por_ano' in col]
features.extend(interaction_cols[:10])

target = 'emissao'

# Definir períodos de treino e teste (80% treino, 20% teste)
train_years = range(1970, 2016)  # 46 anos (80%)
test_years = range(2016, 2020)   # 4 anos (20%)

print(f"Features selecionadas: {len(features)} features")
print(f"Anos de treino: {min(train_years)}-{max(train_years)}")
print(f"Anos de teste: {min(test_years)}-{max(test_years)}")

# One-hot encoding para variáveis categóricas
categorical_features = [col for col in features if col not in ['ano', 'decada', 'ano_normalizado', 'is_agropecuaria', 'is_energia', 'is_mudanca_uso_terra', 'emission_factor'] and n2o_data_enhanced[col].dtype == 'object']
print(f"Features categóricas a serem codificadas: {len(categorical_features)} features")

data_encoded = pd.get_dummies(n2o_data_enhanced[features + [target]], columns=categorical_features)

# Divisão temporal (sem vazamento de dados)
X_train = data_encoded[data_encoded['ano'].isin(train_years)].drop(columns=[target])
y_train = data_encoded[data_encoded['ano'].isin(train_years)][target]
X_test = data_encoded[data_encoded['ano'].isin(test_years)].drop(columns=[target])
y_test = data_encoded[data_encoded['ano'].isin(test_years)][target]

print(f"Conjuntos de dados:")
print(f"  X_train: {X_train.shape[0]} linhas x {X_train.shape[1]} colunas")
print(f"  y_train: {y_train.shape[0]} valores")
print(f"  X_test: {X_test.shape[0]} linhas x {X_test.shape[1]} colunas")
print(f"  y_test: {y_test.shape[0]} valores")

# Normalização de features
scaler = StandardScaler()
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

# 5. Treinamento e avaliação de modelos
print("\n=== TREINAMENTO E AVALIAÇÃO DE MODELOS ===")

# Baseline: Regressão Linear
print("\n1. Treinando Regressão Linear (baseline)...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)
print(f"Regressão Linear:")
print(f"  RMSE: {lr_rmse:.2f}")
print(f"  MAE: {lr_mae:.2f}")
print(f"  R²: {lr_r2:.2f}")

# Random Forest
print("\n2. Treinando Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, 
                              min_samples_leaf=4, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)
print(f"Random Forest:")
print(f"  RMSE: {rf_rmse:.2f}")
print(f"  MAE: {rf_mae:.2f}")
print(f"  R²: {rf_r2:.2f}")

# 3. Random Forest Otimizado
print("\n3. Otimizando Random Forest...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [8, 12],
    'min_samples_leaf': [4, 8]
}
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print(f"Melhores hiperparâmetros para Random Forest: {grid_search.best_params_}")
best_rf = grid_search.best_estimator_
rf_opt_pred = best_rf.predict(X_test_scaled)
rf_opt_rmse = np.sqrt(mean_squared_error(y_test, rf_opt_pred))
rf_opt_mae = mean_absolute_error(y_test, rf_opt_pred)
rf_opt_r2 = r2_score(y_test, rf_opt_pred)
print(f"Random Forest (otimizado):")
print(f"  RMSE: {rf_opt_rmse:.2f}")
print(f"  MAE: {rf_opt_mae:.2f}")
print(f"  R²: {rf_opt_r2:.2f}")

# 4. Gradient Boosting
print("\n4. Treinando Gradient Boosting...")
gb_model = GradientBoostingRegressor(
    n_estimators=200, 
    learning_rate=0.05,
    max_depth=6,
    min_samples_leaf=8,
    subsample=0.8,
    random_state=42
)
gb_model.fit(X_train_scaled, y_train)
gb_pred = gb_model.predict(X_test_scaled)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
gb_mae = mean_absolute_error(y_test, gb_pred)
gb_r2 = r2_score(y_test, gb_pred)
print(f"Gradient Boosting:")
print(f"  RMSE: {gb_rmse:.2f}")
print(f"  MAE: {gb_mae:.2f}")
print(f"  R²: {gb_r2:.2f}")

# Identificar o melhor modelo
models = {
    "Regressão Linear": (lr_model, lr_rmse),
    "Random Forest": (rf_model, rf_rmse),
    "Random Forest (otimizado)": (best_rf, rf_opt_rmse),
    "Gradient Boosting": (gb_model, gb_rmse)
}

best_model_name = min(models, key=lambda k: models[k][1])
best_model, best_rmse = models[best_model_name]
print(f"\nO melhor modelo baseado em RMSE é: {best_model_name} (RMSE: {best_rmse:.2f})")

# Importância das features para o melhor modelo
if hasattr(best_model, 'feature_importances_'):
    feature_importances = pd.DataFrame({
        'Feature': X_train_scaled.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("\nImportância das features (top 10):")
    print(feature_importances.head(10))

# 6. Gerar predições finais
print("\n=== GERAÇÃO DO DATASET FINAL DE PREVISÕES ===")

# Usar o melhor modelo para gerar predições
test_indices = n2o_data_enhanced[n2o_data_enhanced['ano'].isin(test_years)].index
best_predictions = best_model.predict(X_test_scaled)

# Criar dataframe final
final_data = n2o_data.loc[test_indices].copy()
final_data['previsao'] = best_predictions

# Criar diretório de saída se não existir
output_dir = "../outputs"
if not Path(output_dir).exists():
    Path(output_dir).mkdir(parents=True)
    print(f"Diretório criado: {output_dir}")

# Salvar dataset final
output_path = f"{output_dir}/n2o_predictions.csv"
final_data.to_csv(output_path, index=False)
print(f"Dataset final salvo em: {output_path}")
print(f"Dimensões do dataset final: {final_data.shape[0]} linhas x {final_data.shape[1]} colunas")

# Visualizar distribuição das predições vs. valores reais
plt.figure(figsize=(10, 6))
plt.scatter(y_test, best_predictions, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--')
plt.xlabel('Valores Reais')
plt.ylabel('Valores Previstos')
plt.title(f'Distribuição de Valores Reais vs. Previstos ({best_model_name})')
plt.savefig(f"{output_dir}/predicoes_vs_reais.png")
print(f"Gráfico de dispersão salvo em: {output_dir}/predicoes_vs_reais.png")

print("\n=== PROCESSO CONCLUÍDO COM SUCESSO ===")
print(f"O modelo {best_model_name} foi selecionado como o melhor modelo.")
print(f"As previsões foram salvas em: {output_path}")