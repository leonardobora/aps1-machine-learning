import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
from pathlib import Path
import json

# Configurações de visualização
plt.style.use('ggplot')
pd.set_option('display.max_columns', 50)
warnings.filterwarnings('ignore')

# Criar diretório para salvar figuras e resultados
output_dir = "../outputs/improved_model"
figures_dir = "../outputs/improved_model/figures"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

def load_data(file_path):
    """
    Carrega os dados do arquivo CSV e retorna um DataFrame.
    """
    print(f"Carregando dados de {file_path}...")
    try:
        return pd.read_csv(file_path, encoding='utf-8')
    except FileNotFoundError:
        alt_path = "data/br_seeg_emissoes_brasil.csv"
        print(f"Arquivo não encontrado. Tentando caminho alternativo: {alt_path}")
        return pd.read_csv(alt_path, encoding='utf-8')

def filter_gas_data(data, gas_name):
    """
    Filtra os dados para um gás específico.
    """
    print(f"\n=== FILTRANDO DADOS DE {gas_name} ===")
    filtered_data = data[data['gas'].str.contains(gas_name, case=False, na=False)]
    print(f"Registros de {gas_name}: {len(filtered_data)}")
    return filtered_data

def handle_missing_values(data):
    """
    Trata valores ausentes nos dados.
    """
    print("\n=== TRATAMENTO DE VALORES AUSENTES ===")
    processed_data = data.copy()
    
    # Preencher valores ausentes em colunas categóricas com "Desconhecido"
    cat_columns = processed_data.select_dtypes(include=['object']).columns
    for col in cat_columns:
        if processed_data[col].isnull().sum() > 0:
            processed_data[col].fillna('Desconhecido', inplace=True)
    
    # Preencher valores ausentes em 'emissao' com 0
    if 'emissao' in processed_data.columns and processed_data['emissao'].isnull().sum() > 0:
        processed_data['emissao'].fillna(0, inplace=True)
    
    return processed_data

def detect_outliers_by_group(df, group_cols, target_col, threshold=3):
    """
    Detecta outliers por grupo (por exemplo, por setor).
    """
    result = df.copy()
    result['is_outlier'] = False
    
    for name, group in df.groupby(group_cols):
        q1 = group[target_col].quantile(0.25)
        q3 = group[target_col].quantile(0.75)
        iqr = q3 - q1
        
        # Usar limites mais permissivos para grupos com naturalmente mais variação
        if name == 'Agropecuária':
            # Usar limites mais amplos para agricultura
            lower = q1 - threshold * 2 * iqr  # Dobro do limite normal
            upper = q3 + threshold * 2 * iqr
        else:
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
        
        outlier_idx = group[(group[target_col] < lower) | (group[target_col] > upper)].index
        result.loc[outlier_idx, 'is_outlier'] = True
    
    return result

def create_features(data):
    """
    Cria features avançadas para o modelo.
    """
    print("\n=== ENGENHARIA DE FEATURES AVANÇADA ===")
    enhanced_data = data.copy()
    
    # Features temporais
    enhanced_data['decada'] = (enhanced_data['ano'] // 10) * 10
    enhanced_data['ano_normalizado'] = (enhanced_data['ano'] - enhanced_data['ano'].min()) / (enhanced_data['ano'].max() - enhanced_data['ano'].min())
    
    # Features específicas de setor
    enhanced_data['is_agropecuaria'] = (enhanced_data['nivel_1'] == 'Agropecuária').astype(int)
    enhanced_data['is_energia'] = (enhanced_data['nivel_1'] == 'Energia').astype(int)
    enhanced_data['is_mudanca_uso_terra'] = (enhanced_data['nivel_1'] == 'Mudança de Uso da Terra e Floresta').astype(int)
    
    # Combinações de níveis hierárquicos
    enhanced_data['nivel_1_2'] = enhanced_data['nivel_1'] + '_' + enhanced_data['nivel_2']
    enhanced_data['nivel_2_3'] = enhanced_data['nivel_2'] + '_' + enhanced_data['nivel_3']
    
    # Features de interação avançadas
    # Interação ano x setor mais granular
    for nivel in ['nivel_1', 'nivel_2']:
        dummies = pd.get_dummies(enhanced_data[nivel], prefix=nivel)
        for col in dummies.columns:
            enhanced_data[f'{col}_por_ano'] = dummies[col] * enhanced_data['ano_normalizado']
            # Adicionar termos polinomiais para capturar tendências não-lineares
            enhanced_data[f'{col}_por_ano_quad'] = enhanced_data[f'{col}_por_ano'] ** 2
    
    # Features específicas para agricultura (maior emissor de N2O)
    if 'nivel_6' in enhanced_data.columns:
        # Identificar animais específicos (maior fonte de emissões)
        is_animal = enhanced_data['nivel_5'] == 'Animal'
        is_aves = enhanced_data['nivel_6'] == 'Aves'
        is_gado = enhanced_data['nivel_6'].str.contains('Gado', na=False)
        
        enhanced_data['is_animal'] = is_animal.astype(int)
        enhanced_data['is_aves'] = is_aves.astype(int)
        enhanced_data['is_gado'] = is_gado.astype(int)
        
        # Interação animal-ano
        enhanced_data['animal_por_ano'] = is_animal.astype(int) * enhanced_data['ano']
        enhanced_data['aves_por_ano'] = is_aves.astype(int) * enhanced_data['ano']
        enhanced_data['gado_por_ano'] = is_gado.astype(int) * enhanced_data['ano']
    
    print(f"Total de features após engenharia: {enhanced_data.shape[1]}")
    return enhanced_data

def prepare_for_modeling(data, target_col, train_years, test_years, log_transform=True):
    """
    Prepara os dados para modelagem com opção de transformação logarítmica.
    """
    print("\n=== PREPARAÇÃO PARA MODELAGEM (COM TRANSFORMAÇÃO LOG) ===")
    
    # Separar features e target
    features = data.drop(columns=[target_col])
    target = data[target_col]
    
    # One-hot encoding para variáveis categóricas
    cat_cols = features.select_dtypes(include=['object']).columns
    features_encoded = pd.get_dummies(features, columns=cat_cols)
    
    # Divisão temporal
    mask_train = features['ano'].isin(train_years)
    mask_test = features['ano'].isin(test_years)
    
    X_train = features_encoded[mask_train]
    X_test = features_encoded[mask_test]
    y_train = target[mask_train]
    y_test = target[mask_test]
    
    # Transformação logarítmica do target (se solicitado)
    if log_transform:
        print("Aplicando transformação logarítmica ao target")
        # Garantir que não há valores negativos antes do log
        min_value = min(y_train.min(), y_test.min())
        if min_value < 0:
            offset = abs(min_value) + 1
            y_train_transformed = np.log1p(y_train + offset)
            y_test_transformed = np.log1p(y_test + offset)
            # Guardar o offset para a transformação inversa
            transform_params = {'offset': offset, 'type': 'log_with_offset'}
        else:
            y_train_transformed = np.log1p(y_train)
            y_test_transformed = np.log1p(y_test)
            transform_params = {'offset': 0, 'type': 'log'}
    else:
        y_train_transformed = y_train.copy()
        y_test_transformed = y_test.copy()
        transform_params = {'type': 'none'}
    
    return X_train, X_test, y_train, y_test, y_train_transformed, y_test_transformed, transform_params

def inverse_transform_predictions(predictions, transform_params):
    """
    Aplica a transformação inversa nas previsões.
    """
    if transform_params['type'] == 'log':
        return np.expm1(predictions)
    elif transform_params['type'] == 'log_with_offset':
        return np.expm1(predictions) - transform_params['offset']
    return predictions

def train_sector_models(data, target_col, train_years, test_years, log_transform=True):
    """
    Treina modelos específicos por setor.
    """
    print("\n=== TREINAMENTO DE MODELOS ESPECÍFICOS POR SETOR ===")
    
    # Dividir por setor
    sectors = data['nivel_1'].unique()
    models = {}
    predictions = {}
    metrics = {}
    
    for sector in sectors:
        print(f"\nTreinando modelo para o setor: {sector}")
        sector_data = data[data['nivel_1'] == sector].copy()
        
        # Preparar dados para este setor
        X_train, X_test, y_train, y_test, y_train_log, y_test_log, transform_params = prepare_for_modeling(
            sector_data, target_col, train_years, test_years, log_transform=log_transform
        )
        
        if len(X_train) < 10 or len(X_test) < 5:
            print(f"  Dados insuficientes para o setor {sector}. Pulando.")
            continue
        
        # Treinar modelo
        if sector == 'Agropecuária':
            # Modelo mais robusto para agricultura
            model = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=42)
        else:
            # Modelo padrão para outros setores
            model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)
        
        # Treinar no target transformado
        model.fit(X_train, y_train_log)
        
        # Prever e inverter a transformação
        y_pred_log = model.predict(X_test)
        y_pred = inverse_transform_predictions(y_pred_log, transform_params)
        
        # Calcular métricas
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  R²: {r2:.2f}")
        
        # Armazenar modelo, previsões e métricas
        models[sector] = model
        predictions[sector] = pd.DataFrame({
            'true': y_test,
            'pred': y_pred,
            'index': y_test.index
        })
        metrics[sector] = {'rmse': rmse, 'mae': mae, 'r2': r2}
    
    return models, predictions, metrics

def combine_predictions(data, sector_predictions, test_years):
    """
    Combina as previsões dos modelos específicos por setor.
    """
    print("\n=== COMBINANDO PREVISÕES DOS MODELOS SETORIAIS ===")
    
    # Preparar dataframe final de previsões
    test_data = data[data['ano'].isin(test_years)].copy()
    test_data['prediction'] = np.nan
    
    # Inserir previsões para cada setor
    for sector, preds in sector_predictions.items():
        for idx, row in preds.iterrows():
            test_data.loc[row['index'], 'prediction'] = row['pred']
    
    # Verificar se há índices sem previsão
    missing_pred = test_data['prediction'].isnull().sum()
    if missing_pred > 0:
        print(f"Atenção: {missing_pred} registros ficaram sem previsão.")
    
    # Calcular métricas globais
    mask_with_pred = ~test_data['prediction'].isnull()
    rmse = np.sqrt(mean_squared_error(test_data.loc[mask_with_pred, 'emissao'], 
                                     test_data.loc[mask_with_pred, 'prediction']))
    mae = mean_absolute_error(test_data.loc[mask_with_pred, 'emissao'], 
                            test_data.loc[mask_with_pred, 'prediction'])
    r2 = r2_score(test_data.loc[mask_with_pred, 'emissao'], 
                 test_data.loc[mask_with_pred, 'prediction'])
    
    print(f"Métricas globais do modelo combinado:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R²: {r2:.2f}")
    
    return test_data, {'rmse': rmse, 'mae': mae, 'r2': r2}

def plot_sector_predictions(sector_predictions, save_dir):
    """
    Plota as previsões por setor para análise visual.
    """
    for sector, preds_df in sector_predictions.items():
        plt.figure(figsize=(12, 6))
        plt.scatter(preds_df['true'], preds_df['pred'], alpha=0.6)
        plt.plot([0, preds_df['true'].max()], [0, preds_df['true'].max()], 'r--')
        plt.title(f'Valores Reais vs. Previstos - Setor: {sector}')
        plt.xlabel('Valores Reais')
        plt.ylabel('Valores Previstos')
        plt.grid(True, alpha=0.3)
        
        # Salvar figura
        filename = f"{save_dir}/predictions_{sector.replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figura salva em: {filename}")
        plt.close()

def save_final_predictions(final_data, output_file):
    """
    Salva as previsões finais no formato solicitado.
    """
    print(f"\n=== SALVANDO PREVISÕES FINAIS EM {output_file} ===")
    
    # Renomear a coluna de previsão para o formato esperado
    result_df = final_data.copy()
    if 'prediction' in result_df.columns:
        result_df.rename(columns={'prediction': 'previsao'}, inplace=True)
    
    # Verificar diretório de saída
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Diretório criado: {output_dir}")
    
    # Salvar CSV
    result_df.to_csv(output_file, index=False)
    print(f"Arquivo salvo em: {output_file}")
    print(f"Tamanho do arquivo: {os.path.getsize(output_file)} bytes")
    print(f"Número de registros: {len(result_df)}")
    
    return result_df

def compare_with_original(original_pred_file, new_pred_file):
    """
    Compara as novas previsões com as originais.
    """
    print("\n=== COMPARAÇÃO COM MODELO ORIGINAL ===")
    
    # Carregar previsões
    try:
        original = pd.read_csv(original_pred_file)
        new = pd.read_csv(new_pred_file)
        
        # Verificar se têm a mesma estrutura
        if 'previsao' in original.columns and 'previsao' in new.columns:
            # Calcular diferenças
            original_rmse = np.sqrt(mean_squared_error(original['emissao'], original['previsao']))
            new_rmse = np.sqrt(mean_squared_error(new['emissao'], new['previsao']))
            
            improvement = (original_rmse - new_rmse) / original_rmse * 100
            
            print(f"RMSE modelo original: {original_rmse:.2f}")
            print(f"RMSE novo modelo: {new_rmse:.2f}")
            print(f"Melhoria: {improvement:.2f}%")
            
            # Visualização comparativa para os primeiros registros
            plt.figure(figsize=(12, 6))
            
            # Amostrar alguns registros para visualização
            sample = original.head(20).copy()
            sample['previsao_nova'] = new.loc[new.index.isin(sample.index), 'previsao'].values
            
            x = np.arange(len(sample))
            width = 0.25
            
            plt.bar(x - width, sample['emissao'], width, label='Valor Real')
            plt.bar(x, sample['previsao'], width, label='Previsão Original')
            plt.bar(x + width, sample['previsao_nova'], width, label='Previsão Nova')
            
            plt.title('Comparação: Real vs. Original vs. Novo Modelo')
            plt.xticks(x, sample.index, rotation=90)
            plt.xlabel('Índice')
            plt.ylabel('Emissão (t)')
            plt.legend()
            plt.tight_layout()
            
            # Salvar figura
            comparison_file = os.path.join(os.path.dirname(new_pred_file), "comparison.png")
            plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {'original_rmse': original_rmse, 'new_rmse': new_rmse, 'improvement': improvement}
        
    except Exception as e:
        print(f"Erro ao comparar modelos: {e}")
    
    return None

def main():
    # Carregar dados
    data_path = "../data/br_seeg_emissoes_brasil.csv"
    data = load_data(data_path)
    
    # Filtrar para N2O
    n2o_data = filter_gas_data(data, 'N2O')
    
    # Tratar valores ausentes
    n2o_data = handle_missing_values(n2o_data)
    
    # Detectar outliers por setor
    n2o_data_with_outliers = detect_outliers_by_group(n2o_data, ['nivel_1'], 'emissao', threshold=3)
    
    # Engenharia de features avançada
    n2o_enhanced = create_features(n2o_data_with_outliers)
    
    # Definir anos de treino e teste
    train_years = range(1970, 2016)
    test_years = range(2016, 2020)
    
    # Treinar modelos específicos por setor
    sector_models, sector_predictions, sector_metrics = train_sector_models(
        n2o_enhanced, 'emissao', train_years, test_years, log_transform=True
    )
    
    # Combinar previsões
    final_data, combined_metrics = combine_predictions(n2o_enhanced, sector_predictions, test_years)
    
    # Plotar previsões por setor
    plot_sector_predictions(sector_predictions, figures_dir)
    
    # Salvar resultados finais
    output_file = f"{output_dir}/n2o_predictions_improved.csv"
    final_predictions = save_final_predictions(final_data, output_file)
    
    # Comparar com o modelo original
    original_pred_file = "../outputs/n2o_predictions.csv"
    comparison = compare_with_original(original_pred_file, output_file)
    
    # Salvar relatório de métricas
    metrics_report = {
        'sector_metrics': sector_metrics,
        'combined_metrics': combined_metrics,
        'comparison': comparison
    }
    
    # Salvar métricas em um arquivo JSON
    metrics_file = f"{output_dir}/metrics_report.json"
    # Converter objetos não serializáveis (como índices) para strings
    serializable_metrics = {}
    for k, v in metrics_report.items():
        if isinstance(v, dict):
            serializable_metrics[k] = {}
            for sk, sv in v.items():
                if isinstance(sv, dict):
                    serializable_metrics[k][str(sk)] = {}
                    for ssk, ssv in sv.items():
                        serializable_metrics[k][str(sk)][str(ssk)] = float(ssv) if isinstance(ssv, (int, float, np.number)) else str(ssv)
                else:
                    serializable_metrics[k][str(sk)] = float(sv) if isinstance(sv, (int, float, np.number)) else str(sv)
        else:
            serializable_metrics[k] = str(v)
    
    with open(metrics_file, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    print("\n=== PROCESSAMENTO CONCLUÍDO ===")
    print(f"Resultados salvos em {output_dir}")
    print(f"Relatório de métricas salvo em {metrics_file}")

if __name__ == "__main__":
    main()