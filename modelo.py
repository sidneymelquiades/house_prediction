import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

# Carregamento dos dados
file_path = './results.csv'
data = pd.read_csv(file_path)
Q1 = data['valor'].quantile(0.25)
Q3 = data['valor'].quantile(0.75)
IQR = Q3 - Q1

# Definindo limites
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Removendo outliers
data = data[(data['valor'] >= limite_inferior) & (data['valor'] <= limite_superior)]


# Pré-processamento dos dados
def preprocess_data(data):
    data = data.drop(columns=['Unnamed: 0'])
    label_encoder = LabelEncoder()
    data['proximometro'] = label_encoder.fit_transform(data['proximometro'])
    scaler = StandardScaler()
    numerical_features = data.columns.drop('valor')
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    return data


# Análise exploratória
def exploratory_analysis(data):
    plt.figure(figsize=(12, 8))
    sns.pairplot(data)
    plt.show()
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.show()


# Divisão dos dados
def split_data(data):
    X = data.drop(columns=['valor'])
    y = data['valor']
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Avaliação dos modelos
def evaluate_models(X_train, y_train):
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor(),
        'Support Vector Regressor': SVR()
    }
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        results[name] = np.mean(scores)
    return sorted(results.items(), key=lambda x: x[1], reverse=True)


# Treinamento do melhor modelo
def train_best_model(X_train, y_train, best_model_name):
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor(),
        'Support Vector Regressor': SVR()
    }
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    return best_model


# Avaliação do modelo no conjunto de teste
def evaluate_on_test(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mae, rmse, r2, y_pred


# Pipeline
def main_pipeline(file_path):
    data = pd.read_csv(file_path)
    data = preprocess_data(data)
    exploratory_analysis(data)
    X_train, X_test, y_train, y_test = split_data(data)
    best_models = evaluate_models(X_train, y_train)
    print(best_models)
    best_model_name = best_models[0][0]
    print(best_model_name)
    best_model = train_best_model(X_train, y_train, best_model_name)
    mae, rmse, r2, y_pred = evaluate_on_test(best_model, X_test, y_test)

    # Salvando os conjuntos de treino e teste
    X_train.to_csv('./X_train.csv', index=False)
    X_test.to_csv('./X_test.csv', index=False)
    y_train.to_csv('./y_train.csv', index=False)
    y_test.to_csv('./y_test.csv', index=False)

    # Salvando o melhor modelo
    with open('./best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    # Imprimindo as métricas
    print(f'MAE: {mae}')
    print(f'RMSE: {rmse}')
    print(f'R2 Score: {r2}')



# Executar o pipeline
main_pipeline('./results.csv')
