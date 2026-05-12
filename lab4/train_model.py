import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from mlflow.models import infer_signature

def train():
    df = pd.read_csv("penguins_raw.csv")
    
    # Очистка и кодирование (как в твоем ноутбуке)
    df = df.dropna()
    le = LabelEncoder()
    for col in ['species', 'island', 'sex']:
        df[col] = le.fit_transform(df[col])

    X = df.drop('body_mass_g', axis=1)
    y = df['body_mass_g']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("Penguin_Research")
    
    with mlflow.start_run(run_name="Jenkins_Optimal_Model"):
        # Используем параметры твоей лучшей модели (n=20, depth=3)
        model = RandomForestRegressor(n_estimators=20, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        mlflow.log_param("n_estimators", 20)
        mlflow.log_param("max_depth", 3)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
        signature = infer_signature(X_train, model.predict(X_train))
        # Логируем модель
        model_info = mlflow.sklearn.log_model(model, "model", signature=signature)
        
        # Печатаем путь к модели для Jenkins
        print(model_info.model_uri)

if __name__ == "__main__":
    train()