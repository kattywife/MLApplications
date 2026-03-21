import pandas as pd
import numpy as np
import mlflow
import joblib
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature


def scale_frame(frame):
    df = frame.copy()
    # Целевая переменная — Price (цена ноутбука)
    # Удаляем колонку Unnamed: 0, если она проскочила при сохранении csv
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    X, y = df.drop(columns=["Price"]), df["Price"]

    scaler = StandardScaler()
    power_trans = PowerTransformer()

    X_scale = scaler.fit_transform(X.values)
    Y_scale = power_trans.fit_transform(y.values.reshape(-1, 1))
    return X_scale, Y_scale, power_trans


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train():
    # Загружаем очищенные данные
    df = pd.read_csv("/tmp/laptops_clear.csv")
    X, Y, power_trans = scale_frame(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    params = {
        "alpha": [0.0001, 0.001, 0.01],
        "penalty": ["l2", "elasticnet"],
        "loss": ["squared_error", "huber"],
        "max_iter": [1000, 2000],
    }

    mlflow.set_experiment("laptop_price_prediction")

    with mlflow.start_run():
        lr = SGDRegressor(random_state=42)
        grid = GridSearchCV(lr, params, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train.reshape(-1))

        best_model = grid.best_estimator_

        # Предсказания
        y_pred = best_model.predict(X_val)

        # Обратное преобразование цены из лог-шкалы в обычную
        y_val_real = power_trans.inverse_transform(y_val.reshape(-1, 1))
        y_pred_real = power_trans.inverse_transform(y_pred.reshape(-1, 1))

        rmse, mae, r2 = eval_metrics(y_val_real, y_pred_real)

        # Логируем в MLflow
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Сохраняем модель в MLflow
        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(best_model, "laptop_model", signature=signature)

        # Сохраняем артефакт локально
        joblib.dump(best_model, "/tmp/laptop_model.pkl")

    print(f"Model trained. R2: {r2:.3f}")
    return True
