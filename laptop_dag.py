from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Обновленный импорт согласно логам Airflow
from airflow.providers.standard.operators.python import PythonOperator
from sklearn.preprocessing import OrdinalEncoder

# Импорт функции обучения из второго файла
from train_model import train

from airflow import DAG


def download_laptop_data():
    url = "https://raw.githubusercontent.com/campusx-official/laptop-price-predictor-regression-project/main/laptop_data.csv"
    df = pd.read_csv(url)
    df.to_csv("/tmp/laptops_raw.csv", index=False)
    print("Data downloaded. Shape:", df.shape)
    print("Columns in file:", df.columns.tolist())


def clear_laptop_data():
    # Читаем данные
    df = pd.read_csv("/tmp/laptops_raw.csv")

    # 1. Очистка строковых колонок (Ram и Weight)
    # Используем regex=False для безопасности
    if "Ram" in df.columns:
        df["Ram"] = df["Ram"].str.replace("GB", "", regex=False).astype("int32")

    if "Weight" in df.columns:
        df["Weight"] = df["Weight"].str.replace("kg", "", regex=False).astype("float32")

    # 2. Удаляем лишние колонки.
    # Добавлен параметр errors='ignore', чтобы DAG не падал, если колонки нет
    cols_to_drop = [
        "Unnamed: 0",
        "Laptop_ID",
        "ScreenResolution",
        "Cpu",
        "Gpu",
        "Memory",
    ]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # 3. Обработка категориальных признаков
    cat_columns = ["Company", "TypeName", "OpSys"]
    # Кодируем только те колонки, которые реально остались в df
    existing_cats = [c for c in cat_columns if c in df.columns]
    if existing_cats:
        encoder = OrdinalEncoder()
        df[existing_cats] = encoder.fit_transform(df[existing_cats].astype(str))

    # 4. Проверка колонки цены (в некоторых версиях она может называться Price_euros)
    if "Price_euros" in df.columns and "Price" not in df.columns:
        df = df.rename(columns={"Price_euros": "Price"})

    # Удаляем пустые значения, если они появились
    df = df.dropna()

    df.to_csv("/tmp/laptops_clear.csv", index=False)
    print("Data cleaned. Final columns:", df.columns.tolist())
    print("Remaining rows:", len(df))


# Настройки DAG
default_args = {
    "owner": "student",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="laptop_price_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 2, 1),
    schedule="@hourly",
    catchup=False,
    max_active_runs=1,
) as dag:
    task_download = PythonOperator(
        task_id="download_data", python_callable=download_laptop_data
    )

    task_clear = PythonOperator(task_id="clear_data", python_callable=clear_laptop_data)

    task_train = PythonOperator(task_id="train_model", python_callable=train)

    # Последовательность шагов
    task_download >> task_clear >> task_train
