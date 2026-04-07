from datetime import datetime
import json
import logging
import os

import pandas as pd
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from hooks import CarsHook  # ← импортируем из plugins/hooks/


# --- Task 1: Функция для загрузки данных ---
def _fetch_cars(conn_id: str, templates_dict: dict, batch_size: int = 100, **_):
    logger = logging.getLogger(__name__)
    output_path = templates_dict["output_path"]

    logger.info("Fetching all cars from the API using CarsHook...")
    hook = CarsHook(conn_id=conn_id)
    cars = list(hook.get_cars(batch_size=batch_size))
    logger.info(f"Fetched {len(cars)} car records")

    # Убедимся, что директория существует
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(cars, f)

    logger.info(f"Saved raw cars data to {output_path}")


# --- Task 2: Функция для очистки данных ---
def _clean_cars_data(templates_dict: dict, **_):
    logger = logging.getLogger(__name__)
    input_path = templates_dict["input_path"]
    output_path = templates_dict["output_path"]

    logger.info(f"Reading raw data from {input_path}")
    df = pd.read_json(input_path)

    if df.empty:
        logger.warning("No data to clean. Exiting.")
        return

    logger.info(f"Initial shape: {df.shape}")

    # 1. Удаление дубликатов
    df.drop_duplicates(inplace=True)
    logger.info(f"Shape after dropping duplicates: {df.shape}")

    # 2. Удаление пропусков
    df.dropna(inplace=True)
    logger.info(f"Shape after dropping NaNs: {df.shape}")
    
    # 3. Преобразование типов (если необходимо)
    # В нашем API признаки Fuel_type и Transmission уже числовые,
    # но для полноты примера убедимся, что они имеют целочисленный тип.
    logger.info(f"Unique Fuel_type values before encoding: {df['Fuel_type'].unique()}")
    df['Fuel_type_encoded'] = pd.factorize(df['Fuel_type'])[0]
    
    # Кодируем 'Transmission'
    logger.info(f"Unique Transmission values before encoding: {df['Transmission'].unique()}")
    df['Transmission_encoded'] = pd.factorize(df['Transmission'])[0]
    
    # Удаляем старые столбцы, если они больше не нужны
    df.drop(['Fuel_type', 'Transmission'], axis=1, inplace=True)
    
    logger.info("Categorical features encoded successfully.")
    
    # Убедимся, что год - это число
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
    df.dropna(subset=['Year'], inplace=True) # Удаляем строки, где год не удалось сконвертировать

    logger.info("Data cleaned successfully.")

    # Создаем директорию для сохранения
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Сохраняем очищенный датасет
    df.to_json(output_path, orient="records", indent=4)
    logger.info(f"Cleaned data saved to {output_path}")
    logger.info(f"Final dataset columns: {df.columns.tolist()}")



with DAG(
    dag_id="02_hook_and_clean",
    description="Fetches and cleans car data from the API.",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
) as dag:

    fetch_task = PythonOperator(
        task_id="fetch_cars_data",
        python_callable=_fetch_cars,
        op_kwargs={"conn_id": "carsapi", "batch_size": 100},
        templates_dict={
            "output_path": "/data/custom_hook/cars.json",
        },
    )

    clean_task = PythonOperator(
        task_id="clean_cars_data",
        python_callable=_clean_cars_data,
        templates_dict={
            "input_path": "/data/custom_hook/cars.json",
            "output_path": "/data/cleaned/cars_cleaned.json",
        },
    )

    # Задаем последовательность выполнения
    fetch_task >> clean_task
    