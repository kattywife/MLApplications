import seaborn as sns
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder

def prepare():
    with open("src/config.yaml") as f:
        config = yaml.safe_load(f)

    # Загрузка
    df = sns.load_dataset(config['data_load']['dataset'])
    df.to_csv(config['data_load']['raw_path'], index=False) # Сохраняем "сырые" данные

    # Очистка (как у Даны в лабе)
    df = df.dropna()
    le = LabelEncoder()
    for col in ['species', 'island', 'sex']:
        df[col] = le.fit_transform(df[col])

    # Генерация признаков (Пункт 2 задания)
    # Создадим индекс массивности: отношение длины клюва к его глубине
    df['bill_index'] = df['bill_length_mm'] / df['bill_depth_mm']

    df.to_csv(config['featurize']['processed_path'], index=False)

if __name__ == "__main__":
    prepare()