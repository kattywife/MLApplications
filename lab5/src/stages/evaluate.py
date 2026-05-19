import pandas as pd
import yaml
import joblib
import json
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def evaluate():
    with open("src/config.yaml") as f:
        config = yaml.safe_load(f)

    model = joblib.load(config['train']['model_path'])
    test_df = pd.read_csv(config['data_split']['testset_path'])

    X_test = test_df.drop('body_mass_g', axis=1)
    y_test = test_df['body_mass_g']

    preds = model.predict(X_test)
    
    metrics = {
        "r2": r2_score(y_test, preds),
        "rmse": np.sqrt(mean_squared_error(y_test, preds))
    }

    with open(config['test']['metrics_path'], "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    evaluate()