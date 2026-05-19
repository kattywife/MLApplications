import pandas as pd
import yaml
import joblib
from sklearn.ensemble import RandomForestRegressor

def train():
    with open("src/config.yaml") as f:
        config = yaml.safe_load(f)

    train_df = pd.read_csv(config['data_split']['trainset_path'])
    X = train_df.drop('body_mass_g', axis=1)
    y = train_df['body_mass_g']

    model = RandomForestRegressor(
        n_estimators=config['train']['n_estimators'],
        max_depth=config['train']['max_depth'],
        random_state=42
    )
    model.fit(X, y)
    
    joblib.dump(model, config['train']['model_path'])

if __name__ == "__main__":
    train()