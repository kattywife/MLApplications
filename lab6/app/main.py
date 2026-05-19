from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from sqlalchemy.orm import Session
from pathlib import Path
from . import database, model as db_model

# Динамический путь к модели
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "penguin_model.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Penguin Weight Predictor")

class PenguinInput(BaseModel):
    species: str
    island: str
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    sex: str

database.Base.metadata.create_all(bind=database.engine)

@app.post("/predict")
def predict(data: PenguinInput, db: Session = Depends(database.get_db)):
    try:
        df = pd.DataFrame([data.model_dump()])
        
        # 1. Маппинг категорий (как в ПР №3/5)
        mapping = {
            "species": {"Adelie": 0, "Chinstrap": 1, "Gentoo": 2},
            "island": {"Biscoe": 0, "Dream": 1, "Torgersen": 2},
            "sex": {"Female": 0, "Male": 1}
        }
        for col, m in mapping.items():
            df[col] = df[col].map(m)

        # 2. Генерация признака (ВАЖНО: как в ПР №5)
        df['bill_index'] = df['bill_length_mm'] / df['bill_depth_mm']

        # 3. Порядок колонок СТРОГО как при обучении
        expected_columns = [
            "species", "island", "bill_length_mm", "bill_depth_mm", 
            "flipper_length_mm", "sex", "bill_index"
        ]
        df = df[expected_columns]

        # 4. Предсказание массы
        prediction = model.predict(df)[0]

        # 5. Сохранение в БД
        db_record = db_model.PenguinPrediction(
            species=data.species,
            bill_length=data.bill_length_mm,
            prediction=float(prediction)
        )
        db.add(db_record)
        db.commit()

        return {"predicted_body_mass_g": round(float(prediction), 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}