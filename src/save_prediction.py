import pandas as pd
import joblib
from src.db import get_mssql_engine

def save_predictions():
    engine = get_mssql_engine()
    df = pd.read_csv("data/processed/fraud_cleaned.csv")

    model = joblib.load("fraud_model.joblib")

    X = df.drop(columns=["Class"])
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    pred_df = pd.DataFrame({
        "record_type": "fraud_transaction",
        "record_key": df.index.astype(str),
        "prediction_label": preds,
        "prediction_score": probs
    })

    pred_df.to_sql(
        "model_predictions",
        con=engine,
        if_exists="append",
        index=False
    )

    print("Predictions saved into MSSQL successfully!")

if __name__ == "__main__":
    save_predictions()