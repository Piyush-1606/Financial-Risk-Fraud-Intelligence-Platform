import pandas as pd

def load_fraud_data():
    df = pd.read_csv("data/raw/creditcard.csv")
    return df

def basic_cleaning(df):
    df = df.copy()
    df = df.drop_duplicates()
    df = df.dropna()
    return df

def add_features(df):
    df = df.copy()
    df["log_amount"] = (df["Amount"] + 1).apply(lambda x: __import__("math").log(x))
    df["time_hours"] = df["Time"] / 3600.0
    return df

if __name__ == "__main__":
    df = load_fraud_data()
    df = basic_cleaning(df)
    df = add_features(df)

    df.to_csv("data/processed/fraud_cleaned.csv", index=False)
    print("Processed fraud data saved successfully!")