import pandas as pd

from src.data_prep import numeric_cols


def prepare_new_client(client_data, columns, scaler):
    """
    Przygotowuje dane pojedynczego klienta do predykcji
    """
    df = pd.DataFrame([client_data])

    df_encoded = pd.get_dummies(df)

    df_aligned = df_encoded.reindex(columns=columns, fill_value=0)

    df_aligned[numeric_cols] = scaler.transform(df_aligned[numeric_cols])

    return df_aligned


def predict_client(model, client_data, columns, scaler):
    """
    Zwraca predykcję (0/1) oraz prawdopodobieństwo niespłacenia kredytu
    """
    X = prepare_new_client(client_data, columns, scaler)

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[:, 1][0]

    return prediction, probability
