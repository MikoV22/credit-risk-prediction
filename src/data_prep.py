import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

column_names = [
    'status_konta', 'czas_trwania_msc', 'historia_kredytowa', 'cel_kredytu',
    'kwota_kredytu', 'stan_oszczednosci', 'zatrudnienie_od',
    'raty_proc_dochodu', 'stan_cywilny_plec', 'inni_dluznicy',
    'lata_w_miejscu_zamieszkania', 'majatek', 'wiek',
    'inne_raty', 'mieszkanie', 'liczba_kredytow_w_banku',
    'zawod', 'liczba_osob_na_utrzymaniu', 'telefon', 'pracownik_zagraniczny',
    'target'
]

numeric_cols = [
    'czas_trwania_msc', 'kwota_kredytu', 'raty_proc_dochodu',
    'lata_w_miejscu_zamieszkania', 'wiek', 'liczba_kredytow_w_banku',
    'liczba_osob_na_utrzymaniu'
]

def load_german_credit(filepath):
    """Wczytuje dane German Credit i przekodowuje target na0/1"""
    df = pd.read_csv(filepath, sep=' ', header=None, names=column_names)
    df['target'] = df['target'].map[{1:0, 2:1}]
    return df

def prepare_data(df, test_size=0.2, random_state=42):
    """Koduje zmienne kategoryczne, dzieli na train/test i skaluje cechy numeryczne"""
    categorical_cols = df.select_dtypes(include='str').columns.tolist()
    if 'target' in categorical_cols:
        """Zabezpieczenie przed targetem w zmiennych kategorycznych"""
        categorical_cols.remove('target')

    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df_encoded.drop('target', axis=1)
    y = df_encoded['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train_scaled, X_test_scaled, y_train, y_test