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

