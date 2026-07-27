import streamlit as st
import joblib

import src.predict as predict

st.set_page_config(page_title="Credit Risk Prediction", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load('models/model.pk1')
    scaler = joblib.load('models/scaler.pk1')
    columns = joblib.load('models/columns.pk1')
    return model, scaler, columns


model, scaler, columns = load_artifacts()

st.title("Ocena ryzyka kredytowego")
st.write("Aplikacja przewiduje, czy klient spłaci kredyt, na podstawie modelu regresji logistycznej wytrenowanego na zbiorze German Credit Data.")

with st.expander("Jak korzystać z aplikacji"):
    st.markdown("""
    **Instrukcja**

    1. Wypełnij formularz poniżej danymi klienta.
    2. Kliknij przycisk **Oceń ryzyko** na dole formularza.
    3. Aplikacja pokaże przewidywaną decyzję oraz prawdopodobieństwo niespłacenia kredytu.

    **Uwagi**

    - Wszystkie pola są wymagane — formularz startuje z wartościami domyślnymi, które możesz zmienić.
    - Kwoty podane są w markach niemieckich (DM), zgodnie z oryginalnym zbiorem danych z lat 90.
    - Wynik ma charakter poglądowy i pochodzi z modelu uczonego na 1000 historycznych wniosków kredytowych.
      Nie jest to narzędzie do podejmowania rzeczywistych decyzji kredytowych.
    """)

st.divider()

st.subheader("Dane klienta")

col1, col2 = st.columns(2)

with col1:
    wiek = st.number_input("Wiek", min_value=18, max_value=100, value=35)
    kwota_kredytu = st.number_input("Kwota kredytu (DM)", min_value=250, max_value=20000, value=3000)
    czas_trwania_msc = st.number_input("Okres kredytowania (miesiące)", min_value=4, max_value=72, value=24)
    raty_proc_dochodu = st.slider("Rata jako % dochodu", min_value=1, max_value=4, value=2)

with col2:
    lata_w_miejscu_zamieszkania = st.slider("Lata w obecnym miejscu zamieszkania", min_value=1, max_value=4, value=2)
    liczba_kredytow_w_banku = st.slider("Liczba kredytów w tym banku", min_value=1, max_value=4, value=1)
    liczba_osob_na_utrzymaniu = st.slider("Liczba osób na utrzymaniu", min_value=1, max_value=2, value=1)


st.subheader("Sytuacja finansowa")

status_konta_map = {
    "Poniżej 0 DM": "A11",
    "0 - 200 DM": "A12",
    "Powyżej 200 DM": "A13",
    "Brak rachunku bieżącego": "A14"
}

stan_oszczednosci_map = {
    "Poniżej 100 DM": "A61",
    "100 - 500 DM": "A62",
    "500 - 1000 DM": "A63",
    "Powyżej 1000 DM": "A64",
    "Brak / nieznane": "A65"
}

historia_kredytowa_map = {
     "Brak kredytów / wszystkie spłacone terminowo": "A30",
    "Wszystkie kredyty w tym banku spłacone": "A31",
    "Obecne kredyty spłacane terminowo": "A32",
    "Opóźnienia w spłacie w przeszłości": "A33",
    "Konto krytyczne / kredyty w innych bankach": "A34"
}

col3, col4 = st.columns(2)

with col3:
    status_konta_label = st.selectbox("Stan rachunku bieżącego", list(status_konta_map.keys()))
    stan_oszczednosci_label = st.selectbox("Oszczędności", list(stan_oszczednosci_map.keys()))

with col4:
    historia_kredytowa_label = st.selectbox("Historia kredytowa", list(historia_kredytowa_map.keys()), index=2)

status_konta = status_konta_map[status_konta_label]
stan_oszczednosci = stan_oszczednosci_map[stan_oszczednosci_label]
historia_kredytowa = historia_kredytowa_map[historia_kredytowa_label]

st.subheader("Sytuacja osobista i zawodowa")

zatrudnienie_map = {
    "Bezrobotny": "A71",
    "Poniżej 1 roku": "A72",
    "1 - 4 lata": "A73",
    "4 - 7 lat": "A74",
    "Powyżej 7 lat": "A75"
}

zawod_map = {
    "Bezrobotny / niewykwalifikowany, nierezydent": "A171",
    "Niewykwalifikowany, rezydent": "A172",
    "Wykwalifikowany pracownik / urzędnik": "A173",
    "Kadra zarządzająca / samozatrudniony": "A174"
}

stan_cywilny_map = {
    "Mężczyzna - rozwiedziony / w separacji": "A91",
    "Kobieta - rozwiedziona / zamężna": "A92",
    "Mężczyzna - kawaler": "A93",
    "Mężczyzna - żonaty / wdowiec": "A94"
}

mieszkanie_map = {
    "Wynajem": "A151",
    "Własność": "A152",
    "Bezpłatnie / u rodziny": "A153"
}

majatek_map = {
    "Nieruchomość": "A121",
    "Ubezpieczenie na życie / oszczędności budowlane": "A122",
    "Samochód lub inne": "A123",
    "Brak / nieznany": "A124"
}

cel_kredytu_map = {
    "Samochód nowy": "A40",
    "Samochód używany": "A41",
    "Meble / wyposażenie": "A42",
    "RTV / AGD": "A43",
    "Sprzęt AGD": "A44",
    "Remont": "A45",
    "Edukacja": "A46",
    "Przekwalifikowanie": "A48",
    "Biznes": "A49",
    "Inne": "A410"
}

inni_dluznicy_map = {
    "Brak": "A101",
    "Współkredytobiorca": "A102",
    "Poręczyciel": "A103"
}

inne_raty_map = {
    "W banku": "A141",
    "W sklepie": "A142",
    "Brak": "A143"
}

telefon_map = {
    "Brak": "A191",
    "Tak, zarejestrowany": "A192"
}

pracownik_zagraniczny_map = {
    "Tak": "A201",
    "Nie": "A202"
}

col5, col6 = st.columns(2)

with col5:
    zatrudnienie_label = st.selectbox("Staż w obecnej pracy", list(zatrudnienie_map.keys()), index=2)
    zawod_label = st.selectbox("Rodzaj zatrudnienia", list(zawod_map.keys()), index=2)
    stan_cywilny_label = st.selectbox("Płeć i stan cywilny", list(stan_cywilny_map.keys()), index=2)
    mieszkanie_label = st.selectbox("Sytuacja mieszkaniowa", list(mieszkanie_map.keys()), index=1)
    majatek_label = st.selectbox("Posiadany majątek", list(majatek_map.keys()))


with col6:
    cel_kredytu_label = st.selectbox("Cel kredytu", list(cel_kredytu_map.keys()), index=3)
    inni_dluznicy_label = st.selectbox("Inni dłużnicy / poręczyciele", list(inni_dluznicy_map.keys()))
    inne_raty_label = st.selectbox("Inne zobowiązania ratalne", list(inne_raty_map.keys()), index=2)
    telefon_label = st.selectbox("Telefon", list(telefon_map.keys()))
    pracownik_zagraniczny_label = st.selectbox("Pracownik zagraniczny", list(pracownik_zagraniczny_map.keys()))


zatrudnienie_od = zatrudnienie_map[zatrudnienie_label]
zawod = zawod_map[zawod_label]
stan_cywilny_plec = stan_cywilny_map[stan_cywilny_label]
mieszkanie = mieszkanie_map[mieszkanie_label]
majatek = majatek_map[majatek_label]
cel_kredytu = cel_kredytu_map[cel_kredytu_label]
inni_dluznicy = inni_dluznicy_map[inni_dluznicy_label]
inne_raty = inne_raty_map[inne_raty_label]
telefon = telefon_map[telefon_label]
pracownik_zagraniczny = pracownik_zagraniczny_map[pracownik_zagraniczny_label]


st.divider()

if st.button("Oceń ryzyko", type="primary"):

    client_data = {
        'status_konta': status_konta,
        'czas_trwania_msc': czas_trwania_msc,
        'historia_kredytowa': historia_kredytowa,
        'cel_kredytu': cel_kredytu,
        'kwota_kredytu': kwota_kredytu,
        'stan_oszczednosci': stan_oszczednosci,
        'zatrudnienie_od': zatrudnienie_od,
        'raty_proc_dochodu': raty_proc_dochodu,
        'stan_cywilny_plec': stan_cywilny_plec,
        'inni_dluznicy': inni_dluznicy,
        'lata_w_miejscu_zamieszkania': lata_w_miejscu_zamieszkania,
        'majatek': majatek,
        'wiek': wiek,
        'inne_raty': inne_raty,
        'mieszkanie': mieszkanie,
        'liczba_kredytow_w_banku': liczba_kredytow_w_banku,
        'zawod': zawod,
        'liczba_osob_na_utrzymaniu': liczba_osob_na_utrzymaniu,
        'telefon': telefon,
        'pracownik_zagraniczny': pracownik_zagraniczny
    }

    prediction, probability = predict.predict_client(model, client_data, columns, scaler)

    st.subheader("Wynik")

    if prediction == 0:
        st.success("Model przewiduje, że klient **spłaci** kredyt.")
    else:
        st.error("Model przewiduje, że klient **nie spłaci** kredytu.")

    st.metric("Prawdopodobieństwo niespłacenia", f"{probability:.1%}")

    st.progress(float(probability))

    st.caption(
        "Wynik pochodzi z modelu regresji logistycznej (ROC-AUC 0.807) wytrenowanego "
        "na 1000 historycznych wniosków. Model wykrywa około 78% klientów, którzy "
        "faktycznie nie spłacili kredytu, ale około 45% ostrzeżeń to fałszywe alarmy. "
        "Wynik ma charakter poglądowy."
    )