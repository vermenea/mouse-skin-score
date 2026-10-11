import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats


WARTOSC_TLO = 0
WARTOSC_SKORA = 1
WARTOSC_ZMIANA = 2
MINIMALNE_POLE_MYSZY = 10000


def wczytaj_csv(nazwa_pliku):
    sciezka = os.path.join(os.getcwd(), nazwa_pliku)
    if not os.path.exists(sciezka):
        raise FileNotFoundError(f"Brak pliku: {nazwa_pliku}")
    return pd.read_csv(sciezka, sep=';')


def znajdz_plik(nazwy_plikow):
    for nazwa in nazwy_plikow:
        if os.path.exists(nazwa):
            return nazwa
    return None


def analiza_maszyn_z_tiff(sciezka):
    obraz = Image.open(sciezka)
    maska = np.array(obraz)

    if len(maska.shape) == 3:
        maska = cv2.cvtColor(maska, cv2.COLOR_BGR2GRAY)

    maska_skory = (maska == WARTOSC_SKORA).astype(np.uint8) * 255
    maska_zmian = (maska == WARTOSC_ZMIANA).astype(np.uint8) * 255
    maska_ciala = cv2.bitwise_or(maska_skory, maska_zmian)

    kontury, _ = cv2.findContours(maska_ciala, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    myszy = []
    for kontur in kontury:
        if cv2.contourArea(kontur) > MINIMALNE_POLE_MYSZY:
            x, y, w, h = cv2.boundingRect(kontur)
            myszy.append({'kontur': kontur, 'x': x})

    myszy = sorted(myszy, key=lambda item: item['x'])
    wyniki = []

    for item in myszy:
        kontur = item['kontur']
        pojedyncza_mysz_maska = np.zeros_like(maska_ciala)
        cv2.drawContours(pojedyncza_mysz_maska, [kontur], -1, 255, -1)

        skora_myszy = cv2.bitwise_and(maska_skory, maska_skory, mask=pojedyncza_mysz_maska)
        zmiany_myszy = cv2.bitwise_and(maska_zmian, maska_zmian, mask=pojedyncza_mysz_maska)

        piksele_skory = cv2.countNonZero(skora_myszy)
        piksele_zmian = cv2.countNonZero(zmiany_myszy)
        calkowita_powierzchnia = piksele_skory + piksele_zmian

        score = (piksele_zmian / calkowita_powierzchnia) * 100 if calkowita_powierzchnia > 0 else 0.0
        wyniki.append(score)

    return np.array(wyniki, dtype=float)


def porownaj_grupy(df_choroba, df_kontrola):
    wyniki_veh = df_choroba['Score (%)']
    wyniki_trap1 = df_kontrola['Score (%)']

    t_stat, p_val_t = stats.ttest_ind(wyniki_veh, wyniki_trap1)
    u_stat, p_val_m = stats.mannwhitneyu(wyniki_veh, wyniki_trap1)

    print(f"Średnia choroba: {wyniki_veh.mean():.1f}%")
    print(f"Średnia kontrola: {wyniki_trap1.mean():.1f}%\n")

    print(f"Wynik testu t-Studenta (p-value): {p_val_t:.4f}")
    print(f"Wynik testu Manna-Whitneya (p-value): {p_val_m:.4f}\n")

    if p_val_m < 0.05:
        print("Różnica jest istotna statystycznie (p < 0.05).")
    else:
        print("Brak istotności statystycznej (p >= 0.05).")


# 1. Najpierw spróbuj odczytać pliki CSV
if os.path.exists('wyniki_veh.csv') and os.path.exists('wyniki_trap1.csv'):
    df_choroba = wczytaj_csv('wyniki_veh.csv')
    df_kontrola = wczytaj_csv('wyniki_trap1.csv')

    if 'Score (%)' not in df_choroba.columns or 'Score (%)' not in df_kontrola.columns:
        raise ValueError("Pliki CSV muszą zawierać kolumnę 'Score (%)'.")

    porownaj_grupy(df_choroba, df_kontrola)

else:
    plik_kontrola = znajdz_plik(['vehicle.tif', 'vehicle.tiff'])
    plik_choroba = znajdz_plik(['maska_wynik.tif', 'maska_wynik.tiff', 'maska_wyniki.tiff'])

    if plik_kontrola and plik_choroba:
        print(f"Wykryto obrazy TIFF: kontrola={plik_kontrola}, choroba={plik_choroba}")

        wyniki_trap1 = analiza_maszyn_z_tiff(plik_kontrola)
        wyniki_veh = analiza_maszyn_z_tiff(plik_choroba)

        df_kontrola = pd.DataFrame({'Score (%)': wyniki_trap1})
        df_choroba = pd.DataFrame({'Score (%)': wyniki_veh})

        df_kontrola.to_csv('wyniki_trap1.csv', index=False, sep=';')
        df_choroba.to_csv('wyniki_veh.csv', index=False, sep=';')

        print("Zapisano wygenerowane pliki CSV: wyniki_trap1.csv, wyniki_veh.csv")
        porownaj_grupy(df_choroba, df_kontrola)
    else:
        print("Nie znaleziono wymaganych plików wejściowych.")
        print("Oczekiwane obrazy TIFF: 'vehicle.tif' / 'vehicle.tiff' oraz 'maska_wynik.tif' / 'maska_wynik.tiff' / 'maska_wyniki.tiff'.")
        print("Albo utwórz pliki CSV: 'wyniki_veh.csv' i 'wyniki_trap1.csv'.")