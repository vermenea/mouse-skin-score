import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats


WARTOSC_TLO = 0
WARTOSC_SKORA = 1
WARTOSC_ZMIANA = 2
MINIMALNE_POLE_MYSZY = 10000
KATALOG_SKRYPTU = os.path.dirname(os.path.abspath(__file__))
KATALOG_WYNIKOW = os.path.join(KATALOG_SKRYPTU, 'scores')


def wczytaj_csv(nazwa_pliku):
    sciezka = nazwa_pliku if os.path.isabs(nazwa_pliku) else os.path.join(KATALOG_SKRYPTU, nazwa_pliku)
    if not os.path.exists(sciezka):
        raise FileNotFoundError(f"Brak pliku: {nazwa_pliku}")
    return pd.read_csv(sciezka, sep=';')


def znajdz_plik(nazwy_plikow):
    for nazwa in nazwy_plikow:
        sciezka = os.path.join(KATALOG_SKRYPTU, nazwa)
        if os.path.exists(sciezka):
            return sciezka
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


def porownaj_grupy(df_choroba, df_kontrola, nazwa_choroba='veh_new_2', nazwa_kontrola='trap1_new_2'):
    wyniki_veh = df_choroba['Score (%)']
    wyniki_trap1 = df_kontrola['Score (%)']

    t_stat, p_val_t = stats.ttest_ind(wyniki_veh, wyniki_trap1)
    u_stat, p_val_m = stats.mannwhitneyu(wyniki_veh, wyniki_trap1)

    generuj_wykresy(df_choroba, df_kontrola, nazwa_choroba, nazwa_kontrola, p_val_t, p_val_m)

    print(f"{nazwa_choroba}: {wyniki_veh.mean():.1f}% ± {wyniki_veh.std():.1f}% SD (n={len(wyniki_veh)})")
    print(f"{nazwa_kontrola}: {wyniki_trap1.mean():.1f}% ± {wyniki_trap1.std():.1f}% SD (n={len(wyniki_trap1)})\n")

    print(f"Wynik testu t-Studenta (p-value): {p_val_t:.4f}")
    print(f"Wynik testu Manna-Whitneya (p-value): {p_val_m:.4f}\n")

    if p_val_m < 0.05:
        print("Różnica jest istotna statystycznie (p < 0.05).")
    else:
        print("Brak istotności statystycznej (p >= 0.05).")


def generuj_wykresy(df_choroba, df_kontrola, nazwa_choroba, nazwa_kontrola, p_val_t, p_val_m):
    os.makedirs(KATALOG_WYNIKOW, exist_ok=True)
    kolory = ['#d62728', '#1f77b4']
    grupy = [
        (df_choroba, nazwa_choroba, kolory[0]),
        (df_kontrola, nazwa_kontrola, kolory[1]),
    ]
    mapy_wynikow = []
    etykiety_myszy = []

    for df, _, _ in grupy:
        etykiety = df['Mysz'].astype(str).tolist() if 'Mysz' in df.columns else [
            f'M{i}' for i in range(1, len(df) + 1)
        ]
        wyniki = df['Score (%)'].astype(float).tolist()
        mapa = dict(zip(etykiety, wyniki))
        if len(mapa) != len(etykiety):
            raise ValueError("Kolumna 'Mysz' nie może zawierać powtarzających się identyfikatorów.")
        mapy_wynikow.append(mapa)
        for etykieta in etykiety:
            if etykieta not in etykiety_myszy:
                etykiety_myszy.append(etykieta)

    x = np.arange(len(etykiety_myszy))
    szerokosc = 0.36
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (grupa, nazwa, kolor) in enumerate(grupy):
        wartosci = [mapy_wynikow[i].get(etykieta, np.nan) for etykieta in etykiety_myszy]
        slupki = ax.bar(x + (i - 0.5) * szerokosc, wartosci, szerokosc, label=nazwa,
                        color=kolor, edgecolor='black')
        for slupek, wartosc in zip(slupki, wartosci):
            if not np.isnan(wartosc):
                ax.annotate(f'{wartosc:.1f}%', (slupek.get_x() + slupek.get_width() / 2, wartosc),
                            xytext=(0, 5), textcoords='offset points', ha='center', va='bottom')

    ax.set_title('Porównanie score dla każdej myszy')
    ax.set_ylabel('Score (%)')
    ax.set_xticks(x, etykiety_myszy)
    ax.legend()
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(KATALOG_WYNIKOW, 'podsumowanie_wynikow_czerwony_niebieski.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    boxplot = ax.boxplot(
        [df_choroba['Score (%)'].astype(float), df_kontrola['Score (%)'].astype(float)],
        tick_labels=[nazwa_choroba, nazwa_kontrola],
        patch_artist=True,
        showfliers=False,
    )
    for pudelko, kolor in zip(boxplot['boxes'], kolory):
        pudelko.set_facecolor(kolor)
        pudelko.set_alpha(0.8)

    generator = np.random.default_rng(0)
    for i, (df, _, kolor) in enumerate(grupy, start=1):
        wartosci = df['Score (%)'].astype(float).to_numpy()
        jitter = generator.uniform(-0.08, 0.08, size=len(wartosci))
        ax.scatter(i + jitter, wartosci, color='black', alpha=0.75, zorder=3)
        ax.annotate(f'średnia {wartosci.mean():.1f}%', (i, wartosci.max()),
                    xytext=(0, 8), textcoords='offset points', ha='center')

    dol, gora = ax.get_ylim()
    zakres = max(gora - dol, 1)
    ax.set_ylim(dol, gora + zakres * 0.3)
    ax.text(
        0.5, 0.98,
        f'p (t-test) = {p_val_t:.4f}\np (M-W) = {p_val_m:.4f}',
        transform=ax.transAxes, ha='center', va='top',
        bbox={'facecolor': 'white', 'edgecolor': '#bdbdbd', 'boxstyle': 'square,pad=0.5'},
    )
    ax.set_title('Rozkład score w grupach')
    ax.set_ylabel('Score (%)')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(KATALOG_WYNIKOW, 'statystyka_boxplot_pvalue_czerwony_niebieski.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


# Najpierw użyj wskazanych wyników scoringu.
plik_scoringu_csv = os.path.join(KATALOG_WYNIKOW, 'wyniki_scoringu.csv')
plik_scoringu_trap1_csv = os.path.join(KATALOG_WYNIKOW, 'wyniki_scoringu_trap1.csv')
if os.path.exists(plik_scoringu_csv) and os.path.exists(plik_scoringu_trap1_csv):
    df_choroba = wczytaj_csv(plik_scoringu_csv)
    df_kontrola = wczytaj_csv(plik_scoringu_trap1_csv)

    if 'Score (%)' not in df_choroba.columns or 'Score (%)' not in df_kontrola.columns:
        raise ValueError("Pliki CSV muszą zawierać kolumnę 'Score (%)'.")

    print("Porównanie wyników: wyniki_scoringu.csv i wyniki_scoringu_trap1.csv")
    porownaj_grupy(df_choroba, df_kontrola, 'veh', 'trap1')

# Zachowaj obsługę wyników dla nowych masek oraz starszych plików.
elif os.path.exists(os.path.join(KATALOG_WYNIKOW, 'wyniki_veh_new_2.csv')) and os.path.exists(os.path.join(KATALOG_WYNIKOW, 'wyniki_trap1_new_2.csv')):
    plik_veh_csv = os.path.join(KATALOG_WYNIKOW, 'wyniki_veh_new_2.csv')
    plik_trap1_csv = os.path.join(KATALOG_WYNIKOW, 'wyniki_trap1_new_2.csv')
    df_choroba = wczytaj_csv(plik_veh_csv)
    df_kontrola = wczytaj_csv(plik_trap1_csv)

    if 'Score (%)' not in df_choroba.columns or 'Score (%)' not in df_kontrola.columns:
        raise ValueError("Pliki CSV muszą zawierać kolumnę 'Score (%)'.")

    print("Porównanie wyników: wyniki_veh_new_2.csv i wyniki_trap1_new_2.csv")
    porownaj_grupy(df_choroba, df_kontrola)

elif os.path.exists(os.path.join(KATALOG_SKRYPTU, 'wyniki_veh.csv')) and os.path.exists(os.path.join(KATALOG_SKRYPTU, 'wyniki_trap1.csv')):
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