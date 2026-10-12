import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image

# ===== KONFIGURACJA =====
# Używamy dokładnych wartości z Fiji
WARTOSC_TLO = 0
WARTOSC_SKORA = 1
WARTOSC_ZMIANA = 2

KATALOG_SKRYPTU = os.path.dirname(os.path.abspath(__file__))
KATALOG_DANYCH = os.path.join(KATALOG_SKRYPTU, 'data')
KATALOG_WYNIKOW = os.path.join(KATALOG_SKRYPTU, 'scores')
plik_maski = os.path.join(KATALOG_DANYCH, 'veh_new_2.tif')
# ========================

def uruchom_analize():
    if not os.path.exists(plik_maski):
        print(f"BŁĄD: Nie znaleziono pliku '{plik_maski}'.")
        return

    # Używamy PIL, aby odczytać surowe indeksy klas (0, 1, 2)
    img = Image.open(plik_maski)
    maska = np.array(img)
    
    unikalne = np.unique(maska)
    print(f"INFO: Znalezione wartości pikseli w masce: {unikalne}")

    # Tworzymy czarną maskę OpenCV do szukania konturów (wymaga formatu uint8)
    obszar_myszy = np.zeros_like(maska, dtype=np.uint8)
    obszar_myszy[maska == WARTOSC_SKORA] = 255
    obszar_myszy[maska == WARTOSC_ZMIANA] = 255

    # Szukanie obrysów myszy
    contours, _ = cv2.findContours(obszar_myszy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    minimalne_pole = 5000 
    dobre_kontury = [c for c in contours if cv2.contourArea(c) > minimalne_pole]
    dobre_kontury = sorted(dobre_kontury, key=lambda c: cv2.boundingRect(c)[0])
    
    print(f"INFO: Znaleziono {len(dobre_kontury)} myszy po odfiltrowaniu szumów.\n")
    
    wyniki = []
    
    for idx, contour in enumerate(dobre_kontury, start=1):
        nazwa_myszy = f"M{idx}"
        
        # Wyizolowanie obszaru (bounding area) dla pojedynczej myszy
        maska_pojedyncza = np.zeros_like(maska, dtype=np.uint8)
        cv2.drawContours(maska_pojedyncza, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Logiczna maska wskazująca tylko wnętrze obecnej myszy
        obszar_bool = (maska_pojedyncza == 255)
        
        # Zliczamy piksele na podstawie wartości 0, 1, 2 wewnątrz obszaru myszy
        piksele_skory = np.sum((maska == WARTOSC_SKORA) & obszar_bool)
        piksele_zmiany = np.sum((maska == WARTOSC_ZMIANA) & obszar_bool)
        
        pole_calkowite = piksele_skory + piksele_zmiany
        
        if pole_calkowite == 0:
            score = 0.0
        else:
            score = (piksele_zmiany / pole_calkowite) * 100
            
        wyniki.append({
            "Mysz": nazwa_myszy,
            "Score (%)": round(score, 1)
        })
        
        print(f"{nazwa_myszy}: {score:.1f}% (Piksele zmiany: {piksele_zmiany}, Piksele czystej skóry: {piksele_skory})")
        
    if wyniki:
        df = pd.DataFrame(wyniki)
        os.makedirs(KATALOG_WYNIKOW, exist_ok=True)
        plik_wynikow = os.path.join(KATALOG_WYNIKOW, 'wyniki_scoringu.csv')
        df.to_csv(plik_wynikow, index=False, sep=';')
        print(f"\nZapisano wyniki do pliku: {plik_wynikow}")
        
        srednia = df["Score (%)"].mean()
        odchylenie = df["Score (%)"].std()
        print(f"Średnia ± SD dla całej grupy: {srednia:.1f}% ± {odchylenie:.1f}%")

if __name__ == "__main__":
    uruchom_analize()