import cv2
import numpy as np
import pandas as pd
import os

KATALOG_SKRYPTU = os.path.dirname(os.path.abspath(__file__))
KATALOG_DANYCH = os.path.join(KATALOG_SKRYPTU, 'data')
KATALOG_WYNIKOW = os.path.join(KATALOG_SKRYPTU, 'scores')

PLIKI_ZDJEC = ['vehicle.jpeg', 'trap1.jpeg']
# ======================================================

def policz_score_dla_pliku(plik_zdjecia):
    sciezka_zdjecia = os.path.join(KATALOG_DANYCH, plik_zdjecia)
    if not os.path.exists(sciezka_zdjecia):
        print(f"BŁĄD: Nie znaleziono pliku '{sciezka_zdjecia}'.")
        return None

    img = cv2.imread(sciezka_zdjecia)
    if img is None:
        print(f"BŁĄD: Nie można odczytać pliku '{sciezka_zdjecia}'.")
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 100, 100])
    upper_green = np.array([85, 255, 255])
    maska_obrysu = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    maska_obrysu = cv2.morphologyEx(maska_obrysu, cv2.MORPH_CLOSE, kernel)

    lower_blue = np.array([90, 100, 100])
    upper_blue = np.array([130, 255, 255])
    maska_zmian = cv2.inRange(hsv, lower_blue, upper_blue)

    contours, _ = cv2.findContours(maska_obrysu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dobre_kontury = [c for c in contours if cv2.contourArea(c) > 5000]
    dobre_kontury = sorted(dobre_kontury, key=lambda c: cv2.boundingRect(c)[0])

    print(f"INFO: {plik_zdjecia} -> wykryto {len(dobre_kontury)} myszy.\n")

    wyniki = []
    for idx, contour in enumerate(dobre_kontury, start=1):
        nazwa_myszy = f"M{idx}"

        maska_pojedynczej_skory = np.zeros(maska_obrysu.shape, dtype=np.uint8)
        cv2.drawContours(maska_pojedynczej_skory, [contour], -1, 255, thickness=cv2.FILLED)

        mysz_zmiany = cv2.bitwise_and(maska_zmian, maska_zmian, mask=maska_pojedynczej_skory)

        piksele_skory = cv2.countNonZero(maska_pojedynczej_skory)
        piksele_zmiany = cv2.countNonZero(mysz_zmiany)

        score = 0.0 if piksele_skory == 0 else (piksele_zmiany / piksele_skory) * 100

        wyniki.append({
            "Mysz": nazwa_myszy,
            "Pole ogolonej skóry (px)": piksele_skory,
            "Pole zmian (px)": piksele_zmiany,
            "Score (%)": round(score, 1)
        })

        print(f"{nazwa_myszy}: {score:.1f}% (pole zmian: {piksele_zmiany}, pole skóry: {piksele_skory})")

    if wyniki:
        df = pd.DataFrame(wyniki)
        os.makedirs(KATALOG_WYNIKOW, exist_ok=True)
        nazwa_wynikowa = os.path.join(
            KATALOG_WYNIKOW,
            f"wyniki_{os.path.splitext(os.path.basename(plik_zdjecia))[0]}.csv"
        )
        df.to_csv(nazwa_wynikowa, index=False, sep=';')

        srednia = df["Score (%)"].mean()
        odchylenie = df["Score (%)"].std()
        print(f"\nŚrednia ± SD: {srednia:.1f}% ± {odchylenie:.1f}%")
        print(f"Zapisano wyniki do pliku: {nazwa_wynikowa}")

    return wyniki


def uruchom_analize():
    for plik in PLIKI_ZDJEC:
        policz_score_dla_pliku(plik)
        print("-" * 60)


if __name__ == "__main__":
    uruchom_analize()