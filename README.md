# Mouse Skin Score

Projekt służy do szacowania odsetka zmian skórnych u myszy na podstawie zdjęć lub masek segmentacyjnych oraz do porównywania wyników między dwiema grupami.

## Skrypty

### `Fiji_analysis.py`

Analizuje maskę TIFF `data/trap1.tif`. Maska powinna zawierać wartości pikseli:

- `0` — tło,
- `1` — skóra,
- `2` — zmiany.

Skrypt wykrywa myszy na podstawie obszarów skóry i zmian, odrzuca kontury o polu nie większym niż 5000 px, a następnie oblicza dla każdej myszy:

`Score (%) = liczba pikseli zmian / (piksele skóry + piksele zmian) × 100`

Wyniki zapisuje do `scores/wyniki_scoringu.csv` (separator `;`) i wypisuje średnią oraz odchylenie standardowe.

### `Manual_analysis.py`

Analizuje zdjęcia `data/vehicle.jpeg` i `data/trap1.jpeg`. Wykorzystuje progi kolorów w przestrzeni HSV do wykrywania obrysu skóry (zielony) i zmian (niebieski), a następnie oblicza score dla każdej wykrytej myszy.

Wyniki zapisuje osobno do `scores/wyniki_vehicle.csv` i `scores/wyniki_trap1.csv`. Wymagane zdjęcia muszą być dostępne w katalogu `data`.

### `statystyka.py`

Porównuje score dwóch grup. W pierwszej kolejności odczytuje `scores/wyniki_veh_new_2.csv` i `scores/wyniki_trap1_new_2.csv`. Jeśli ich nie znajdzie, obsługuje starsze pliki `wyniki_veh.csv` i `wyniki_trap1.csv` w katalogu projektu albo szuka masek TIFF:

- kontrola: `vehicle.tif` lub `vehicle.tiff`,
- choroba: `maska_wynik.tif`, `maska_wynik.tiff` lub `maska_wyniki.tiff`.

Dla masek używa tych samych wartości klas co `Fiji_analysis.py`, ale odrzuca kontury o polu nie większym niż 10000 px. Raportuje średnią ± SD i liczebność obu grup oraz wyniki testu t-Studenta i testu Manna–Whitneya. Wniosek o istotności statystycznej opiera na teście Manna–Whitneya przy progu `p < 0,05`.

Skrypt szuka plików wejściowych względem katalogu projektu, niezależnie od bieżącego katalogu roboczego.

## Uruchomienie

Projekt korzysta z Pythona oraz bibliotek OpenCV, NumPy, pandas, Pillow i SciPy. Przykład konfiguracji środowiska w PowerShell z katalogu projektu:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install opencv-python numpy pandas pillow scipy
```

Uruchom wybrany skrypt:

```powershell
python Fiji_analysis.py
python Manual_analysis.py
python statystyka.py
```

Skrypty `Fiji_analysis.py` i `Manual_analysis.py` odwołują się do katalogów `data` i `scores` względem lokalizacji skryptu. `scores` jest tworzony automatycznie, jeśli trzeba. Pliki wynikowe i wejściowe nie są automatycznie czyszczone.

Lokalne środowisko `.venv` jest pomijane przez Git (zob. `.gitignore`); po sklonowaniu repozytorium należy utworzyć je ponownie i zainstalować zależności.
