# Testování náhodnosti bitových posloupností

Projekt porovnává generátory bitů (PRNG, CSPRNG, TRNG) pomocí statistických testů.
Aplikace podporuje režimy `cherry-pick` a `benchmark`.

## Požadavky
- Python 3.10+

## Instalace
```bash
pip install numpy scipy numba tqdm sounddevice
```

Balíček `sounddevice` je potřeba pouze pro `Ambient Noise Generator`.

## Spuštění
```bash
python main.py
```

## Režimy běhu

### `cherry-pick`
- interaktivní výběr testů a generátorů,
- vyhodnocení nad `sample_iter` vzorky o délce `sample_size` bitů,
- výstup do textového reportu a CSV (`metrics`, `summary`).

### `benchmark`
- dávkové měření času, paměti, p-hodnot a pass-rate,
- běh přes všechny generátory, velikosti vzorků a opakování,
- export metrik po testech a agregace podle třídy generátoru.

## Generátory

### PRNG
- `LCG`
- `Mersenne Twister`
- `PCG64`
- `XORShift32`

### CSPRNG
- `Blum-Blum-Shub`

### TRNG
- `Ambient Noise Generator`

### Kontrolní zdroje
- `Alternating Generator`
- `Repeating Generator`

## Ambientní data
`Ambient Noise Generator` pracuje s daty z mikrofonu:
1. načtení audio rámců (`int16`),
2. extrakce LSB,
3. bělení (`none`, `von-neumann`, `sha256`, `von-neumann+sha256`),
4. vytvoření výstupní bitové posloupnosti.

Data se ukládají do `inputs/audio` jako `.bin` soubory.
Jeden soubor odpovídá jednomu vzorku pro testování.

## Konfigurace (`config.json`)
Podporované klíče:
- `mode`: `cherry-pick` nebo `benchmark`
- `sample_size`: délka jednoho vzorku v bitech
- `sample_iter`: počet vzorků v `cherry-pick`
- `alpha`: hladina významnosti testů
- `bayes_pass_threshold`: práh pro Bayes verdikt
- `output_dir`: kořen výstupních souborů
- `benchmark_sample_sizes`: seznam délek vzorků pro `benchmark`
- `benchmark_repeats`: počet opakování na kombinaci
- `ambient_sample_rate`: vzorkovací frekvence mikrofonu
- `ambient_whitening`: metoda bělení
- `ambient_hash_block_bytes`: velikost bloku pro SHA-256 bělení

Aktuální příklad konfigurace:
```json
{
  "sample_size": 1000000,
  "sample_iter": 500,
  "output_dir": "outputs",
  "mode": "benchmark",
  "alpha": 0.01,
  "bayes_pass_threshold": 0.95,
  "ambient_sample_rate": 48000,
  "ambient_whitening": "von-neumann+sha256",
  "ambient_hash_block_bytes": 32,
  "benchmark_sample_sizes": [1000000],
  "benchmark_repeats": 500
}
```

## Výstupy
Výstupy se ukládají do `outputs/<Generator Name>/`.

### Textový report
- `output_<mode>-<timestamp>.txt`
- obsahuje konfiguraci běhu, metriky testů, verdikty a agregované souhrny.

### CSV (`cherry-pick`)
- `...-cherry_pick_metrics.csv`
- `...-cherry_pick_summary.csv`

`cherry_pick_metrics.csv` obsahuje mimo jiné:
- `pass_rate`
- `mean_p`, `median_p`
- `pass_posterior_chance`
- `classical_randomness_chance`, `classical_fail_chance`
- `bayes_randomness_chance`, `bayes_fail_chance`
- `verdict` (pass-rate/NIST), `bayes_verdict`

`cherry_pick_summary.csv` obsahuje agregace přes testy pro obě větve:
- classical (`avg_`, `geo_`, `min` přes testy),
- bayes (`avg_`, `geo_`, `min` přes testy).

Pole `randomness_chance` a `fail_chance` jsou zachována jako kompatibilní aliasy Bayes větve.

### CSV (`benchmark`)
- `...-benchmark_metrics.csv`
- `...-benchmark_class_summary.csv`

## Interpretace metrik
- Hlavní verdikt `PASS/FAIL` je určen přes `pass_rate` vůči NIST proporčnímu prahu `min_pass_rate`.
- Bayes verdikt je doplňkový a vychází z `bayes_randomness_chance` vůči `bayes_pass_threshold`.
- `classical_randomness_chance = pass_rate`.
- `bayes_randomness_chance = P(theta >= min_pass_rate | data)` při modelu Beta-Binomial s Jeffreys prior.

## Implementované testy
- Frequency (Monobit)
- Runs
- Block Frequency
- Autocorrelation
- Spectral (DFT)
- Linear Complexity
- Diehard Birthday Spacings (inspirovaný)
- Dieharder Byte Distribution (inspirovaný)
