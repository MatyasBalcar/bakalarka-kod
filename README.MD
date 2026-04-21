# Testování náhodnosti bitových posloupností

Projekt porovnává generátory bitů (PRNG, CSPRNG, TRNG) pomocí statistických testů. Aplikace podporuje režimy `cherry-pick` a `benchmark`.

## Požadavky
- Python 3.10+

## Instalace
Základní závislosti:
```bash
pip install numpy scipy numba tqdm
```

Volitelně (pouze pro `Ambient Noise Generator`):
```bash
pip install sounddevice
```

## Spuštění
```bash
python main.py
```

Běh aplikace je řízen souborem `config.json`.

## Režimy běhu

### `cherry-pick`
- Interaktivní výběr testů a generátorů.
- Vyhodnocení nad `sample_iter` vzorky o délce `sample_size` bitů.
- Export výsledků do textového reportu a CSV souborů (`metrics`, `summary`).

### `benchmark`
- Dávkové měření výkonu a výsledků testů (čas, paměť, p-hodnoty, `pass_rate`).
- Spuštění přes všechny generátory, velikosti vzorků a opakování dle konfigurace.
- Export metrik po testech a agregace podle třídy generátoru.

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
1. Načtení audio rámců (`int16`).
2. Extrakce LSB.
3. Bělení (`none`, `von-neumann`, `sha256`, `von-neumann+sha256`).
4. Sestavení výstupní bitové posloupnosti.

Data se ukládají do `inputs/audio` jako `.bin` soubory. Jeden soubor odpovídá jednomu vzorku pro testování.

## Konfigurace (`config.json`)

Podporované klíče:
- `mode`: `cherry-pick` nebo `benchmark`
- `sample_size`: délka jednoho vzorku v bitech
- `sample_iter`: počet vzorků v režimu `cherry-pick`
- `alpha`: hladina významnosti testů
- `bayes_pass_threshold`: rozhodovací práh pro Bayes verdikt (viz „Interpretace metrik“)
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

Výstupy se ukládají do `outputs/<Generator Name>/`, a ty použité v práci jsou uloženy ve složkách s jejich parametry, tedy například 500*1000000.

### Textový report
- `output_<mode>-<timestamp>.txt`
- Obsahuje konfiguraci běhu, metriky testů, verdikty a agregované souhrny.

### CSV (`cherry-pick`)
- `...-cherry_pick_metrics.csv`
- `...-cherry_pick_summary.csv`

`cherry_pick_metrics.csv` obsahuje mimo jiné:
- `pass_rate`
- `mean_p`, `median_p`
- `pass_posterior_chance`
- `classical_randomness_chance`, `classical_fail_chance`
- `bayes_randomness_chance`, `bayes_fail_chance`
- `verdict` (klasický verdikt na základě `pass_rate`), `bayes_verdict`

`cherry_pick_summary.csv` obsahuje agregace přes testy pro obě větve:
- classical (`avg_`, `geo_`, `min` přes testy),
- bayes (`avg_`, `geo_`, `min` přes testy).

Pole `randomness_chance` a `fail_chance` jsou ponechána jako kompatibilní aliasy pro Bayes větev.

### CSV (`benchmark`)
- `...-benchmark_metrics.csv`
- `...-benchmark_class_summary.csv`

## Interpretace metrik

- Výsledky obsahují p-hodnoty, ukazatele uniformity a další souhrnné statistiky.
- Klasický verdikt `PASS/FAIL` je určen porovnáním `pass_rate` s NIST proporčním prahem `min_pass_rate` (tj. minimální požadovaný podíl úspěchů napříč `sample_iter` iteracemi daného testu). Hodnota `min_pass_rate` je prahová hodnota definovaná metodikou NIST a typicky je odvozena z `alpha` a počtu iterací.
- Bayesovský verdikt (`bayes_verdict`) je doplňkový a vychází z porovnání `bayes_randomness_chance` vůči `bayes_pass_threshold` (konfigurační práh v `config.json`).
- `classical_randomness_chance = pass_rate`.
- `bayes_randomness_chance = P(theta >= min_pass_rate | data)`, kde `data` reprezentuje počty úspěchů/neúspěchů testu napříč iteracemi; použit je model Beta-Binomial s Jeffreys prior.

## Implementované testy
- Frequency (Monobit)
- Runs
- Block Frequency
- Autocorrelation
- Spectral (DFT)
- Linear Complexity
- Diehard Birthday Spacings (implementace inspirovaná sadou Diehard)
- Dieharder Byte Distribution (implementace inspirovaná sadou Dieharder)
