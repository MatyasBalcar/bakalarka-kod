# Testování náhodnosti bitových posloupností

Projekt obsahuje nástroj pro generování a statistické testování bitových posloupností. Umožňuje porovnávat různé typy generátorů náhodných a pseudonáhodných dat, spouštět nad nimi vybranou sadu statistických testů a ukládat výsledky do textových a CSV reportů.

Aplikace podporuje dva hlavní režimy běhu:

- `single-source` – podrobná analýza jednoho vybraného zdroje dat,
- `benchmark` – dávkové měření více generátorů a testů včetně časové a paměťové náročnosti.

---

## Požadavky

- Python 3.10 nebo novější

Základní závislosti:

```bash
pip install numpy scipy numba tqdm
```

Volitelná závislost pro práci se zvukovým vstupem:

```bash
pip install sounddevice
```

Balíček `sounddevice` je potřeba pouze v případě, že se používá `Ambient Noise Generator` pro přímé získávání dat z mikrofonu.

---

## Spuštění programu

Program se spouští příkazem:

```bash
python main.py
```

Běh aplikace je řízen konfiguračním souborem `config.json`. Po spuštění může program podle zvoleného režimu nabídnout interaktivní výběr testů nebo generátorů.

---

## Struktura projektu

Hlavní soubory projektu:

- `main.py` – řídicí skript aplikace, načítání konfigurace, spouštění režimů a ukládání výstupů,
- `generators.py` – implementace generátorů a zdrojů vstupních dat,
- `tests.py` – implementace statistických testů a výpočet souhrnných metrik,
- `ui.py` – jednoduché konzolové rozhraní pro výběr testů a generátorů,
- `collect_entropy_data.py` – pomocný skript pro sběr entropie a ukládání do `.bin` souborů,
- `config.json` – hlavní konfigurační soubor.

---

## Režimy běhu

### `single-source`

Režim `single-source` slouží k podrobné analýze jednoho zvoleného zdroje dat.

V tomto režimu program:

1. vybere jeden generátor nebo datový zdroj,
2. vygeneruje nebo načte `sample_iter` vzorků,
3. každý vzorek má délku `sample_size` bitů,
4. nad každým vzorkem spustí zvolené statistické testy,
5. vypočítá p-hodnoty, podíl úspěšných průchodů a doplňkové metriky,
6. uloží textový report a CSV soubory.

Tento režim je vhodný hlavně pro detailní vyhodnocení konkrétního generátoru.

### `benchmark`

Režim `benchmark` slouží k dávkovému porovnání více generátorů a testů.

V tomto režimu program:

1. projde dostupné generátory,
2. otestuje je pro velikosti uvedené v `benchmark_sample_sizes`,
3. každý běh opakuje podle hodnoty `benchmark_repeats`,
4. měří průměrný čas generování,
5. měří průměrný čas a paměťovou náročnost testů,
6. ukládá detailní metriky i agregované výsledky podle tříd generátorů.

Tento režim je vhodný zejména pro výkonnostní srovnání.

---

## Implementované generátory

### PRNG

- `LCG`
- `XORShift32`
- `Mersenne Twister`
- `PCG64`

### CSPRNG

- `Blum-Blum-Shub`

### TRNG

- `Ambient Noise Generator`
- `AudioSampleBatchGenerator`

### Kontrolní zdroje

- `Alternating Generator`
- `Repeating Generator`

Kontrolní zdroje záměrně vytvářejí nenáhodné posloupnosti. Slouží k ověření, že testovací sada dokáže zachytit zjevné statistické odchylky.

---

## Ambientní data

`Ambient Noise Generator` získává data ze zvukového vstupu. Zpracování probíhá následovně:

1. načtení audio rámců ve formátu `int16`,
2. extrakce vybraného nejméně významného bitu,
3. volitelné bělení dat,
4. sestavení výsledné bitové posloupnosti.

Podporované režimy bělení:

- `none`,
- `von-neumann`,
- `sha256`,
- `von-neumann+sha256`.

Data získaná z ambientního šumu se ukládají jako `.bin` soubory. Jeden soubor odpovídá jednomu vzorku pro testování.

Výchozí složka pro audio data je:

```text
inputs/audio
```

Uživatel může přidat také vlastní datové sady do složky:

```text
inputs/custom/<nazev-datove-sady>
```

Každý vstupní soubor musí mít příponu `.bin`.

---

## Konfigurace

Běh programu se nastavuje v souboru `config.json`.

Příklad konfigurace:

```json
{
  "sample_size": 1000000,
  "sample_iter": 500,
  "output_dir": "outputs",
  "mode": "single-source",
  "alpha": 0.01,
  "bayes_pass_threshold": 0.95,
  "ambient_sample_rate": 48000,
  "ambient_whitening": "von-neumann+sha256",
  "ambient_hash_block_bytes": 32,
  "benchmark_sample_sizes": [1000000],
  "benchmark_repeats": 500
}
```

Význam hlavních položek:

- `mode` – režim běhu programu, tedy `single-source` nebo `benchmark`,
- `sample_size` – délka jednoho vzorku v bitech,
- `sample_iter` – počet vzorků v režimu `single-source`,
- `output_dir` – kořenová složka pro výstupní soubory,
- `alpha` – hladina významnosti pro jednotlivé statistické testy,
- `bayes_pass_threshold` – rozhodovací práh pro doplňkový Bayesovský verdikt,
- `ambient_sample_rate` – vzorkovací frekvence mikrofonu,
- `ambient_whitening` – zvolený způsob bělení dat z ambientního šumu,
- `ambient_hash_block_bytes` – velikost bloku vstupních dat pro SHA-256 bělení,
- `benchmark_sample_sizes` – velikosti vzorků testované v režimu `benchmark`,
- `benchmark_repeats` – počet opakování v režimu `benchmark`.

---

## Výstupy programu

Výstupy programu se ukládají do složky nastavené pomocí `output_dir`, standardně tedy do:

```text
outputs
```

Výstupy jsou dále členěny podle názvu generátoru, například:

```text
outputs/PCG64
outputs/Blum-Blum-Shub
outputs/Ambient Noise Generator
```

Každý nový běh programu vytváří soubory s časovým razítkem v názvu. Tím se snižuje riziko přepsání starších výsledků.

### Referenční výstupy použité v práci

Výsledky, které byly použity při zpracování bakalářské práce, jsou v projektu uložené odděleně ve složkách podle konfigurace běhu:

```text
outputs/<nazev-generatoru>/500*1000000
```

Toto členění označuje, že dané výsledky byly získány při nastavení:

- `sample_iter = 500`,
- `sample_size = 1000000`.

Tyto referenční soubory není nutné znovu generovat. Jsou v projektu ponechány jako podklad pro výsledky uvedené v práci.

Oddělení referenčních výstupů od nově vygenerovaných souborů má dva důvody:

1. výsledky použité v práci zůstávají snadno dohledatelné,
2. nové běhy programu se nemíchají s daty, která sloužila k vyhodnocení v práci.

Při vlastním testování mohou vznikat nové výstupy přímo ve složkách jednotlivých generátorů. Tyto nové soubory není třeba přesouvat do složek `500*1000000`, pokud neslouží jako referenční výstupy pro práci.

---

## Typy výstupních souborů

### Textový report

Textový report má název ve tvaru:

```text
output_<mode>-<timestamp>.txt
```

Obsahuje zejména:

- použitou konfiguraci,
- seznam spuštěných testů,
- počet úspěšných vzorků,
- p-hodnoty,
- verdikty testů,
- průměrné časy generování a testování,
- souhrnné výsledky pro daný generátor.

### CSV výstupy pro `single-source`

Režim `single-source` vytváří zejména soubory:

```text
...-single_source_metrics.csv
...-single_source_summary.csv
```

Soubor `single_source_metrics.csv` obsahuje detailní výsledky pro jednotlivé testy. Mezi hlavní sloupce patří:

- `generator`,
- `generator_class`,
- `test`,
- `sample_size`,
- `sample_iter`,
- `pass_rate`,
- `mean_p`,
- `median_p`,
- `pass_posterior_chance`,
- `avg_gen_time_ms`,
- `avg_test_time_ms`,
- `verdict`,
- `bayes_verdict`.

Soubor `single_source_summary.csv` obsahuje souhrnné výsledky pro celý generátor, například:

- počet splněných testů,
- průměrnou hodnotu `pass_rate`,
- seznam testů s výsledkem `PASS`,
- seznam testů s výsledkem `FAIL`,
- průměrný čas generování,
- průměrný čas testování.

### CSV výstupy pro `benchmark`

Režim `benchmark` vytváří zejména soubory:

```text
...-benchmark_metrics.csv
...-benchmark_class_summary.csv
```

Soubor `benchmark_metrics.csv` obsahuje výsledky pro jednotlivé kombinace generátoru, testu a velikosti vzorku.

Soubor `benchmark_class_summary.csv` obsahuje agregované výsledky podle tříd generátorů, například `PRNG`, `CSPRNG`, `TRNG` nebo kontrolní zdroje.

---

## Interpretace metrik

Výsledky obsahují p-hodnoty, ukazatele empirické úspěšnosti a další souhrnné statistiky.

### `p-value`

P-hodnota vyjadřuje, zda výsledek konkrétního testu vykazuje statisticky významnou odchylku od nulové hypotézy. Pokud je p-hodnota menší než `alpha`, daný vzorek testem neprošel.

### `pass_rate`

Metrika `pass_rate` udává podíl vzorků, které v daném testu prošly:

```text
pass_rate = počet úspěšných vzorků / celkový počet vzorků
```

Tato hodnota je hlavní metrikou pro klasický verdikt `PASS/FAIL`.

### `min_pass_rate`

Hodnota `min_pass_rate` není přímo nastavena v `config.json`. Program ji dopočítává podle metodiky NIST z hodnot `alpha` a `sample_iter`.

Představuje minimální požadovaný podíl úspěšných vzorků pro daný test.

Hlavní výsledek `PASS/FAIL` je určen porovnáním:

```text
pass_rate >= min_pass_rate
```

Pokud tato nerovnost platí, test je označen jako `PASS`. V opačném případě je označen jako `FAIL`.

### Bayesovská doplňková metrika

Program počítá také doplňkovou Bayesovskou metriku:

```text
pass_posterior_chance = P(theta >= min_pass_rate | data)
```

kde `theta` označuje neznámou skutečnou pravděpodobnost úspěšného průchodu testem a `data` představují pozorované počty úspěšných a neúspěšných vzorků.

Pro výpočet je použit beta-binomický model s Jeffreysovým priorem.

Konfigurační hodnota `bayes_pass_threshold` se používá až pro rozhodnutí, zda je Bayesovský verdikt označen jako `PASS`:

```text
pass_posterior_chance >= bayes_pass_threshold
```

Bayesovský verdikt je pouze doplňkový. Nenahrazuje hlavní rozhodnutí založené na `pass_rate` a `min_pass_rate`.

### Poznámka k názvům sloupců

Některé exportované sloupce, například `classical_randomness_chance`, `bayes_randomness_chance`, `randomness_chance` nebo `fail_chance`, slouží jako pomocné interpretační metriky nebo zpětně kompatibilní aliasy. Nemají být chápány jako důkaz absolutní náhodnosti generátoru.

Statistické testy nemohou náhodnost prokázat. Mohou pouze indikovat, zda testovaná data vykazují odchylky od zvoleného modelu.

---

## Implementované testy

- Frequency (Monobit) Test
- Runs Test
- Block Frequency Test
- Autocorrelation Test
- Spectral Test
- Linear Complexity Test
- Diehard Birthday Spacings Test
- Dieharder Byte Distribution Test

---

## Doporučený postup pro ověření projektu

1. Nainstalovat potřebné závislosti.
2. Zkontrolovat nastavení v `config.json`.
3. Spustit program pomocí `python main.py`.
4. Vybrat požadovaný režim a případně generátor.
5. Zkontrolovat nové výstupy ve složce `outputs/<nazev-generatoru>`.
6. Referenční výsledky použité v práci hledat ve složkách `outputs/<nazev-generatoru>/500*1000000`.

---

## Poznámky k interpretaci výsledků

Výsledky je nutné interpretovat v kontextu použité konfigurace, délky vzorků, počtu opakování a konkrétní implementace testů.

Hodnoty časové náročnosti se mohou mezi běhy mírně lišit v závislosti na aktuálním zatížení systému.

Výsledky statistických testů neposkytují absolutní důkaz náhodnosti. Slouží jako empirické indikátory shody testovaných dat se zvoleným modelem náhodnosti.
