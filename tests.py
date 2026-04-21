"""
Tento modul obsahuje testy
"""

import time
import tracemalloc
from abc import ABC, abstractmethod

import numpy as np
import scipy.special as spc
from numba import njit
from tqdm import tqdm

TEST_COMPLEXITY = {
    "MonobitTest": {"time": "O(n)", "space": "O(n)"},
    "RunsTest": {"time": "O(n)", "space": "O(1)"},
    "BlockFrequencyTest": {"time": "O(n)", "space": "O(n)"},
    "AutocorrelationTest": {"time": "O(n)", "space": "O(1)"},
    "SpectralTest": {"time": "O(n log n)", "space": "O(n)"},
    "LinearComplexityTest": {"time": "O(n * m)", "space": "O(m)"},
    "DiehardBirthdaySpacingsTest": {"time": "O(n log n)", "space": "O(n)"},
    "DieharderByteDistributionTest": {"time": "O(n)", "space": "O(1)"},
}


def execute_with_metrics(test: "TestStrategy", bits: np.ndarray, alpha: float = 0.01) -> dict:
    """Vrátí p-value + metriky výkonu pro jeden běh testu."""
    tracemalloc.start()
    start = time.perf_counter()
    p_value = test.execute(bits)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "p_value": float(p_value),
        "elapsed_sec": elapsed,
        "peak_bytes": int(peak),
        "passed": bool(p_value >= alpha),
    }


def evaluate_pvalues(
    p_values: list[float],
    alpha: float,
    min_pass_rate: float,
    bayes_pass_threshold: float = 0.95,
) -> dict:
    """Agreguje kvalitu testu nad sadou p-hodnot pro jeden generátor."""
    if not p_values:
        return {
            "pass_rate": 0.0,
            "mean_p": 0.0,
            "median_p": 0.0,
            "stability_score": 0.0,
            "final_score": 0.0,
            "pass_posterior_chance": 0.0,
            "classical_randomness_chance": 0.0,
            "classical_fail_chance": 1.0,
            "bayes_randomness_chance": 0.0,
            "bayes_fail_chance": 1.0,
            "randomness_chance": 0.0,
            "fail_chance": 1.0,
            "verdict": "FAIL",
        }

    arr = np.array(p_values, dtype=float)
    pass_rate = float(np.mean(arr >= alpha))
    mean_p = float(np.mean(arr))
    median_p = float(np.median(arr))

    # Uniformni p-hodnoty maji sigma ~= 0.288675; cim mensi odchylka, tim lepsi stabilita.
    sigma_u01 = 1.0 / np.sqrt(12.0)
    stability_score = float(max(0.0, min(1.0, 1.0 - (np.std(arr) / sigma_u01))))

    final_score = float(0.6 * pass_rate + 0.25 * mean_p + 0.15 * stability_score)

    # 1) Bayesovsky: posteriorni pravdepodobnost, ze skutecna uspesnost testu
    # je alespon min_pass_rate pri modelu Bernoulli(pruchod/nepruchod).
    # Prior: Jeffreys Beta(0.5, 0.5), posterior: Beta(k+0.5, n-k+0.5).
    n = arr.size
    k = int(np.sum(arr >= alpha))
    a_post = k + 0.5
    b_post = (n - k) + 0.5
    threshold = float(np.clip(min_pass_rate, 0.0, 1.0))
    pass_posterior_chance = float(1.0 - spc.betainc(a_post, b_post, threshold))
    pass_posterior_chance = float(np.clip(pass_posterior_chance, 0.0, 1.0))

    classical_randomness_chance = pass_rate
    classical_fail_chance = float(1.0 - classical_randomness_chance)

    bayes_randomness_chance = pass_posterior_chance
    bayes_fail_chance = float(1.0 - bayes_randomness_chance)

    # Backward-compatible aliases keep existing CSV/report consumers functional.
    randomness_chance = bayes_randomness_chance
    fail_chance = bayes_fail_chance
    bayes_threshold = float(np.clip(bayes_pass_threshold, 0.0, 1.0))
    verdict = "PASS" if bayes_randomness_chance >= bayes_threshold else "FAIL"

    return {
        "pass_rate": pass_rate,
        "mean_p": mean_p,
        "median_p": median_p,
        "stability_score": stability_score,
        "final_score": final_score,
        "pass_posterior_chance": pass_posterior_chance,
        "classical_randomness_chance": classical_randomness_chance,
        "classical_fail_chance": classical_fail_chance,
        "bayes_randomness_chance": bayes_randomness_chance,
        "bayes_fail_chance": bayes_fail_chance,
        "randomness_chance": randomness_chance,
        "fail_chance": fail_chance,
        "verdict": verdict,
    }


@njit
def fast_berlekamp_massey(block: np.ndarray) -> int:
    """njit pro rychlost"""
    n = len(block)
    b = np.zeros(n, dtype=np.int32)
    c = np.zeros(n, dtype=np.int32)
    b[0] = 1
    c[0] = 1
    l = 0
    m = -1

    for i in range(n):
        d = 0
        for j in range(l + 1):
            d ^= c[j] * block[i - j]
        if d == 1:
            t = np.copy(c)
            shift = i - m
            for j in range(n - shift):
                c[j + shift] ^= b[j]
            if 2 * l <= i:
                l = i + 1 - l
                m = i
                b = t
    return l


class TestStrategy(ABC):
    """
    Obecna trida pro testy, vsechny implementuji execute
    """

    @abstractmethod
    def execute(self, bits: np.ndarray) -> float:
        pass


# ============================
# * NIST
# ============================


class MonobitTest(TestStrategy):
    """Frequency (Monobit) test.

    Ověřuje globální vyváženost nul a jedniček v celé sekvenci.
    Testuje nulovou hypotézu, že bity jsou nezávislé a P(1)=0.5.

    Návratová hodnota:
    - p-hodnota v intervalu <0,1>; nízká hodnota znamená podezření na bias.
    """

    def execute(self, bits: np.ndarray) -> float:
        n = len(bits)
        x = 2 * bits.astype(int) - 1
        s_obs = np.abs(np.sum(x)) / np.sqrt(n)
        return spc.erfc(s_obs / np.sqrt(2))


class RunsTest(TestStrategy):
    """Runs test (test běhů).

    Sleduje, zda počet přechodů 0->1 a 1->0 odpovídá náhodné sekvenci.
    Nejprve kontroluje předpoklad testu: podíl jedniček musí být blízko 0.5.

    Návratová hodnota:
    - p-hodnota; při porušení předpokladu vyváženosti vrací 0.0.
    """

    def execute(self, bits: np.ndarray) -> float:
        n = len(bits)
        pi = np.sum(bits) / n
        tau = 2.0 / np.sqrt(n)

        if abs(pi - 0.5) >= tau:
            return 0.0000

        v_n_obs = np.sum(bits[:-1] != bits[1:]) + 1
        num = abs(v_n_obs - 2 * n * pi * (1 - pi))
        den = 2 * np.sqrt(2 * n) * pi * (1 - pi)
        return spc.erfc(num / den)


class BlockFrequencyTest(TestStrategy):
    """Block Frequency test.

    Dělí sekvenci na bloky stejné délky a v každém bloku hodnotí podíl jedniček.
    Oproti Monobit testu je citlivější na lokální odchylky v částech sekvence.

    Návratová hodnota:
    - p-hodnota z chí-kvadrát statistiky,
    - 0.0 pokud je sekvence kratší než zvolený block_size.
    """

    def execute(self, bits: np.ndarray, block_size: int = 128) -> float:
        n = len(bits)
        if n < block_size:
            return 0.0

        n_blocks = n // block_size
        blocks = np.reshape(bits[:n_blocks * block_size], (n_blocks, block_size))
        pi = np.mean(blocks, axis=1)
        chi_squared = 4.0 * block_size * np.sum((pi - 0.5) ** 2)
        return spc.gammaincc(n_blocks / 2.0, chi_squared / 2.0)


class AutocorrelationTest(TestStrategy):
    """Autocorrelation test.

    Porovnává sekvenci s její verzí posunutou o lag d a měří počet neshod.
    Ověřuje, zda mezi bity na vzdálenost d nevzniká systematická závislost.

    Návratová hodnota:
    - p-hodnota z normalizovaného z-score,
    - 0.0 pokud je vstup kratší nebo stejně dlouhý jako lag d.
    """

    def execute(self, bits: np.ndarray, d: int = 1) -> float:
        n = len(bits)
        if n <= d:
            return 0.0

        disagreements = np.sum(bits[: n - d] != bits[d:])
        expected = (n - d) / 2.0
        variance = (n - d) / 4.0
        z = (disagreements - expected) / np.sqrt(variance)
        return spc.erfc(abs(z) / np.sqrt(2.0))


class SpectralTest(TestStrategy):
    """Discrete Fourier Transform (Spectral) test.

    Převádí bity na hodnoty {-1,+1}, počítá FFT a analyzuje amplitudové spektrum.
    Hledá periodické vzory, které by v náhodné sekvenci neměly být výrazné.

    Návratová hodnota:
    - p-hodnota; nízká hodnota značí odchylku od očekávaného spektrálního chování.
    """

    def execute(self, bits: np.ndarray) -> float:
        n = len(bits)
        x = 2 * bits.astype(int) - 1
        s = np.fft.fft(x)
        modulus = np.abs(s[0:n // 2])

        t_95 = np.sqrt(-n * np.log(0.05))
        n_1 = np.sum(modulus < t_95)
        n_0 = 0.95 * (n / 2)
        d = (n_1 - n_0) / np.sqrt(n * 0.95 * 0.05 / 4)
        return spc.erfc(abs(d) / np.sqrt(2))


class LinearComplexityTest(TestStrategy):
    """Linear Complexity test.

    Odhaduje lineární složitost bloků sekvence pomocí Berlekamp-Massey algoritmu.
    Testuje, zda rozdělení složitostí odpovídá očekávání pro náhodný zdroj.

    Návratová hodnota:
    - p-hodnota z chí-kvadrát statistiky nad kategoriemi složitosti,
    - 0.0 pokud je vstup kratší než velikost bloku m.
    """

    def execute(self, bits: np.ndarray, m: int = 500) -> float:
        n = len(bits)
        if n < m:
            return 0.0

        k = 6
        n_blocks = n // m
        blocks = np.reshape(bits[:n_blocks * m], (n_blocks, m))

        mu = m / 2.0 + (9 + (-1) ** (m + 1)) / 36.0 - (m / 3.0 + 2.0 / 9.0) / (2 ** m)
        v = np.zeros(k + 1, dtype=int)

        for block in tqdm(blocks, desc="Počítání Linear Complexity", leave=False):
            l = fast_berlekamp_massey(block)
            t = (-1) ** m * (l - mu) + 2.0 / 9.0

            if t <= -2.5:
                v[0] += 1
            elif t <= -1.5:
                v[1] += 1
            elif t <= -0.5:
                v[2] += 1
            elif t <= 0.5:
                v[3] += 1
            elif t <= 1.5:
                v[4] += 1
            elif t <= 2.5:
                v[5] += 1
            else:
                v[6] += 1

        pi = np.array([0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833])
        chi_squared = np.sum(((v - n_blocks * pi) ** 2) / (n_blocks * pi))

        return spc.gammaincc(k / 2.0, chi_squared / 2.0)


# ============================
# * Diehard / Dieharder pouze obdobne testy, neni to original
# ============================


class DiehardBirthdaySpacingsTest(TestStrategy):
    """Diehard-inspired Birthday Spacings test.

    Bloky bitů převádí na čísla ("narozeniny"), seřadí je a analyzuje kolize mezer.
    Počet kolizí porovnává s Poissonovým očekáváním pro náhodný výběr.

    Návratová hodnota:
    - p-hodnota z normalizovaného rozdílu oproti očekávání,
    - 0.0 pokud není dostatek dat pro zadané parametry.
    """

    def execute(self, bits: np.ndarray, n_samples: int = 512, bits_per_sample: int = 24) -> float:
        required = n_samples * bits_per_sample
        if len(bits) < required:
            return 0.0

        # Převedeme bloky bitů na čísla v intervalu [0, 2^bits_per_sample).
        raw = bits[:required].reshape(n_samples, bits_per_sample)
        weights = (1 << np.arange(bits_per_sample - 1, -1, -1, dtype=np.int64))
        birthdays = (raw.astype(np.int64) * weights).sum(axis=1)

        birthdays.sort()
        spacings = np.diff(birthdays)
        if spacings.size == 0:
            return 0.0

        _, counts = np.unique(spacings, return_counts=True)
        collisions = np.sum(np.maximum(counts - 1, 0))

        m = float(2 ** bits_per_sample)
        lam = (n_samples ** 3) / (4.0 * m)
        if lam <= 0:
            return 0.0

        z = (collisions - lam) / np.sqrt(lam)
        return float(spc.erfc(abs(z) / np.sqrt(2.0)))


class DieharderByteDistributionTest(TestStrategy):
    """Dieharder-inspired byte distribution test.

    Převádí bitovou sekvenci na bajty (0..255) a testuje uniformitu histogramu.
    Ověřuje, zda některé bajtové hodnoty nejsou zastoupené systematicky častěji.

    Návratová hodnota:
    - p-hodnota z chí-kvadrát testu,
    - 0.0 pokud je k dispozici méně než 256 bajtů.
    """

    def execute(self, bits: np.ndarray) -> float:
        n = len(bits)
        n_bytes = n // 8
        if n_bytes < 256:
            return 0.0

        byte_bits = bits[:n_bytes * 8].reshape(n_bytes, 8)
        weights = (1 << np.arange(7, -1, -1, dtype=np.int64))
        values = (byte_bits.astype(np.int64) * weights).sum(axis=1)

        obs = np.bincount(values, minlength=256)
        expected = n_bytes / 256.0
        chi_squared = np.sum((obs - expected) ** 2 / expected)

        return float(spc.gammaincc((256 - 1) / 2.0, chi_squared / 2.0))

