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


def execute_with_metrics(test_strategy: "TestStrategy", bits: np.ndarray, alpha: float = 0.01) -> dict:
    """Vrátí p-value + metriky výkonu pro jeden běh testu."""
    tracemalloc.start()
    start = time.perf_counter()
    p_value = test_strategy.execute(bits)
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

    p_value_array = np.array(p_values, dtype=float)
    pass_rate = float(np.mean(p_value_array >= alpha))
    mean_p = float(np.mean(p_value_array))
    median_p = float(np.median(p_value_array))

    sigma_u01 = 1.0 / np.sqrt(12.0)
    stability_score = float(max(0.0, min(1.0, 1.0 - (np.std(p_value_array) / sigma_u01))))

    final_score = float(0.6 * pass_rate + 0.25 * mean_p + 0.15 * stability_score)

    sample_count = p_value_array.size
    passed_count = int(np.sum(p_value_array >= alpha))
    alpha_posterior = passed_count + 0.5
    beta_posterior = (sample_count - passed_count) + 0.5
    threshold = float(np.clip(min_pass_rate, 0.0, 1.0))
    pass_posterior_chance = float(1.0 - spc.betainc(alpha_posterior, beta_posterior, threshold))
    pass_posterior_chance = float(np.clip(pass_posterior_chance, 0.0, 1.0))

    classical_randomness_chance = pass_rate
    classical_fail_chance = float(1.0 - classical_randomness_chance)

    bayes_randomness_chance = pass_posterior_chance
    bayes_fail_chance = float(1.0 - bayes_randomness_chance)

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
    sequence_length = len(block)
    previous_connection = np.zeros(sequence_length, dtype=np.int32)
    connection = np.zeros(sequence_length, dtype=np.int32)
    previous_connection[0] = 1
    connection[0] = 1
    linear_complexity = 0
    last_update_index = -1

    for bit_index in range(sequence_length):
        discrepancy = 0
        for coefficient_index in range(linear_complexity + 1):
            discrepancy ^= connection[coefficient_index] * block[bit_index - coefficient_index]
        if discrepancy == 1:
            previous_connection_snapshot = np.copy(connection)
            shift = bit_index - last_update_index
            for coefficient_index in range(sequence_length - shift):
                connection[coefficient_index + shift] ^= previous_connection[coefficient_index]
            if 2 * linear_complexity <= bit_index:
                linear_complexity = bit_index + 1 - linear_complexity
                last_update_index = bit_index
                previous_connection = previous_connection_snapshot
    return linear_complexity


class TestStrategy(ABC):
    """
    Obecna trida pro testy, vsechny implementuji execute
    """

    @abstractmethod
    def execute(self, bits: np.ndarray) -> float:
        pass


class MonobitTest(TestStrategy):
    """Frequency (Monobit) test.

    Ověřuje globální vyváženost nul a jedniček v celé sekvenci.
    Testuje nulovou hypotézu, že bity jsou nezávislé a P(1)=0.5.

    Návratová hodnota:
    - p-hodnota v intervalu <0,1>; nízká hodnota znamená podezření na bias.
    """

    def execute(self, bits: np.ndarray) -> float:
        sequence_length = len(bits)
        mapped_bits = 2 * bits.astype(int) - 1
        observed_statistic = np.abs(np.sum(mapped_bits)) / np.sqrt(sequence_length)
        return spc.erfc(observed_statistic / np.sqrt(2))


class RunsTest(TestStrategy):
    """Runs test (test běhů).

    Sleduje, zda počet přechodů 0->1 a 1->0 odpovídá náhodné sekvenci.
    Nejprve kontroluje předpoklad testu: podíl jedniček musí být blízko 0.5.

    Návratová hodnota:
    - p-hodnota; při porušení předpokladu vyváženosti vrací 0.0.
    """

    def execute(self, bits: np.ndarray) -> float:
        sequence_length = len(bits)
        one_ratio = np.sum(bits) / sequence_length
        balance_threshold = 2.0 / np.sqrt(sequence_length)

        if abs(one_ratio - 0.5) >= balance_threshold:
            return 0.0000

        observed_run_count = np.sum(bits[:-1] != bits[1:]) + 1
        numerator = abs(observed_run_count - 2 * sequence_length * one_ratio * (1 - one_ratio))
        denominator = 2 * np.sqrt(2 * sequence_length) * one_ratio * (1 - one_ratio)
        return spc.erfc(numerator / denominator)


class BlockFrequencyTest(TestStrategy):
    """Block Frequency test.

    Dělí sekvenci na bloky stejné délky a v každém bloku hodnotí podíl jedniček.
    Oproti Monobit testu je citlivější na lokální odchylky v částech sekvence.

    Návratová hodnota:
    - p-hodnota z chí-kvadrát statistiky,
    - 0.0 pokud je sekvence kratší než zvolený block_size.
    """

    def execute(self, bits: np.ndarray, block_size: int = 128) -> float:
        sequence_length = len(bits)
        if sequence_length < block_size:
            return 0.0

        block_count = sequence_length // block_size
        blocks = np.reshape(bits[:block_count * block_size], (block_count, block_size))
        block_one_ratios = np.mean(blocks, axis=1)
        chi_squared = 4.0 * block_size * np.sum((block_one_ratios - 0.5) ** 2)
        return spc.gammaincc(block_count / 2.0, chi_squared / 2.0)


class AutocorrelationTest(TestStrategy):
    """Autocorrelation test.

    Porovnává sekvenci s její verzí posunutou o lag d a měří počet neshod.
    Ověřuje, zda mezi bity na vzdálenost d nevzniká systematická závislost.

    Návratová hodnota:
    - p-hodnota z normalizovaného z-score,
    - 0.0 pokud je vstup kratší nebo stejně dlouhý jako lag d.
    """

    def execute(self, bits: np.ndarray, lag: int = 1) -> float:
        sequence_length = len(bits)
        if sequence_length <= lag:
            return 0.0

        disagreement_count = np.sum(bits[: sequence_length - lag] != bits[lag:])
        expected_disagreements = (sequence_length - lag) / 2.0
        variance = (sequence_length - lag) / 4.0
        z_score = (disagreement_count - expected_disagreements) / np.sqrt(variance)
        return spc.erfc(abs(z_score) / np.sqrt(2.0))


class SpectralTest(TestStrategy):
    """Discrete Fourier Transform (Spectral) test.

    Převádí bity na hodnoty {-1,+1}, počítá FFT a analyzuje amplitudové spektrum.
    Hledá periodické vzory, které by v náhodné sekvenci neměly být výrazné.

    Návratová hodnota:
    - p-hodnota; nízká hodnota značí odchylku od očekávaného spektrálního chování.
    """

    def execute(self, bits: np.ndarray) -> float:
        sequence_length = len(bits)
        mapped_bits = 2 * bits.astype(int) - 1
        spectrum = np.fft.fft(mapped_bits)
        modulus = np.abs(spectrum[0:sequence_length // 2])

        threshold_95 = np.sqrt(-sequence_length * np.log(0.05))
        peaks_below_threshold = np.sum(modulus < threshold_95)
        expected_peaks_below_threshold = 0.95 * (sequence_length / 2)
        normalized_difference = (
                (peaks_below_threshold - expected_peaks_below_threshold)
                / np.sqrt(sequence_length * 0.95 * 0.05 / 4)
        )
        return spc.erfc(abs(normalized_difference) / np.sqrt(2))


class LinearComplexityTest(TestStrategy):
    """Linear Complexity test.

    Odhaduje lineární složitost bloků sekvence pomocí Berlekamp-Massey algoritmu.
    Testuje, zda rozdělení složitostí odpovídá očekávání pro náhodný zdroj.

    Návratová hodnota:
    - p-hodnota z chí-kvadrát statistiky nad kategoriemi složitosti,
    - 0.0 pokud je vstup kratší než velikost bloku m.
    """

    def execute(self, bits: np.ndarray, block_size: int = 500) -> float:
        sequence_length = len(bits)
        if sequence_length < block_size:
            return 0.0

        category_count = 6
        block_count = sequence_length // block_size
        blocks = np.reshape(bits[:block_count * block_size], (block_count, block_size))

        expected_mean = (
                block_size / 2.0
                + (9 + (-1) ** (block_size + 1)) / 36.0
                - (block_size / 3.0 + 2.0 / 9.0) / (2 ** block_size)
        )
        category_histogram = np.zeros(category_count + 1, dtype=int)

        for block in tqdm(blocks, desc="Počítání Linear Complexity", leave=False):
            linear_complexity = fast_berlekamp_massey(block)
            centered_value = (-1) ** block_size * (linear_complexity - expected_mean) + 2.0 / 9.0

            if centered_value <= -2.5:
                category_histogram[0] += 1
            elif centered_value <= -1.5:
                category_histogram[1] += 1
            elif centered_value <= -0.5:
                category_histogram[2] += 1
            elif centered_value <= 0.5:
                category_histogram[3] += 1
            elif centered_value <= 1.5:
                category_histogram[4] += 1
            elif centered_value <= 2.5:
                category_histogram[5] += 1
            else:
                category_histogram[6] += 1

        category_probabilities = np.array([0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833])
        chi_squared = np.sum(
            ((category_histogram - block_count * category_probabilities) ** 2)
            / (block_count * category_probabilities)
        )

        return spc.gammaincc(category_count / 2.0, chi_squared / 2.0)


class DiehardBirthdaySpacingsTest(TestStrategy):
    """Diehard-inspired Birthday Spacings test.

    Bloky bitů převádí na čísla ("narozeniny"), seřadí je a analyzuje kolize mezer.
    Počet kolizí porovnává s Poissonovým očekáváním pro náhodný výběr.

    Návratová hodnota:
    - p-hodnota z normalizovaného rozdílu oproti očekávání,
    - 0.0 pokud není dostatek dat pro zadané parametry.
    """

    def execute(self, bits: np.ndarray, n_samples: int = 512, bits_per_sample: int = 24) -> float:
        required_bits = n_samples * bits_per_sample
        if len(bits) < required_bits:
            return 0.0

        sample_bit_matrix = bits[:required_bits].reshape(n_samples, bits_per_sample)
        weights = (1 << np.arange(bits_per_sample - 1, -1, -1, dtype=np.int64))
        birthdays = (sample_bit_matrix.astype(np.int64) * weights).sum(axis=1)

        birthdays.sort()
        spacings = np.diff(birthdays)
        if spacings.size == 0:
            return 0.0

        _, counts = np.unique(spacings, return_counts=True)
        collisions = np.sum(np.maximum(counts - 1, 0))

        sample_space_size = float(2 ** bits_per_sample)
        expected_collision_count = (n_samples ** 3) / (4.0 * sample_space_size)
        if expected_collision_count <= 0:
            return 0.0

        z_score = (collisions - expected_collision_count) / np.sqrt(expected_collision_count)
        return float(spc.erfc(abs(z_score) / np.sqrt(2.0)))


class DieharderByteDistributionTest(TestStrategy):
    """Dieharder-inspired byte distribution test.

    Převádí bitovou sekvenci na bajty (0..255) a testuje uniformitu histogramu.
    Ověřuje, zda některé bajtové hodnoty nejsou zastoupené systematicky častěji.

    Návratová hodnota:
    - p-hodnota z chí-kvadrát testu,
    - 0.0 pokud je k dispozici méně než 256 bajtů.
    """

    def execute(self, bits: np.ndarray) -> float:
        sequence_length = len(bits)
        byte_count = sequence_length // 8
        if byte_count < 256:
            return 0.0

        byte_bits = bits[:byte_count * 8].reshape(byte_count, 8)
        weights = (1 << np.arange(7, -1, -1, dtype=np.int64))
        values = (byte_bits.astype(np.int64) * weights).sum(axis=1)

        observed_counts = np.bincount(values, minlength=256)
        expected_count_per_value = byte_count / 256.0
        chi_squared = np.sum((observed_counts - expected_count_per_value) ** 2 / expected_count_per_value)

        return float(spc.gammaincc((256 - 1) / 2.0, chi_squared / 2.0))
