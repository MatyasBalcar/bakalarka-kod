"""
Main soubor, spustitelny
"""

import csv
import datetime
import json
import math
import os
import sys
import time
import tracemalloc
from collections import defaultdict
from typing import TypedDict

import numpy as np
import scipy.special as spc
from tqdm import tqdm

from generators import (
    AlternatingGenerator,
    AmbientNoiseGenerator,
    AudioSampleBatchGenerator,
    BlumBlumShub,
    LCG,
    MersenneTwister,
    PCG64Wrapper,
    RepeatingGenerator,
    XORShift32,
)
from tests import (
    MonobitTest,
    RunsTest,
    BlockFrequencyTest,
    AutocorrelationTest,
    SpectralTest,
    LinearComplexityTest,
    DiehardBirthdaySpacingsTest,
    DieharderByteDistributionTest,
    TEST_COMPLEXITY,
    execute_with_metrics,
    evaluate_pvalues,
)
from ui import print_tests, print_generators, get_generator_with_index


def profile_generator(generator, size_bits: int):
    """Měří čas a peak paměť při generování bitového pole."""
    tracemalloc.start()
    start = time.perf_counter()
    bits = generator.generate(size_bits=size_bits)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return bits, elapsed, peak


def write_complexity_section(file, tests_to_run):
    file.write("Teoretická složitost testů:\n")
    for test_name, test_obj in tests_to_run:
        meta = TEST_COMPLEXITY.get(test_obj.__class__.__name__, {"time": "N/A", "space": "N/A"})
        file.write(f"  - {test_name}: čas {meta['time']}, paměť {meta['space']}\n")
    file.write("\n")


def make_output_paths(output_txt_path: str) -> dict:
    base = output_txt_path.rsplit(".", 1)[0]
    return {
        "single_source_metrics": f"{base}-single_source_metrics.csv",
        "single_source_summary": f"{base}-single_source_summary.csv",
        "benchmark_metrics": f"{base}-benchmark_metrics.csv",
        "benchmark_class": f"{base}-benchmark_class_summary.csv",
    }


AMBIENT_GENERATOR_NAME = "Ambient Noise Generator"
AMBIENT_AUDIO_DIR = os.path.join("inputs", "audio")
CUSTOM_INPUTS_DIR = os.path.join("inputs", "custom")


class CustomDatasetMeta(TypedDict):
    folder_path: str
    files: list[str]


def list_bin_files_in_dir(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        return []
    files = []
    for name in sorted(os.listdir(directory)):
        if not name.lower().endswith(".bin"):
            continue
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            files.append(path)
    return files


def discover_custom_dataset_generators(custom_root: str = CUSTOM_INPUTS_DIR) -> dict[str, CustomDatasetMeta]:
    if not os.path.isdir(custom_root):
        return {}

    discovered: dict[str, CustomDatasetMeta] = {}
    for folder_name in sorted(os.listdir(custom_root)):
        folder_path = os.path.join(custom_root, folder_name)
        if not os.path.isdir(folder_path):
            continue
        files = list_bin_files_in_dir(folder_path)
        if not files:
            continue
        discovered[folder_name] = {
            "folder_path": folder_path,
            "files": files,
        }
    return discovered


def list_audio_bin_files(audio_dir: str = AMBIENT_AUDIO_DIR) -> list[str]:
    return list_bin_files_in_dir(audio_dir)


def generate_audio_bin_files(
        audio_dir: str,
        sample_size_bits: int,
        count: int,
        sample_rate: int,
        whitening: str,
        hash_block_bytes: int,
) -> list[str]:
    os.makedirs(audio_dir, exist_ok=True)
    run_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    generator = AmbientNoiseGenerator(
        sample_rate=sample_rate,
        channels=1,
        block_frames=4096,
        lsb_index=1,
        whitening=whitening,
        hash_block_bytes=hash_block_bytes,
        device=None,
    )
    generated_paths = []
    try:
        for sample_index in tqdm(range(1, count + 1), desc="Nahrávání ambient datasetu", leave=False):
            bits = generator.generate(sample_size_bits)
            filename = f"ambient-{run_stamp}-{sample_index:03d}.bin"
            path = os.path.join(audio_dir, filename)
            with open(path, "wb") as output_handle:
                output_handle.write(np.packbits(bits.astype(np.uint8), bitorder="big").tobytes())
            generated_paths.append(path)
    finally:
        generator.close()

    return generated_paths


def prepare_audio_dataset(
        sample_size_bits: int,
        sample_count: int,
        sample_rate: int,
        whitening: str,
        hash_block_bytes: int,
        audio_dir: str = AMBIENT_AUDIO_DIR,
) -> list[str]:
    os.makedirs(audio_dir, exist_ok=True)
    required_bytes = (sample_size_bits + 7) // 8

    existing = list_audio_bin_files(audio_dir)
    compatible = [p for p in existing if os.path.getsize(p) >= required_bytes]

    if existing:
        if len(compatible) < sample_count:
            print(
                f"V {audio_dir} je jen {len(compatible)} kompatibilních souborů, "
                f"ale potřeba je {sample_count}."
            )
            answer = input("Přegenerovat celý audio dataset? [Y/n]: ").strip().lower()
            if answer in ("", "y", "yes"):
                for p in existing:
                    try:
                        os.remove(p)
                    except OSError:
                        pass
                return generate_audio_bin_files(
                    audio_dir=audio_dir,
                    sample_size_bits=sample_size_bits,
                    count=sample_count,
                    sample_rate=sample_rate,
                    whitening=whitening,
                    hash_block_bytes=hash_block_bytes,
                )
            raise RuntimeError("Nedostatek audio souborů pro zvolený sample_iter.")

        answer = input(
            f"V {audio_dir} už jsou audio soubory ({len(existing)}). "
            "Přepsat je novými? [y/N]: "
        ).strip().lower()
        if answer in ("y", "yes"):
            for p in existing:
                try:
                    os.remove(p)
                except OSError:
                    pass
            return generate_audio_bin_files(
                audio_dir=audio_dir,
                sample_size_bits=sample_size_bits,
                count=sample_count,
                sample_rate=sample_rate,
                whitening=whitening,
                hash_block_bytes=hash_block_bytes,
            )
        return compatible[:sample_count]

    return generate_audio_bin_files(
        audio_dir=audio_dir,
        sample_size_bits=sample_size_bits,
        count=sample_count,
        sample_rate=sample_rate,
        whitening=whitening,
        hash_block_bytes=hash_block_bytes,
    )


def prepare_audio_files_for_benchmark(
        sample_sizes: list[int],
        repeats: int,
        audio_dir: str = AMBIENT_AUDIO_DIR,
) -> list[str]:
    required_count = repeats
    available_files = list_audio_bin_files(audio_dir)

    if len(available_files) < required_count:
        raise RuntimeError(
            "Benchmark audio validation failed: "
            f"v {audio_dir} je jen {len(available_files)} souborů, "
            f"ale benchmark potřebuje {required_count}."
        )

    selected_files = available_files[:required_count]
    max_required_size = max(sample_sizes) if sample_sizes else 0

    for index, path in enumerate(selected_files, start=1):
        required_bytes = (int(max_required_size) + 7) // 8
        actual_bytes = os.path.getsize(path)
        if actual_bytes < required_bytes:
            raise RuntimeError(
                "Benchmark audio validation failed: "
                f"soubor #{index} je příliš krátký ({path}). "
                f"Potřeba >= {required_bytes} B pro {max_required_size} bitů, "
                f"nalezeno {actual_bytes} B."
            )

    return selected_files


def write_csv_rows(path: str, headers: list[str], rows: list[dict]):
    with open(path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def compute_pass_chances(
        pass_count: int,
        total_count: int,
        alpha: float,
        bayes_pass_threshold: float,
) -> dict:
    if total_count <= 0:
        return {
            "min_pass_rate": 0.0,
            "empirical_pass_rate": 0.0,
            "empirical_fail_rate": 1.0,
            "posterior_pass_threshold_probability": 0.0,
            "posterior_fail_threshold_probability": 1.0,
            "passrate_verdict": "FAIL",
            "bayes_verdict": "FAIL",
        }

    p_hat = 1.0 - alpha
    limit = 3.0 * math.sqrt((p_hat * alpha) / total_count)
    min_pass_rate = p_hat - limit

    empirical_pass_rate = float(pass_count / total_count)
    empirical_fail_rate = float(1.0 - empirical_pass_rate)

    a_post = pass_count + 0.5
    b_post = (total_count - pass_count) + 0.5
    threshold = float(np.clip(min_pass_rate, 0.0, 1.0))
    posterior_pass_threshold_probability = float(1.0 - spc.betainc(a_post, b_post, threshold))
    posterior_pass_threshold_probability = float(np.clip(posterior_pass_threshold_probability, 0.0, 1.0))
    posterior_fail_threshold_probability = float(1.0 - posterior_pass_threshold_probability)

    bayes_threshold = float(np.clip(bayes_pass_threshold, 0.0, 1.0))
    passrate_verdict = "PASS" if empirical_pass_rate >= min_pass_rate else "FAIL"
    bayes_verdict = "PASS" if posterior_pass_threshold_probability >= bayes_threshold else "FAIL"

    return {
        "min_pass_rate": float(min_pass_rate),
        "empirical_pass_rate": empirical_pass_rate,
        "empirical_fail_rate": empirical_fail_rate,
        "posterior_pass_threshold_probability": posterior_pass_threshold_probability,
        "posterior_fail_threshold_probability": posterior_fail_threshold_probability,
        "passrate_verdict": passrate_verdict,
        "bayes_verdict": bayes_verdict,
    }


def run_single_source_mode(
        generators,
        tests_to_run,
        sample_size: int,
        num_samples: int,
        alpha: float,
        bayes_pass_threshold: float,
        file,
        csv_paths: dict,
):
    single_source_rows = []
    single_source_summary_rows = []

    note = (
        "Pozn.: Hlavni verdikt PASS/FAIL je podle pass-rate vuci NIST proportion prahu. "
        "Bayesovske metriky (Jeffreys prior) jsou pouze doplnkove a exportuji se do CSV.\n\n"
    )
    print(note, end="")
    file.write(note)

    for gen_name, generator in generators.items():
        p_hat = 1.0 - alpha
        limit = 3 * math.sqrt((p_hat * alpha) / num_samples)
        min_pass_rate = p_hat - limit

        header = (
            "==================================================\n"
            f"Vícestupňové testování: {gen_name}\n"
            f"Třída generátoru: {getattr(generator, 'generator_class', 'UNKNOWN')}\n"
            f"Parametry: {num_samples} vzorků po {sample_size} bitech\n"
            f"Minimální požadovaná úspěšnost (Podle NIST) (Proportion): {min_pass_rate * 100:.2f} %\n"
            "--------------------------------------------------\n"
        )
        print(header, end="")
        file.write(header)

        pass_counts = {name: 0 for name, _ in tests_to_run}
        p_values_by_test = {name: [] for name, _ in tests_to_run}
        test_times_ms_by_test = {name: [] for name, _ in tests_to_run}
        verdict_by_test = {}
        gen_times_ms = []
        empirical_pass_rates_for_generator = []
        posterior_pass_probabilities_for_generator = []
        pass_rates_for_generator = []

        for _ in tqdm(range(num_samples), desc=f"Zpracování vzorků pro {gen_name}", leave=False):
            sample_bits, sample_elapsed_sec, _ = profile_generator(generator, size_bits=sample_size)
            gen_times_ms.append(sample_elapsed_sec * 1000.0)

            for test_name, test in tests_to_run:
                measured = execute_with_metrics(test, sample_bits, alpha=alpha)
                p_values_by_test[test_name].append(measured["p_value"])
                test_times_ms_by_test[test_name].append(measured["elapsed_sec"] * 1000.0)
                if measured["passed"]:
                    pass_counts[test_name] += 1

        for test_name in pass_counts:
            metrics = evaluate_pvalues(
                p_values=p_values_by_test[test_name],
                alpha=alpha,
                min_pass_rate=min_pass_rate,
                bayes_pass_threshold=bayes_pass_threshold,
            )
            passrate_verdict = "PASS" if metrics["pass_rate"] >= min_pass_rate else "FAIL"
            bayes_verdict = metrics["verdict"]

            gen_class = getattr(generator, "generator_class", "UNKNOWN")
            avg_gen_time_ms = float(np.mean(gen_times_ms)) if gen_times_ms else 0.0
            avg_test_time_ms = float(np.mean(test_times_ms_by_test[test_name])) if test_times_ms_by_test[
                test_name] else 0.0
            single_source_rows.append({
                "generator": gen_name,
                "generator_class": gen_class,
                "test": test_name,
                "sample_size": sample_size,
                "sample_iter": num_samples,
                "pass_rate": metrics["pass_rate"],
                "mean_p": metrics["mean_p"],
                "median_p": metrics["median_p"],
                "empirical_pass_rate": metrics["empirical_pass_rate"],
                "empirical_fail_rate": metrics["empirical_fail_rate"],
                "posterior_pass_threshold_probability": metrics["posterior_pass_threshold_probability"],
                "posterior_fail_threshold_probability": metrics["posterior_fail_threshold_probability"],
                "avg_gen_time_ms": avg_gen_time_ms,
                "avg_test_time_ms": avg_test_time_ms,
                "verdict": passrate_verdict,
                "bayes_verdict": bayes_verdict,
            })
            pass_rates_for_generator.append(metrics["pass_rate"])
            empirical_pass_rates_for_generator.append(metrics["empirical_pass_rate"])
            posterior_pass_probabilities_for_generator.append(metrics["posterior_pass_threshold_probability"])

            output_str = (
                f"  {test_name}:\n"
                f"   - Úspěšných vzorků: {pass_counts[test_name]}/{num_samples} ({metrics['pass_rate'] * 100:.2f} %)\n"
                f"   - Mean p-value: {metrics['mean_p']:.6f}, Median p-value: {metrics['median_p']:.6f}\n"
                f"   - Avg gen time: {avg_gen_time_ms:.2f} ms, Avg test time: {avg_test_time_ms:.2f} ms\n"
                f"   - Empirický podíl úspěšných vzorků: {metrics['empirical_pass_rate']:.4f}, "
                f"Empirický podíl neúspěšných: {metrics['empirical_fail_rate']:.4f}\n"
                f"   - Posteriorní pravděpodobnost splnění prahu: {metrics['posterior_pass_threshold_probability']:.4f}, "
                f"Nesplnění prahu: {metrics['posterior_fail_threshold_probability']:.4f}\n"
                f"   - Verdikt testu (pass-rate): {passrate_verdict}\n"
                f"   - Bayes verdict (doplnkove): {bayes_verdict}\n\n"
            )
            verdict_by_test[test_name] = passrate_verdict

            print(output_str, end="")
            file.write(output_str)

        pass_tests = [name for name, verdict in verdict_by_test.items() if verdict == "PASS"]
        fail_tests = [name for name, verdict in verdict_by_test.items() if verdict != "PASS"]
        pass_count = len(pass_tests)
        total_tests = len(verdict_by_test)
        pass_ratio = (pass_count / total_tests) if total_tests else 0.0

        generator_summary_text = (
            f"  => Souhrn generátoru {gen_name}: {pass_count}/{total_tests} testů PASS\n"
            f"     PASS: {', '.join(pass_tests) if pass_tests else '-'}\n"
            f"     FAIL: {', '.join(fail_tests) if fail_tests else '-'}\n\n"
        )
        print(generator_summary_text, end="")
        file.write(generator_summary_text)

        avg_gen_time_ms = float(np.mean(gen_times_ms)) if gen_times_ms else 0.0
        all_test_timings = [t for values in test_times_ms_by_test.values() for t in values]
        avg_test_time_ms_all_tests = float(np.mean(all_test_timings)) if all_test_timings else 0.0
        generator_avg_pass_rate = float(np.mean(pass_rates_for_generator)) if pass_rates_for_generator else 0.0
        avg_empirical_pass_rate = (
            float(np.mean(empirical_pass_rates_for_generator))
            if empirical_pass_rates_for_generator
            else 0.0
        )
        if empirical_pass_rates_for_generator:
            clipped_empirical = np.clip(np.array(empirical_pass_rates_for_generator, dtype=float), 1e-12, 1.0)
            geo_empirical_pass_rate = float(np.exp(np.mean(np.log(clipped_empirical))))
        else:
            geo_empirical_pass_rate = 0.0
        min_empirical_pass_rate = (
            float(np.min(empirical_pass_rates_for_generator))
            if empirical_pass_rates_for_generator
            else 0.0
        )
        empirical_fail_rate = float(1.0 - min_empirical_pass_rate)

        avg_posterior_pass_threshold_probability = (
            float(np.mean(posterior_pass_probabilities_for_generator))
            if posterior_pass_probabilities_for_generator
            else 0.0
        )
        if posterior_pass_probabilities_for_generator:
            clipped_posterior = np.clip(np.array(posterior_pass_probabilities_for_generator, dtype=float), 1e-12, 1.0)
            geo_posterior_pass_threshold_probability = float(np.exp(np.mean(np.log(clipped_posterior))))
        else:
            geo_posterior_pass_threshold_probability = 0.0
        min_posterior_pass_threshold_probability = (
            float(np.min(posterior_pass_probabilities_for_generator))
            if posterior_pass_probabilities_for_generator
            else 0.0
        )
        posterior_fail_threshold_probability = float(1.0 - min_posterior_pass_threshold_probability)
        single_source_summary_rows.append({
            "generator": gen_name,
            "generator_class": getattr(generator, "generator_class", "UNKNOWN"),
            "pass_count": pass_count,
            "total_tests": total_tests,
            "pass_ratio": pass_ratio,
            "pass_tests": "; ".join(pass_tests),
            "fail_tests": "; ".join(fail_tests),
            "avg_gen_time_ms": avg_gen_time_ms,
            "avg_test_time_ms": avg_test_time_ms_all_tests,
            "avg_pass_rate": generator_avg_pass_rate,
            "aggregation": "min_across_tests",
            "avg_empirical_pass_rate": avg_empirical_pass_rate,
            "geo_empirical_pass_rate": geo_empirical_pass_rate,
            "min_empirical_pass_rate": min_empirical_pass_rate,
            "empirical_fail_rate": empirical_fail_rate,
            "avg_posterior_pass_threshold_probability": avg_posterior_pass_threshold_probability,
            "geo_posterior_pass_threshold_probability": geo_posterior_pass_threshold_probability,
            "min_posterior_pass_threshold_probability": min_posterior_pass_threshold_probability,
            "posterior_fail_threshold_probability": posterior_fail_threshold_probability,
        })

        generation_line = (
            f"  => Průměrný čas generování pro {gen_name}: {avg_gen_time_ms:.2f} ms\n"
            f"  => Průměrný čas testu pro {gen_name}: {avg_test_time_ms_all_tests:.2f} ms\n"
            f"  => Průměrný pass-rate přes všechny testy: {generator_avg_pass_rate * 100:.2f} %\n"
            f"  => Empirický podíl úspěšných vzorků (průměr přes testy): {avg_empirical_pass_rate:.4f} (0-1)\n"
            f"  => Empirický podíl úspěšných vzorků (geometrický průměr): {geo_empirical_pass_rate:.4f} (0-1)\n"
            f"  => Empirický podíl úspěšných vzorků (minimum přes testy): {min_empirical_pass_rate:.4f} (0-1)\n"
            f"  => Empirický podíl neúspěšných vzorků: {empirical_fail_rate:.4f} (0-1)\n"
            f"  => Posteriorní pravděpodobnost splnění prahu (průměr přes testy): "
            f"{avg_posterior_pass_threshold_probability:.4f} (0-1)\n"
            f"  => Posteriorní pravděpodobnost splnění prahu (geometrický průměr): "
            f"{geo_posterior_pass_threshold_probability:.4f} (0-1)\n"
            f"  => Posteriorní pravděpodobnost splnění prahu (minimum přes testy): "
            f"{min_posterior_pass_threshold_probability:.4f} (0-1)\n"
            f"  => Posteriorní pravděpodobnost nesplnění prahu: {posterior_fail_threshold_probability:.4f} (0-1)\n\n"
        )
        print(generation_line, end="")
        file.write(generation_line)

    write_csv_rows(
        csv_paths["single_source_metrics"],
        [
            "generator", "generator_class", "test", "sample_size", "sample_iter",
            "pass_rate", "mean_p", "median_p",
            "empirical_pass_rate", "empirical_fail_rate",
            "posterior_pass_threshold_probability", "posterior_fail_threshold_probability",
            "avg_gen_time_ms", "avg_test_time_ms", "verdict", "bayes_verdict",
        ],
        single_source_rows,
    )
    write_csv_rows(
        csv_paths["single_source_summary"],
        [
            "generator", "generator_class", "pass_count", "total_tests", "pass_ratio",
            "pass_tests", "fail_tests", "avg_gen_time_ms", "avg_test_time_ms",
            "avg_pass_rate",
            "aggregation",
            "avg_empirical_pass_rate", "geo_empirical_pass_rate", "min_empirical_pass_rate", "empirical_fail_rate",
            "avg_posterior_pass_threshold_probability", "geo_posterior_pass_threshold_probability", "min_posterior_pass_threshold_probability", "posterior_fail_threshold_probability",
        ],
        single_source_summary_rows,
    )

    csv_info = (
        "CSV export:\n"
        f"- {csv_paths['single_source_metrics']}\n"
        f"- {csv_paths['single_source_summary']}\n"
    )
    print(csv_info)
    file.write(csv_info)
    print()
    file.write("\n")


def run_benchmark_mode(
        generators,
        tests_to_run,
        sample_sizes,
        repeats: int,
        alpha: float,
        bayes_pass_threshold: float,
        file,
        csv_paths: dict,
):
    file.write("Benchmark režim: čas + paměť + p-value\n")
    file.write(f"Počet opakování na kombinaci: {repeats}\n")
    file.write(f"Testované velikosti: {sample_sizes}\n\n")
    write_complexity_section(file, tests_to_run)

    stats = defaultdict(lambda: {
        "count": 0,
        "gen_time_sum": 0.0,
        "gen_peak_sum": 0,
        "test_time_sum": 0.0,
        "test_peak_sum": 0,
        "p_value_sum": 0.0,
        "pass_sum": 0,
    })
    class_stats = defaultdict(lambda: {
        "count": 0,
        "gen_time_sum": 0.0,
        "gen_peak_sum": 0,
        "test_time_sum": 0.0,
        "test_peak_sum": 0,
        "p_value_sum": 0.0,
        "pass_sum": 0,
    })

    for gen_name, generator in generators.items():
        for size_bits in sample_sizes:
            if isinstance(generator, AudioSampleBatchGenerator):
                generator.reset_samples()
            for _ in tqdm(range(repeats), desc=f"Benchmark {gen_name} @ {size_bits}", leave=False):
                sample_bits, gen_time, gen_peak = profile_generator(generator, size_bits=size_bits)

                if not isinstance(sample_bits, np.ndarray):
                    sample_bits = np.array(sample_bits, dtype=np.uint8)

                for test_name, test in tests_to_run:
                    measured = execute_with_metrics(test, sample_bits, alpha=alpha)
                    key = (gen_name, test_name, size_bits)
                    row = stats[key]
                    row["count"] += 1
                    row["gen_time_sum"] += gen_time
                    row["gen_peak_sum"] += gen_peak
                    row["test_time_sum"] += measured["elapsed_sec"]
                    row["test_peak_sum"] += measured["peak_bytes"]
                    row["p_value_sum"] += measured["p_value"]
                    row["pass_sum"] += int(measured["passed"])

                    gen_class = getattr(generator, "generator_class", "UNKNOWN")
                    class_key = (gen_class, size_bits)
                    class_row = class_stats[class_key]
                    class_row["count"] += 1
                    class_row["gen_time_sum"] += gen_time
                    class_row["gen_peak_sum"] += gen_peak
                    class_row["test_time_sum"] += measured["elapsed_sec"]
                    class_row["test_peak_sum"] += measured["peak_bytes"]
                    class_row["p_value_sum"] += measured["p_value"]
                    class_row["pass_sum"] += int(measured["passed"])

    file.write("Výsledky benchmarku (průměry):\n")
    benchmark_rows = []
    class_pass_summary = defaultdict(lambda: {
        "empirical_pass_values": [],
        "posterior_pass_values": [],
    })
    for gen_name, test_name, size_bits in sorted(
            stats.keys(),
            key=lambda key_item: (key_item[2], key_item[0], key_item[1]),
    ):
        row = stats[(gen_name, test_name, size_bits)]
        run_count = row["count"]
        avg_gen_ms = (row["gen_time_sum"] / run_count) * 1000.0
        avg_test_ms = (row["test_time_sum"] / run_count) * 1000.0
        avg_gen_kib = row["gen_peak_sum"] / run_count / 1024.0
        avg_test_kib = row["test_peak_sum"] / run_count / 1024.0
        pass_rate = row["pass_sum"] / run_count
        avg_p = row["p_value_sum"] / run_count
        pass_chances = compute_pass_chances(
            pass_count=row["pass_sum"],
            total_count=run_count,
            alpha=alpha,
            bayes_pass_threshold=bayes_pass_threshold,
        )

        line = (
            f"[{size_bits} bit] {gen_name} | {test_name}\n"
            f"  - gen:  {avg_gen_ms:.2f} ms, peak {avg_gen_kib:.2f} KiB\n"
            f"  - test: {avg_test_ms:.2f} ms, peak {avg_test_kib:.2f} KiB\n"
            f"  - avg p-value: {avg_p:.6f}, pass-rate: {pass_rate * 100:.2f} %\n\n"
        )
        print(line, end="")
        file.write(line)

        generator = generators[gen_name]
        generator_class = getattr(generator, "generator_class", "UNKNOWN")
        class_key = (generator_class, size_bits)
        class_pass_summary[class_key]["empirical_pass_values"].append(pass_chances["empirical_pass_rate"])
        class_pass_summary[class_key]["posterior_pass_values"].append(
            pass_chances["posterior_pass_threshold_probability"]
        )

        benchmark_rows.append({
            "sample_size": size_bits,
            "generator": gen_name,
            "test": test_name,
            "avg_gen_ms": avg_gen_ms,
            "avg_test_ms": avg_test_ms,
            "avg_gen_kib": avg_gen_kib,
            "avg_test_kib": avg_test_kib,
            "avg_p_value": avg_p,
            "pass_rate": pass_rate,
            "empirical_pass_rate": pass_chances["empirical_pass_rate"],
            "empirical_fail_rate": pass_chances["empirical_fail_rate"],
            "posterior_pass_threshold_probability": pass_chances["posterior_pass_threshold_probability"],
            "posterior_fail_threshold_probability": pass_chances["posterior_fail_threshold_probability"],
            "passrate_verdict": pass_chances["passrate_verdict"],
            "bayes_verdict": pass_chances["bayes_verdict"],
        })

    file.write("==================== Agregace podle tříd generátorů ====================\n")
    print("==================== Agregace podle tříd generátorů ====================")
    class_rows = []
    for gen_class, size_bits in sorted(class_stats.keys(), key=lambda key_item: (key_item[1], key_item[0])):
        row = class_stats[(gen_class, size_bits)]
        run_count = row["count"]
        avg_gen_ms = (row["gen_time_sum"] / run_count) * 1000.0
        avg_test_ms = (row["test_time_sum"] / run_count) * 1000.0
        avg_gen_kib = row["gen_peak_sum"] / run_count / 1024.0
        avg_test_kib = row["test_peak_sum"] / run_count / 1024.0
        pass_rate = row["pass_sum"] / run_count
        avg_p = row["p_value_sum"] / run_count
        pass_chances = compute_pass_chances(
            pass_count=row["pass_sum"],
            total_count=run_count,
            alpha=alpha,
            bayes_pass_threshold=bayes_pass_threshold,
        )
        empirical_pass_values = class_pass_summary[(gen_class, size_bits)]["empirical_pass_values"]
        posterior_pass_values = class_pass_summary[(gen_class, size_bits)]["posterior_pass_values"]

        avg_empirical_pass_rate = float(np.mean(empirical_pass_values)) if empirical_pass_values else 0.0
        geo_empirical_pass_rate = (
            float(np.exp(np.mean(np.log(np.clip(np.array(empirical_pass_values, dtype=float), 1e-12, 1.0)))))
            if empirical_pass_values
            else 0.0
        )
        min_empirical_pass_rate = float(np.min(empirical_pass_values)) if empirical_pass_values else 0.0

        avg_posterior_pass_threshold_probability = float(np.mean(posterior_pass_values)) if posterior_pass_values else 0.0
        geo_posterior_pass_threshold_probability = (
            float(np.exp(np.mean(np.log(np.clip(np.array(posterior_pass_values, dtype=float), 1e-12, 1.0)))))
            if posterior_pass_values
            else 0.0
        )
        min_posterior_pass_threshold_probability = float(np.min(posterior_pass_values)) if posterior_pass_values else 0.0
        line = (
            f"[{size_bits} bit] {gen_class}\n"
            f"  - gen:  {avg_gen_ms:.2f} ms, peak {avg_gen_kib:.2f} KiB\n"
            f"  - test: {avg_test_ms:.2f} ms, peak {avg_test_kib:.2f} KiB\n"
            f"  - avg p-value: {avg_p:.6f}, pass-rate: {pass_rate * 100:.2f} %\n\n"
        )
        print(line, end="")
        file.write(line)
        class_rows.append({
            "sample_size": size_bits,
            "generator_class": gen_class,
            "avg_gen_ms": avg_gen_ms,
            "avg_test_ms": avg_test_ms,
            "avg_gen_kib": avg_gen_kib,
            "avg_test_kib": avg_test_kib,
            "avg_p_value": avg_p,
            "pass_rate": pass_rate,
            "empirical_pass_rate": pass_chances["empirical_pass_rate"],
            "empirical_fail_rate": pass_chances["empirical_fail_rate"],
            "posterior_pass_threshold_probability": pass_chances["posterior_pass_threshold_probability"],
            "posterior_fail_threshold_probability": pass_chances["posterior_fail_threshold_probability"],
            "passrate_verdict": pass_chances["passrate_verdict"],
            "bayes_verdict": pass_chances["bayes_verdict"],
            "avg_empirical_pass_rate": avg_empirical_pass_rate,
            "geo_empirical_pass_rate": geo_empirical_pass_rate,
            "min_empirical_pass_rate": min_empirical_pass_rate,
            "avg_posterior_pass_threshold_probability": avg_posterior_pass_threshold_probability,
            "geo_posterior_pass_threshold_probability": geo_posterior_pass_threshold_probability,
            "min_posterior_pass_threshold_probability": min_posterior_pass_threshold_probability,
        })

    write_csv_rows(
        csv_paths["benchmark_metrics"],
        [
            "sample_size", "generator", "test", "avg_gen_ms", "avg_test_ms",
            "avg_gen_kib", "avg_test_kib", "avg_p_value", "pass_rate",
            "empirical_pass_rate", "empirical_fail_rate",
            "posterior_pass_threshold_probability", "posterior_fail_threshold_probability",
            "passrate_verdict", "bayes_verdict",
        ],
        benchmark_rows,
    )
    write_csv_rows(
        csv_paths["benchmark_class"],
        [
            "sample_size", "generator_class", "avg_gen_ms", "avg_test_ms",
            "avg_gen_kib", "avg_test_kib", "avg_p_value", "pass_rate",
            "empirical_pass_rate", "empirical_fail_rate",
            "posterior_pass_threshold_probability", "posterior_fail_threshold_probability",
            "passrate_verdict", "bayes_verdict",
            "avg_empirical_pass_rate", "geo_empirical_pass_rate", "min_empirical_pass_rate",
            "avg_posterior_pass_threshold_probability", "geo_posterior_pass_threshold_probability", "min_posterior_pass_threshold_probability",
        ],
        class_rows,
    )

    csv_info = (
        "CSV export:\n"
        f"- {csv_paths['benchmark_metrics']}\n"
        f"- {csv_paths['benchmark_class']}\n"
    )
    print(csv_info)
    file.write(csv_info)


if __name__ == "__main__":
    with open("config.json", "r", encoding="utf-8") as cfg_file:
        config = json.load(cfg_file)

    sample_size = int(config.get("sample_size", 1_000_000))
    num_samples = int(config.get("sample_iter", 500))
    output_dir = config.get("output_dir", "outputs")
    mode = config.get("mode", "single-source").lower()
    mode = {
        "single_source": "single-source",
    }.get(mode, mode)
    alpha = float(config.get("alpha", 0.01))
    bayes_pass_threshold = float(config.get("bayes_pass_threshold", 0.95))
    benchmark_sizes = config.get("benchmark_sample_sizes", [10000, 100000, 1000000])
    benchmark_repeats = int(config.get("benchmark_repeats", 3))
    ambient_sample_rate = int(config.get("ambient_sample_rate", 48000))
    ambient_whitening = str(config.get("ambient_whitening", "von-neumann+sha256"))
    ambient_hash_block_bytes = int(config.get("ambient_hash_block_bytes", 4096))
    ambient_dataset_info = None

    os.makedirs(output_dir, exist_ok=True)
    run_stamp = str(datetime.datetime.now())

    all_generators = {
        "LCG": LCG(seed=42),
        "Mersenne Twister": MersenneTwister(seed=42),
        "PCG64": PCG64Wrapper(seed=42),
        "XORShift32": XORShift32(seed=42),
        "Blum-Blum-Shub": BlumBlumShub(p=30000000091, q=40000000003, seed=123456789),
        AMBIENT_GENERATOR_NAME: None,
        "Alternating Generator": AlternatingGenerator(1),
        "Repeating Generator": RepeatingGenerator(1),
    }
    custom_generator_sources = {}
    for folder_name, metadata in discover_custom_dataset_generators().items():
        display_name = folder_name
        if display_name in all_generators:
            display_name = f"{folder_name} (custom)"
        custom_generator_sources[display_name] = metadata
        all_generators[display_name] = None

    all_tests = [
        ("Frequency (Monobit) Test", MonobitTest()),
        ("Runs Test", RunsTest()),
        ("Block Frequency Test", BlockFrequencyTest()),
        ("Autocorrelation Test", AutocorrelationTest()),
        ("Spectral Test", SpectralTest()),
        ("Linear Complexity Test", LinearComplexityTest()),
        ("Diehard Birthday Spacings Test", DiehardBirthdaySpacingsTest()),
        ("Dieharder Byte Distribution Test", DieharderByteDistributionTest()),
    ]

    print_tests(all_tests)

    generators = all_generators
    tests_to_run = all_tests
    single_source_runs = []

    try:
        if mode == "benchmark":
            generators = dict(all_generators)
            benchmark_audio_files = prepare_audio_files_for_benchmark(
                sample_sizes=benchmark_sizes,
                repeats=benchmark_repeats,
            )
            generators[AMBIENT_GENERATOR_NAME] = AudioSampleBatchGenerator(
                filepaths=benchmark_audio_files,
                strict=False,
                enforce_size_bits=True,
            )
            ambient_dataset_info = f"{AMBIENT_AUDIO_DIR} ({len(benchmark_audio_files)} files)"
            for gen_name, metadata in custom_generator_sources.items():
                generators[gen_name] = AudioSampleBatchGenerator(
                    filepaths=metadata["files"],
                    strict=False,
                    enforce_size_bits=False,
                    warn_on_short_sample=True,
                )

        if mode == "single-source":
            print_generators(all_generators)
            selected_generator_name, selected_generator = get_generator_with_index(all_generators)
            if selected_generator_name is None:
                for gen_name, gen_obj in all_generators.items():
                    run_generator = gen_obj
                    run_num_samples = num_samples
                    run_ambient_info = None

                    if gen_name == AMBIENT_GENERATOR_NAME:
                        file_paths = prepare_audio_dataset(
                            sample_size_bits=sample_size,
                            sample_count=num_samples,
                            sample_rate=ambient_sample_rate,
                            whitening=ambient_whitening,
                            hash_block_bytes=ambient_hash_block_bytes,
                        )
                        run_generator = AudioSampleBatchGenerator(filepaths=file_paths, strict=True)
                        run_num_samples = len(file_paths)
                        run_ambient_info = f"{AMBIENT_AUDIO_DIR} ({len(file_paths)} files)"
                    elif gen_name in custom_generator_sources:
                        metadata = custom_generator_sources[gen_name]
                        run_generator = AudioSampleBatchGenerator(
                            filepaths=metadata["files"],
                            strict=False,
                            enforce_size_bits=False,
                            warn_on_short_sample=True,
                        )
                        run_ambient_info = (
                            f"{metadata['folder_path']} ({len(metadata['files'])} files)"
                        )

                    single_source_runs.append({
                        "name": gen_name,
                        "generator": run_generator,
                        "num_samples": run_num_samples,
                        "ambient_info": run_ambient_info,
                    })
            else:
                run_num_samples = num_samples
                run_ambient_info = None
                if selected_generator_name == AMBIENT_GENERATOR_NAME:
                    file_paths = prepare_audio_dataset(
                        sample_size_bits=sample_size,
                        sample_count=num_samples,
                        sample_rate=ambient_sample_rate,
                        whitening=ambient_whitening,
                        hash_block_bytes=ambient_hash_block_bytes,
                    )
                    selected_generator = AudioSampleBatchGenerator(filepaths=file_paths, strict=True)
                    run_num_samples = len(file_paths)
                    run_ambient_info = f"{AMBIENT_AUDIO_DIR} ({len(file_paths)} files)"
                elif selected_generator_name in custom_generator_sources:
                    metadata = custom_generator_sources[selected_generator_name]
                    selected_generator = AudioSampleBatchGenerator(
                        filepaths=metadata["files"],
                        strict=False,
                        enforce_size_bits=False,
                        warn_on_short_sample=True,
                    )
                    run_ambient_info = (
                        f"{metadata['folder_path']} ({len(metadata['files'])} files)"
                    )

                single_source_runs.append({
                    "name": selected_generator_name,
                    "generator": selected_generator,
                    "num_samples": run_num_samples,
                    "ambient_info": run_ambient_info,
                })
    except Exception as exc:
        print(exc)
        sys.exit(1)

    if mode == "benchmark":
        mode_output_dir = os.path.join(output_dir, "benchmarks")
        os.makedirs(mode_output_dir, exist_ok=True)
        output_file = os.path.join(mode_output_dir, f"output_{mode}-{run_stamp}.txt")
        csv_paths = make_output_paths(output_file)

        with open(output_file, "w", encoding="utf-8") as file:
            file.write(f"Mode: {mode}\n")
            file.write(
                "Ambient noise config: "
                f"whitening={ambient_whitening}, "
                f"sample_rate={ambient_sample_rate}, hash_block_bytes={ambient_hash_block_bytes}\n"
            )
            if ambient_dataset_info:
                file.write(f"Ambient audio dataset: {ambient_dataset_info}\n")
            file.write("\n")

            run_benchmark_mode(
                generators=generators,
                tests_to_run=tests_to_run,
                sample_sizes=benchmark_sizes,
                repeats=benchmark_repeats,
                alpha=alpha,
                bayes_pass_threshold=bayes_pass_threshold,
                file=file,
                csv_paths=csv_paths,
            )

        print(f"\nVýsledky byly úspěšně uloženy do: {output_file}")
    elif mode == "single-source":
        if not single_source_runs:
            print("Single-source režim nemá žádný generátor ke spuštění.")
            sys.exit(1)

        output_files = []
        for run_cfg in single_source_runs:
            gen_name = run_cfg["name"]
            mode_output_dir = os.path.join(output_dir, gen_name)
            os.makedirs(mode_output_dir, exist_ok=True)

            per_run_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

            output_file = os.path.join(mode_output_dir, f"output_{mode}-{per_run_stamp}.txt")
            csv_paths = make_output_paths(output_file)

            with open(output_file, "w", encoding="utf-8") as file:
                file.write(f"Mode: {mode}\n")
                file.write(
                    "Ambient noise config: "
                    f"whitening={ambient_whitening}, "
                    f"sample_rate={ambient_sample_rate}, hash_block_bytes={ambient_hash_block_bytes}\n"
                )
                if run_cfg["ambient_info"]:
                    file.write(f"Ambient audio dataset: {run_cfg['ambient_info']}\n")
                file.write("\n")

                write_complexity_section(file, tests_to_run)
                run_single_source_mode(
                    generators={gen_name: run_cfg["generator"]},
                    tests_to_run=tests_to_run,
                    sample_size=sample_size,
                    num_samples=run_cfg["num_samples"],
                    alpha=alpha,
                    bayes_pass_threshold=bayes_pass_threshold,
                    file=file,
                    csv_paths=csv_paths,
                )

            output_files.append(output_file)

        print("\nVýsledky byly úspěšně uloženy do:")
        for path in output_files:
            print(f"- {path}")
    else:
        print(f"Neznámý režim: {mode}. Použij 'benchmark' nebo 'single-source'.")
        sys.exit(1)
