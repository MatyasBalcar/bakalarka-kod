import csv
import datetime
import json
import math
import os
import sys
import time
import tracemalloc
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from generators import (
    LCG,
    MersenneTwister,
    PCG64Wrapper,
    XORShift32,
    BlumBlumShub, RepeatingGenerator, AlternatingGenerator,
    AmbientNoiseGenerator,
    AudioSampleBatchGenerator,
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

"""
TODO
graph the results
table the results
"""


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


def make_output_paths(output_txt_path: str, mode: str) -> dict:
    base = output_txt_path.rsplit(".", 1)[0]
    return {
        "cherry_pick_metrics": f"{base}-cherry_pick_metrics.csv",
        "cherry_pick_summary": f"{base}-cherry_pick_summary.csv",
        "benchmark_metrics": f"{base}-benchmark_metrics.csv",
        "benchmark_class": f"{base}-benchmark_class_summary.csv",
    }


AMBIENT_GENERATOR_NAME = "Ambient Noise Generator"
AMBIENT_AUDIO_DIR = os.path.join("inputs", "audio")


def list_audio_bin_files(audio_dir: str = AMBIENT_AUDIO_DIR) -> list[str]:
    if not os.path.isdir(audio_dir):
        return []
    files = []
    for name in sorted(os.listdir(audio_dir)):
        if not name.lower().endswith(".bin"):
            continue
        path = os.path.join(audio_dir, name)
        if os.path.isfile(path):
            files.append(path)
    return files


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
    result = []
    try:
        for i in tqdm(range(1, count + 1), desc="Nahrávání ambient datasetu", leave=False):
            bits = generator.generate(sample_size_bits)
            filename = f"ambient-{run_stamp}-{i:03d}.bin"
            path = os.path.join(audio_dir, filename)
            with open(path, "wb") as f:
                f.write(np.packbits(bits.astype(np.uint8), bitorder="big").tobytes())
            result.append(path)
    finally:
        generator.close()

    return result


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
    # Benchmark spotřebuje jeden soubor na každou kombinaci velikost x opakování.
    required_count = len(sample_sizes) * repeats
    available_files = list_audio_bin_files(audio_dir)

    if len(available_files) < required_count:
        raise RuntimeError(
            "Benchmark audio validation failed: "
            f"v {audio_dir} je jen {len(available_files)} souborů, "
            f"ale benchmark potřebuje {required_count}."
        )

    selected_files = available_files[:required_count]
    required_sizes = [size_bits for size_bits in sample_sizes for _ in range(repeats)]

    for index, (path, size_bits) in enumerate(zip(selected_files, required_sizes), start=1):
        required_bytes = (int(size_bits) + 7) // 8
        actual_bytes = os.path.getsize(path)
        if actual_bytes < required_bytes:
            raise RuntimeError(
                "Benchmark audio validation failed: "
                f"soubor #{index} je příliš krátký ({path}). "
                f"Potřeba >= {required_bytes} B pro {size_bits} bitů, "
                f"nalezeno {actual_bytes} B."
            )

    return selected_files


def write_csv_rows(path: str, headers: list[str], rows: list[dict]):
    with open(path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def run_cherry_pick_mode(
        generators,
        tests_to_run,
        sample_size: int,
        num_samples: int,
        alpha: float,
        bayes_pass_threshold: float,
        file,
        csv_paths: dict,
):
    cherry_pick_rows = []
    cherry_pick_summary_rows = []

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
        final_scores_for_generator = []
        classical_randomness_chances_for_generator = []
        bayes_randomness_chances_for_generator = []
        pass_rates_for_generator = []

        for _ in tqdm(range(num_samples), desc=f"Zpracování vzorků pro {gen_name}", leave=False):
            sample_start = time.perf_counter()
            sample_bits = generator.generate(size_bits=sample_size)
            sample_elapsed_ms = (time.perf_counter() - sample_start) * 1000.0
            gen_times_ms.append(sample_elapsed_ms)

            for test_name, test in tests_to_run:
                test_start = time.perf_counter()
                p_value = test.execute(sample_bits)
                test_elapsed_ms = (time.perf_counter() - test_start) * 1000.0
                p_values_by_test[test_name].append(float(p_value))
                test_times_ms_by_test[test_name].append(test_elapsed_ms)
                if p_value >= alpha:
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
            cherry_pick_rows.append({
                "generator": gen_name,
                "generator_class": gen_class,
                "test": test_name,
                "sample_size": sample_size,
                "sample_iter": num_samples,
                "pass_rate": metrics["pass_rate"],
                "mean_p": metrics["mean_p"],
                "median_p": metrics["median_p"],
                "stability_score": metrics["stability_score"],
                "final_score": metrics["final_score"],
                "pass_posterior_chance": metrics["pass_posterior_chance"],
                "classical_randomness_chance": metrics["classical_randomness_chance"],
                "classical_fail_chance": metrics["classical_fail_chance"],
                "bayes_randomness_chance": metrics["bayes_randomness_chance"],
                "bayes_fail_chance": metrics["bayes_fail_chance"],
                "randomness_chance": metrics["randomness_chance"],
                "fail_chance": metrics["fail_chance"],
                "avg_gen_time_ms": avg_gen_time_ms,
                "avg_test_time_ms": avg_test_time_ms,
                "verdict": passrate_verdict,
                "bayes_verdict": bayes_verdict,
            })
            pass_rates_for_generator.append(metrics["pass_rate"])
            final_scores_for_generator.append(metrics["final_score"])
            classical_randomness_chances_for_generator.append(metrics["classical_randomness_chance"])
            bayes_randomness_chances_for_generator.append(metrics["bayes_randomness_chance"])

            output_str = (
                f"  {test_name}:\n"
                f"   - Úspěšných vzorků: {pass_counts[test_name]}/{num_samples} ({metrics['pass_rate'] * 100:.2f} %)\n"
                f"   - Mean p-value: {metrics['mean_p']:.6f}, Median p-value: {metrics['median_p']:.6f}\n"
                f"   - Stability score: {metrics['stability_score']:.4f}, Final score: {metrics['final_score']:.4f}\n"
                f"   - Pass posterior: {metrics['pass_posterior_chance']:.4f}\n"
                f"   - Avg gen time: {avg_gen_time_ms:.2f} ms, Avg test time: {avg_test_time_ms:.2f} ms\n"
                f"   - Odhad náhodnosti (classical): {metrics['classical_randomness_chance']:.4f}, Šance selhání (classical): {metrics['classical_fail_chance']:.4f}\n"
                f"   - Odhad náhodnosti (bayes): {metrics['bayes_randomness_chance']:.4f}, Šance selhání (bayes): {metrics['bayes_fail_chance']:.4f}\n"
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
        generator_avg_final_score = float(np.mean(final_scores_for_generator)) if final_scores_for_generator else 0.0
        generator_avg_pass_rate = float(np.mean(pass_rates_for_generator)) if pass_rates_for_generator else 0.0
        generator_avg_classical_randomness_chance = (
            float(np.mean(classical_randomness_chances_for_generator))
            if classical_randomness_chances_for_generator
            else 0.0
        )
        if classical_randomness_chances_for_generator:
            clipped_classical = np.clip(np.array(classical_randomness_chances_for_generator, dtype=float), 1e-12, 1.0)
            generator_geo_classical_randomness_chance = float(np.exp(np.mean(np.log(clipped_classical))))
        else:
            generator_geo_classical_randomness_chance = 0.0
        generator_classical_randomness_chance = (
            float(np.min(classical_randomness_chances_for_generator))
            if classical_randomness_chances_for_generator
            else 0.0
        )
        generator_classical_fail_chance = float(1.0 - generator_classical_randomness_chance)

        generator_avg_bayes_randomness_chance = (
            float(np.mean(bayes_randomness_chances_for_generator))
            if bayes_randomness_chances_for_generator
            else 0.0
        )
        if bayes_randomness_chances_for_generator:
            clipped_bayes = np.clip(np.array(bayes_randomness_chances_for_generator, dtype=float), 1e-12, 1.0)
            generator_geo_bayes_randomness_chance = float(np.exp(np.mean(np.log(clipped_bayes))))
        else:
            generator_geo_bayes_randomness_chance = 0.0
        generator_bayes_randomness_chance = (
            float(np.min(bayes_randomness_chances_for_generator))
            if bayes_randomness_chances_for_generator
            else 0.0
        )
        generator_bayes_fail_chance = float(1.0 - generator_bayes_randomness_chance)
        cherry_pick_summary_rows.append({
            "generator": gen_name,
            "generator_class": getattr(generator, "generator_class", "UNKNOWN"),
            "pass_count": pass_count,
            "total_tests": total_tests,
            "pass_ratio": pass_ratio,
            "pass_tests": "; ".join(pass_tests),
            "fail_tests": "; ".join(fail_tests),
            "avg_gen_time_ms": avg_gen_time_ms,
            "avg_test_time_ms": avg_test_time_ms_all_tests,
            "avg_final_score": generator_avg_final_score,
            "avg_pass_rate": generator_avg_pass_rate,
            "aggregation": "min_across_tests",
            "avg_classical_randomness_chance": generator_avg_classical_randomness_chance,
            "geo_classical_randomness_chance": generator_geo_classical_randomness_chance,
            "classical_randomness_chance": generator_classical_randomness_chance,
            "classical_fail_chance": generator_classical_fail_chance,
            "avg_bayes_randomness_chance": generator_avg_bayes_randomness_chance,
            "geo_bayes_randomness_chance": generator_geo_bayes_randomness_chance,
            "bayes_randomness_chance": generator_bayes_randomness_chance,
            "bayes_fail_chance": generator_bayes_fail_chance,
            "avg_randomness_chance": generator_avg_bayes_randomness_chance,
            "geo_randomness_chance": generator_geo_bayes_randomness_chance,
            "randomness_chance": generator_bayes_randomness_chance,
            "fail_chance": generator_bayes_fail_chance,
        })

        generation_line = (
            f"  => Průměrný čas generování pro {gen_name}: {avg_gen_time_ms:.2f} ms\n"
            f"  => Průměrný čas testu pro {gen_name}: {avg_test_time_ms_all_tests:.2f} ms\n"
            f"  => Průměrný pass-rate přes všechny testy: {generator_avg_pass_rate * 100:.2f} %\n"
            f"  => Odhad náhodnosti (classical průměr přes testy): {generator_avg_classical_randomness_chance:.4f} (0-1)\n"
            f"  => Odhad náhodnosti (classical geometrický průměr): {generator_geo_classical_randomness_chance:.4f} (0-1)\n"
            f"  => Odhad náhodnosti sekvence generátoru (classical min přes testy): {generator_classical_randomness_chance:.4f} (0-1)\n"
            f"  => Odhad šance selhání (classical): {generator_classical_fail_chance:.4f} (0-1)\n"
            f"  => Odhad náhodnosti (bayes průměr přes testy): {generator_avg_bayes_randomness_chance:.4f} (0-1)\n"
            f"  => Odhad náhodnosti (bayes geometrický průměr): {generator_geo_bayes_randomness_chance:.4f} (0-1)\n"
            f"  => Odhad náhodnosti sekvence generátoru (bayes min přes testy): {generator_bayes_randomness_chance:.4f} (0-1)\n"
            f"  => Odhad šance selhání (bayes): {generator_bayes_fail_chance:.4f} (0-1)\n\n"
        )
        print(generation_line, end="")
        file.write(generation_line)

    write_csv_rows(
        csv_paths["cherry_pick_metrics"],
        [
            "generator", "generator_class", "test", "sample_size", "sample_iter",
            "pass_rate", "mean_p", "median_p", "stability_score", "final_score",
            "pass_posterior_chance",
            "classical_randomness_chance", "classical_fail_chance",
            "bayes_randomness_chance", "bayes_fail_chance",
            "randomness_chance", "fail_chance", "avg_gen_time_ms", "avg_test_time_ms", "verdict", "bayes_verdict",
        ],
        cherry_pick_rows,
    )
    write_csv_rows(
        csv_paths["cherry_pick_summary"],
        [
            "generator", "generator_class", "pass_count", "total_tests", "pass_ratio",
            "pass_tests", "fail_tests", "avg_gen_time_ms", "avg_test_time_ms", "avg_final_score",
            "avg_pass_rate",
            "aggregation",
            "avg_classical_randomness_chance", "geo_classical_randomness_chance",
            "classical_randomness_chance", "classical_fail_chance",
            "avg_bayes_randomness_chance", "geo_bayes_randomness_chance",
            "bayes_randomness_chance", "bayes_fail_chance",
            "avg_randomness_chance", "geo_randomness_chance",
            "randomness_chance", "fail_chance",
        ],
        cherry_pick_summary_rows,
    )

    csv_info = (
        "CSV export:\n"
        f"- {csv_paths['cherry_pick_metrics']}\n"
        f"- {csv_paths['cherry_pick_summary']}\n"
    )
    print(csv_info)
    file.write(csv_info)
    print()
    file.write("\n")


def run_benchmark_mode(generators, tests_to_run, sample_sizes, repeats: int, alpha: float, file, csv_paths: dict):
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
                    c_row = class_stats[class_key]
                    c_row["count"] += 1
                    c_row["gen_time_sum"] += gen_time
                    c_row["gen_peak_sum"] += gen_peak
                    c_row["test_time_sum"] += measured["elapsed_sec"]
                    c_row["test_peak_sum"] += measured["peak_bytes"]
                    c_row["p_value_sum"] += measured["p_value"]
                    c_row["pass_sum"] += int(measured["passed"])

    file.write("Výsledky benchmarku (průměry):\n")
    benchmark_rows = []
    for gen_name, test_name, size_bits in sorted(stats.keys(), key=lambda x: (x[2], x[0], x[1])):
        row = stats[(gen_name, test_name, size_bits)]
        c = row["count"]
        avg_gen_ms = (row["gen_time_sum"] / c) * 1000.0
        avg_test_ms = (row["test_time_sum"] / c) * 1000.0
        avg_gen_kib = row["gen_peak_sum"] / c / 1024.0
        avg_test_kib = row["test_peak_sum"] / c / 1024.0
        pass_rate = row["pass_sum"] / c
        avg_p = row["p_value_sum"] / c

        line = (
            f"[{size_bits} bit] {gen_name} | {test_name}\n"
            f"  - gen:  {avg_gen_ms:.2f} ms, peak {avg_gen_kib:.2f} KiB\n"
            f"  - test: {avg_test_ms:.2f} ms, peak {avg_test_kib:.2f} KiB\n"
            f"  - avg p-value: {avg_p:.6f}, pass-rate: {pass_rate * 100:.2f} %\n\n"
        )
        print(line, end="")
        file.write(line)
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
        })

    file.write("==================== Agregace podle tříd generátorů ====================\n")
    print("==================== Agregace podle tříd generátorů ====================")
    class_rows = []
    for gen_class, size_bits in sorted(class_stats.keys(), key=lambda x: (x[1], x[0])):
        row = class_stats[(gen_class, size_bits)]
        c = row["count"]
        avg_gen_ms = (row["gen_time_sum"] / c) * 1000.0
        avg_test_ms = (row["test_time_sum"] / c) * 1000.0
        avg_gen_kib = row["gen_peak_sum"] / c / 1024.0
        avg_test_kib = row["test_peak_sum"] / c / 1024.0
        pass_rate = row["pass_sum"] / c
        avg_p = row["p_value_sum"] / c
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
        })

    write_csv_rows(
        csv_paths["benchmark_metrics"],
        ["sample_size", "generator", "test", "avg_gen_ms", "avg_test_ms", "avg_gen_kib", "avg_test_kib", "avg_p_value",
         "pass_rate"],
        benchmark_rows,
    )
    write_csv_rows(
        csv_paths["benchmark_class"],
        ["sample_size", "generator_class", "avg_gen_ms", "avg_test_ms", "avg_gen_kib", "avg_test_kib", "avg_p_value",
         "pass_rate"],
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

    # loading cfg
    SAMPLE_SIZE = int(config.get("sample_size", 1_000_000))
    NUM_SAMPLES = int(config.get("sample_iter", 100))
    OUTPUT_DIR = config.get("output_dir", "outputs")
    MODE = config.get("mode", "cherry-pick").lower()
    MODE = {"chery-pick": "cherry-pick", "cherry_pick": "cherry-pick"}.get(MODE, MODE)
    ALPHA = float(config.get("alpha", 0.01))
    BAYES_PASS_THRESHOLD = float(config.get("bayes_pass_threshold", 0.95))
    BENCHMARK_SIZES = config.get("benchmark_sample_sizes", [10000, 100000, 1000000])
    BENCHMARK_REPEATS = int(config.get("benchmark_repeats", 3))
    AMBIENT_SAMPLE_RATE = int(config.get("ambient_sample_rate", 48000))
    AMBIENT_WHITENING = str(config.get("ambient_whitening", "von-neumann+sha256"))
    AMBIENT_HASH_BLOCK_BYTES = int(config.get("ambient_hash_block_bytes", 4096))
    ambient_dataset_info = None

    os.makedirs(OUTPUT_DIR, exist_ok=True)
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

    # Inicializace testů
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
    cherry_pick_runs = []

    try:
        if MODE == "benchmark":
            generators = dict(all_generators)
            benchmark_audio_files = prepare_audio_files_for_benchmark(
                sample_sizes=BENCHMARK_SIZES,
                repeats=BENCHMARK_REPEATS,
            )
            generators[AMBIENT_GENERATOR_NAME] = AudioSampleBatchGenerator(
                filepaths=benchmark_audio_files,
                strict=True,
            )
            ambient_dataset_info = f"{AMBIENT_AUDIO_DIR} ({len(benchmark_audio_files)} files)"

        if MODE == "cherry-pick":
            print_generators(all_generators)
            selected_generator_name, selected_generator = get_generator_with_index(all_generators)
            if selected_generator_name is None:
                for gen_name, gen_obj in all_generators.items():
                    run_generator = gen_obj
                    run_num_samples = NUM_SAMPLES
                    run_ambient_info = None

                    if gen_name == AMBIENT_GENERATOR_NAME:
                        file_paths = prepare_audio_dataset(
                            sample_size_bits=SAMPLE_SIZE,
                            sample_count=NUM_SAMPLES,
                            sample_rate=AMBIENT_SAMPLE_RATE,
                            whitening=AMBIENT_WHITENING,
                            hash_block_bytes=AMBIENT_HASH_BLOCK_BYTES,
                        )
                        run_generator = AudioSampleBatchGenerator(filepaths=file_paths, strict=True)
                        run_num_samples = len(file_paths)
                        run_ambient_info = f"{AMBIENT_AUDIO_DIR} ({len(file_paths)} files)"

                    cherry_pick_runs.append({
                        "name": gen_name,
                        "generator": run_generator,
                        "num_samples": run_num_samples,
                        "ambient_info": run_ambient_info,
                    })
            else:
                run_num_samples = NUM_SAMPLES
                run_ambient_info = None
                if selected_generator_name == AMBIENT_GENERATOR_NAME:
                    file_paths = prepare_audio_dataset(
                        sample_size_bits=SAMPLE_SIZE,
                        sample_count=NUM_SAMPLES,
                        sample_rate=AMBIENT_SAMPLE_RATE,
                        whitening=AMBIENT_WHITENING,
                        hash_block_bytes=AMBIENT_HASH_BLOCK_BYTES,
                    )
                    selected_generator = AudioSampleBatchGenerator(filepaths=file_paths, strict=True)
                    run_num_samples = len(file_paths)
                    run_ambient_info = f"{AMBIENT_AUDIO_DIR} ({len(file_paths)} files)"

                cherry_pick_runs.append({
                    "name": selected_generator_name,
                    "generator": selected_generator,
                    "num_samples": run_num_samples,
                    "ambient_info": run_ambient_info,
                })
    except Exception as e:
        print(e)
        sys.exit(1)

    if MODE == "benchmark":
        mode_output_dir = os.path.join(OUTPUT_DIR, "benchmarks")
        os.makedirs(mode_output_dir, exist_ok=True)
        OUTPUT_FILE = os.path.join(mode_output_dir, f"output_{MODE}-{run_stamp}.txt")
        CSV_PATHS = make_output_paths(OUTPUT_FILE, MODE)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
            file.write(f"Mode: {MODE}\n")
            file.write(
                "Ambient noise config: "
                f"whitening={AMBIENT_WHITENING}, "
                f"sample_rate={AMBIENT_SAMPLE_RATE}, hash_block_bytes={AMBIENT_HASH_BLOCK_BYTES}\n"
            )
            if ambient_dataset_info:
                file.write(f"Ambient audio dataset: {ambient_dataset_info}\n")
            file.write("\n")

            run_benchmark_mode(
                generators=generators,
                tests_to_run=tests_to_run,
                sample_sizes=BENCHMARK_SIZES,
                repeats=BENCHMARK_REPEATS,
                alpha=ALPHA,
                file=file,
                csv_paths=CSV_PATHS,
            )

        print(f"\nVýsledky byly úspěšně uloženy do: {OUTPUT_FILE}")
    elif MODE == "cherry-pick":
        if not cherry_pick_runs:
            print("Cherry-pick režim nemá žádný generátor ke spuštění.")
            sys.exit(1)

        output_files = []
        for run_cfg in cherry_pick_runs:
            gen_name = run_cfg["name"]
            mode_output_dir = os.path.join(OUTPUT_DIR, gen_name)
            os.makedirs(mode_output_dir, exist_ok=True)

            per_run_stamp = str(datetime.datetime.now())
            output_file = os.path.join(mode_output_dir, f"output_{MODE}-{per_run_stamp}.txt")
            csv_paths = make_output_paths(output_file, MODE)

            with open(output_file, "w", encoding="utf-8") as file:
                file.write(f"Mode: {MODE}\n")
                file.write(
                    "Ambient noise config: "
                    f"whitening={AMBIENT_WHITENING}, "
                    f"sample_rate={AMBIENT_SAMPLE_RATE}, hash_block_bytes={AMBIENT_HASH_BLOCK_BYTES}\n"
                )
                if run_cfg["ambient_info"]:
                    file.write(f"Ambient audio dataset: {run_cfg['ambient_info']}\n")
                file.write("\n")

                write_complexity_section(file, tests_to_run)
                run_cherry_pick_mode(
                    generators={gen_name: run_cfg["generator"]},
                    tests_to_run=tests_to_run,
                    sample_size=SAMPLE_SIZE,
                    num_samples=run_cfg["num_samples"],
                    alpha=ALPHA,
                    bayes_pass_threshold=BAYES_PASS_THRESHOLD,
                    file=file,
                    csv_paths=csv_paths,
                )

            output_files.append(output_file)

        print("\nVýsledky byly úspěšně uloženy do:")
        for path in output_files:
            print(f"- {path}")
    else:
        print(f"Neznámý režim: {MODE}. Použij 'benchmark' nebo 'cherry-pick'.")
        sys.exit(1)
