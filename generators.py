import hashlib
import importlib
import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from tqdm import tqdm


class Generator(ABC):
    generator_class = "UNKNOWN"

    @abstractmethod
    def generate(self, size_bits: int) -> np.ndarray:
        pass


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    if bits.size == 0:
        return b""
    packed = np.packbits(bits.astype(np.uint8), bitorder="big")
    return packed.tobytes()


def _bytes_to_bits(data: bytes) -> np.ndarray:
    if not data:
        return np.array([], dtype=np.uint8)
    arr = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(arr, bitorder="big").astype(np.uint8)


def _von_neumann_extract(bits: np.ndarray) -> np.ndarray:
    if bits.size < 2:
        return np.array([], dtype=np.uint8)

    pair_count = bits.size // 2
    reshaped = bits[: pair_count * 2].reshape(pair_count, 2)
    keep = reshaped[:, 0] != reshaped[:, 1]
    if not np.any(keep):
        return np.array([], dtype=np.uint8)

    kept = reshaped[keep]
    return kept[:, 0].astype(np.uint8)


def _load_bits_from_bin_file(filepath: str) -> np.ndarray:
    if not filepath.lower().endswith(".bin"):
        raise ValueError("Only .bin files are supported.")
    byte_array = np.fromfile(filepath, dtype=np.uint8)
    bits = np.unpackbits(byte_array, bitorder="big")
    if bits.size == 0:
        raise ValueError(f"Input file is empty: {filepath}")
    return bits.astype(np.uint8)


class AudioSampleBatchGenerator(Generator):
    """One-file-per-sample generator for pre-captured ambient datasets."""

    generator_class = "TRNG"

    def __init__(
            self,
            filepaths: list[str],
            strict: bool = True,
            enforce_size_bits: bool = True,
            warn_on_short_sample: bool = False,
    ):
        if not filepaths:
            raise ValueError("AudioSampleBatchGenerator requires at least one .bin file.")

        self.filepaths = filepaths
        self.strict = strict
        self.enforce_size_bits = enforce_size_bits
        self.warn_on_short_sample = warn_on_short_sample
        self._bits_by_file = [_load_bits_from_bin_file(path) for path in filepaths]
        self._sample_cursor = 0
        self._warned_short_sample = False

    def generate(self, size_bits: int) -> np.ndarray:
        if size_bits <= 0:
            return np.array([], dtype=np.uint8)

        if self._sample_cursor >= len(self._bits_by_file):
            if self.strict:
                raise ValueError(
                    "No more audio samples available. "
                    "Increase sample count or disable strict mode."
                )
            self._sample_cursor = 0

        bits = self._bits_by_file[self._sample_cursor]
        self._sample_cursor += 1

        if bits.size < size_bits:
            if self.enforce_size_bits:
                raise ValueError(
                    f"Audio sample is too short. Required {size_bits} bits, available {bits.size}."
                )
            if self.warn_on_short_sample and not self._warned_short_sample:
                print(
                    "WARNING: Dataset sample is shorter than requested sample_size "
                    f"({bits.size} < {size_bits}). Running anyway."
                )
                self._warned_short_sample = True

        if self.enforce_size_bits:
            return bits[:size_bits]
        return bits


class AmbientNoiseGenerator(Generator):
    generator_class = "TRNG"

    def __init__(
            self,
            sample_rate: int = 48_000,
            channels: int = 1,
            block_frames: int = 4096,
            lsb_index: int = 0,
            whitening: str = "von-neumann+sha256",
            hash_block_bytes: int = 4096,
            device: str | None = None,
            capture_path: str | None = None,
            capture_overwrite: bool = True,
            replay_path: str | None = None,
            replay_loop: bool = False,
    ):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if channels <= 0:
            raise ValueError("channels must be > 0")
        if block_frames <= 0:
            raise ValueError("block_frames must be > 0")
        if lsb_index < 0 or lsb_index > 3:
            raise ValueError("lsb_index must be in range 0..3")
        if whitening not in ("none", "von-neumann", "sha256", "von-neumann+sha256"):
            raise ValueError("Unsupported whitening mode")
        if hash_block_bytes <= 0:
            raise ValueError("hash_block_bytes must be > 0")
        if capture_path and replay_path:
            raise ValueError("Use either capture_path or replay_path, not both at once")

        self.sample_rate = sample_rate
        self.channels = channels
        self.block_frames = block_frames
        self.lsb_index = lsb_index
        self.whitening = whitening
        self.hash_block_bytes = hash_block_bytes
        self.hash_block_bits = hash_block_bytes * 8
        self.device = device

        self.capture_path = capture_path
        self.capture_overwrite = capture_overwrite
        self.replay_path = replay_path
        self.replay_loop = replay_loop

        self._sd: Any = None
        self._stream: Any = None
        self._capture_file = None
        self._capture_pending_bits = np.array([], dtype=np.uint8)

        self._replay_bits = np.array([], dtype=np.uint8)
        self._replay_cursor = 0

        self._pending_output_bits = np.array([], dtype=np.uint8)
        self._pending_vn_bits = np.array([], dtype=np.uint8)
        self._pending_hash_input_bits = np.array([], dtype=np.uint8)

        if self.replay_path:
            self._load_replay_bits(self.replay_path)
        elif self.capture_path:
            self._open_capture_file(self.capture_path)

    def _open_capture_file(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        mode = "wb" if self.capture_overwrite else "ab"
        self._capture_file = open(path, mode)

    def _flush_capture(self, force: bool = False) -> None:
        if self._capture_file is None:
            return
        if self._capture_pending_bits.size == 0:
            return

        ready_count = (self._capture_pending_bits.size // 8) * 8
        if force and ready_count < self._capture_pending_bits.size:
            ready_count = self._capture_pending_bits.size

        if ready_count <= 0:
            return

        chunk = self._capture_pending_bits[:ready_count]
        self._capture_pending_bits = self._capture_pending_bits[ready_count:]
        self._capture_file.write(_bits_to_bytes(chunk))
        self._capture_file.flush()

    def _capture_bits(self, bits: np.ndarray) -> None:
        if self._capture_file is None or bits.size == 0:
            return
        self._capture_pending_bits = np.concatenate([self._capture_pending_bits, bits])
        self._flush_capture(force=False)

    def _load_replay_bits(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Replay file not found: {path}")
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            raise ValueError(f"Replay file is empty: {path}")
        self._replay_bits = np.unpackbits(data, bitorder="big").astype(np.uint8)

    def _ensure_stream(self) -> None:
        if self._stream is not None:
            return
        try:
            self._sd = importlib.import_module("sounddevice")
        except Exception as exc:
            raise RuntimeError(
                "AmbientNoiseGenerator requires package 'sounddevice'. "
                "Install with: pip install sounddevice"
            ) from exc

        self._stream = self._sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.block_frames,
            device=self.device,
        )
        self._stream.start()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._flush_capture(force=True)
        if self._capture_file is not None:
            self._capture_file.close()
            self._capture_file = None

    def __del__(self):
        self.close()

    def _read_raw_lsb_bits(self) -> np.ndarray:
        data, _overflowed = self._stream.read(self.block_frames)
        flat = data.reshape(-1).astype(np.int16)
        unsigned = flat.astype(np.uint16)
        return np.bitwise_and(unsigned >> self.lsb_index, 1).astype(np.uint8)

    def _apply_whitening(self, bits: np.ndarray) -> np.ndarray:
        out = bits

        if self.whitening in ("von-neumann", "von-neumann+sha256"):
            if self._pending_vn_bits.size:
                out = np.concatenate([self._pending_vn_bits, out])
                self._pending_vn_bits = np.array([], dtype=np.uint8)
            if out.size % 2 == 1:
                self._pending_vn_bits = out[-1:].copy()
                out = out[:-1]
            out = _von_neumann_extract(out)

        if self.whitening in ("sha256", "von-neumann+sha256"):
            if self._pending_hash_input_bits.size:
                out = np.concatenate([self._pending_hash_input_bits, out])
                self._pending_hash_input_bits = np.array([], dtype=np.uint8)

            digests_bits = []
            while out.size >= self.hash_block_bits:
                block_bits = out[:self.hash_block_bits]
                out = out[self.hash_block_bits:]
                digest = hashlib.sha256(_bits_to_bytes(block_bits)).digest()
                digests_bits.append(_bytes_to_bits(digest))

            self._pending_hash_input_bits = out
            if digests_bits:
                return np.concatenate(digests_bits)
            return np.array([], dtype=np.uint8)

        return out

    def generate(self, size_bits: int) -> np.ndarray:
        if size_bits <= 0:
            return np.array([], dtype=np.uint8)

        if self.replay_path:
            end = self._replay_cursor + size_bits
            if end <= self._replay_bits.size:
                out = self._replay_bits[self._replay_cursor:end]
                self._replay_cursor = end
                return out
            if not self.replay_loop:
                remaining = self._replay_bits.size - self._replay_cursor
                raise ValueError(
                    "Replay file does not contain enough bits. "
                    f"Requested {size_bits}, remaining {remaining}."
                )
            idx = (np.arange(size_bits, dtype=np.int64) + self._replay_cursor) % self._replay_bits.size
            out = self._replay_bits[idx]
            self._replay_cursor = (self._replay_cursor + size_bits) % self._replay_bits.size
            return out

        self._ensure_stream()

        while self._pending_output_bits.size < size_bits:
            raw_bits = self._read_raw_lsb_bits()
            whitened = self._apply_whitening(raw_bits)
            if whitened.size:
                self._pending_output_bits = np.concatenate([self._pending_output_bits, whitened])

        out = self._pending_output_bits[:size_bits]
        self._pending_output_bits = self._pending_output_bits[size_bits:]
        self._capture_bits(out)
        return out


class LCG(Generator):
    # * PRNG
    generator_class = "PRNG"

    def __init__(self, seed: int, a: int = 1664525, c: int = 1013904223, m: int = 2 ** 32):
        self.state = seed
        self.a = a
        self.c = c
        self.m = m

    def generate(self, size_bits: int) -> np.ndarray:
        size_ints = (size_bits // 32) + 1
        results = np.zeros(size_ints, dtype=np.uint32)

        for i in range(size_ints):
            self.state = (self.a * self.state + self.c) % self.m
            results[i] = self.state

        bit_array = np.unpackbits(results.view(np.uint8))
        return bit_array[:size_bits]


class MersenneTwister(Generator):
    # * PRNG
    generator_class = "PRNG"

    _N = 624
    _M = 397
    _MATRIX_A = 0x9908B0DF
    _UPPER_MASK = 0x80000000
    _LOWER_MASK = 0x7FFFFFFF

    def __init__(self, seed: int):
        self._mt = np.zeros(self._N, dtype=np.uint32)
        self._index = self._N
        self._seed(seed)

    def _seed(self, seed: int) -> None:
        self._mt[0] = np.uint32(seed & 0xFFFFFFFF)
        for i in range(1, self._N):
            prev = int(self._mt[i - 1])
            value = 1812433253 * (prev ^ (prev >> 30)) + i
            self._mt[i] = np.uint32(value & 0xFFFFFFFF)
        self._index = self._N

    def _twist(self) -> None:
        for i in range(self._N):
            x = (int(self._mt[i]) & self._UPPER_MASK) | (int(self._mt[(i + 1) % self._N]) & self._LOWER_MASK)
            x_a = x >> 1
            if x & 1:
                x_a ^= self._MATRIX_A
            self._mt[i] = np.uint32(int(self._mt[(i + self._M) % self._N]) ^ x_a)
        self._index = 0

    def _next_uint32(self) -> np.uint32:
        if self._index >= self._N:
            self._twist()

        y = int(self._mt[self._index])
        self._index += 1

        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return np.uint32(y & 0xFFFFFFFF)

    def generate(self, size_bits: int) -> np.ndarray:
        if size_bits <= 0:
            return np.array([], dtype=np.uint8)

        size_ints = (size_bits + 31) // 32
        values = np.empty(size_ints, dtype=np.uint32)
        for i in range(size_ints):
            values[i] = self._next_uint32()

        return np.unpackbits(values.view(np.uint8))[:size_bits]


class PCG64Wrapper(Generator):
    # * PRNG [CONTROL, NOT IMPLEMENTED]
    generator_class = "PRNG"

    def __init__(self, seed: int):
        self.rng = np.random.Generator(np.random.PCG64(seed))

    def generate(self, size_bits: int) -> np.ndarray:
        size_bytes = (size_bits // 8) + 1
        random_bytes = self.rng.bytes(size_bytes)
        return np.unpackbits(np.frombuffer(random_bytes, dtype=np.uint8))[:size_bits]


class XORShift32(Generator):
    # * PRNG
    generator_class = "PRNG"

    def __init__(self, seed: int):
        self.state = np.uint32(seed if seed != 0 else 2463534242)

    def _next_uint32(self) -> np.uint32:
        x = self.state
        x ^= (x << np.uint32(13))
        x ^= (x >> np.uint32(17))
        x ^= (x << np.uint32(5))
        self.state = np.uint32(x)
        return self.state

    def generate(self, size_bits: int) -> np.ndarray:
        size_ints = (size_bits // 32) + 1
        results = np.zeros(size_ints, dtype=np.uint32)

        for i in range(size_ints):
            results[i] = self._next_uint32()

        return np.unpackbits(results.view(np.uint8))[:size_bits]


class OSUrandomGenerator(Generator):
    # * CSPRNG [CONTROL, NOT IMPLEMENTED]
    generator_class = "CSPRNG"

    def generate(self, size_bits: int) -> np.ndarray:
        size_bytes = (size_bits // 8) + 1
        random_bytes = os.urandom(size_bytes)
        return np.unpackbits(np.frombuffer(random_bytes, dtype=np.uint8))[:size_bits]


class BlumBlumShub(Generator):
    # * CSPRNG
    generator_class = "CSPRNG"

    def __init__(self, p: int, q: int, seed: int):
        self.n = p * q
        self.state = seed % self.n

    def generate(self, size_bits: int) -> np.ndarray:
        results = np.zeros(size_bits, dtype=np.uint8)

        for i in tqdm(range(size_bits), desc="Generování BBS", leave=False):
            self.state = (self.state ** 2) % self.n
            results[i] = self.state % 2

        return results


class AlternatingGenerator(Generator):
    """Záměrně špatný generátor pro testování detekce slabých generátorů.
    Produkuje porad 1010101....
    Ale presto projde nejake testy (frequency a block frequency), ostatni fail"""

    # * [NOT A REAL GENERATOR]
    generator_class = "PR"

    def __init__(self, seed: int):
        self.seed = seed

    def generate(self, size_bits: int) -> np.ndarray:
        pattern = np.array([1, 0], dtype=np.uint8)
        return np.tile(pattern, (size_bits + 1) // 2)[:size_bits]


class RepeatingGenerator(Generator):
    """
    Opakuje stejnou hodnotu stale, tohle uz failne vsechno
    """
    # * [NOT A REAL GENERATOR]
    generator_class = "PR"

    def __init__(self, seed: int):
        self.seed = seed

    def generate(self, size_bits: int) -> np.ndarray:
        pattern = np.array([1], dtype=np.uint8)
        return np.tile(pattern, (size_bits + 1))[:size_bits]
