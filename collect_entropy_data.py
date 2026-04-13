"""Collect external entropy data and save it as a .bin file.

Default source uses os.urandom (kernel CSPRNG seeded by system entropy).
Optionally, you can use /dev/random for stronger blocking entropy behavior
on Unix-like systems.

For experimental TRNG workflows, source=mic captures ambient microphone noise,
extracts low-order (LSB) bits, and can apply debiasing/conditioning.
"""

import argparse
import hashlib
import importlib
import os
import sys
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def resolve_output_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if path.suffix.lower() != ".bin":
        raise ValueError("Output file must use .bin extension.")
    return path


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


def von_neumann_extract(bits: np.ndarray) -> np.ndarray:
    """Debias bitstream: 01->0, 10->1, discard 00/11."""
    if bits.size < 2:
        return np.array([], dtype=np.uint8)

    pair_count = bits.size // 2
    reshaped = bits[: pair_count * 2].reshape(pair_count, 2)
    keep = reshaped[:, 0] != reshaped[:, 1]
    if not np.any(keep):
        return np.array([], dtype=np.uint8)

    kept = reshaped[keep]
    return kept[:, 0].astype(np.uint8)


class MicrophoneEntropySource:
    """Capture int16 audio frames from microphone and expose LSB bits."""

    def __init__(
            self,
            sample_rate: int,
            channels: int,
            block_frames: int,
            lsb_index: int,
            device: str | None,
    ):
        if sample_rate <= 0:
            raise ValueError("--sample-rate must be > 0")
        if channels <= 0:
            raise ValueError("--channels must be > 0")
        if block_frames <= 0:
            raise ValueError("--mic-block-frames must be > 0")
        if lsb_index < 0 or lsb_index > 3:
            raise ValueError("--lsb-index must be in range 0..3")

        self.sample_rate = sample_rate
        self.channels = channels
        self.block_frames = block_frames
        self.lsb_index = lsb_index
        self.device = device

        self._stream: Any = None
        self._sd: Any = None

    def __enter__(self) -> "MicrophoneEntropySource":
        try:
            self._sd = importlib.import_module("sounddevice")
        except Exception as exc:
            raise RuntimeError(
                "source=mic requires 'sounddevice'. Install with: pip install sounddevice"
            ) from exc

        self._stream = self._sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.block_frames,
            device=self.device,
        )
        self._stream.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read_raw_lsb_bits(self) -> np.ndarray:
        data, overflowed = self._stream.read(self.block_frames)
        if overflowed:
            # Continue; occasional overflow can happen on busy systems.
            pass
        flat = data.reshape(-1).astype(np.int16)
        unsigned = flat.astype(np.uint16)
        bits = np.bitwise_and(unsigned >> self.lsb_index, 1).astype(np.uint8)
        return bits


class MicrophoneEntropyPipeline:
    """Mic entropy pipeline: capture -> extract LSB -> optional whitening."""

    def __init__(
            self,
            source: MicrophoneEntropySource,
            whitening: str,
            hash_block_bytes: int,
    ):
        if hash_block_bytes <= 0:
            raise ValueError("--hash-block-bytes must be > 0")

        self.source = source
        self.whitening = whitening
        self.hash_block_bits = hash_block_bytes * 8

        self._pending_output_bits = np.array([], dtype=np.uint8)
        self._pending_vn_bits = np.array([], dtype=np.uint8)
        self._pending_hash_input_bits = np.array([], dtype=np.uint8)

    def _apply_whitening(self, bits: np.ndarray) -> np.ndarray:
        out = bits

        if self.whitening in ("von-neumann", "von-neumann+sha256"):
            if self._pending_vn_bits.size:
                out = np.concatenate([self._pending_vn_bits, out])
                self._pending_vn_bits = np.array([], dtype=np.uint8)

            if out.size % 2 == 1:
                self._pending_vn_bits = out[-1:].copy()
                out = out[:-1]
            out = von_neumann_extract(out)

        if self.whitening in ("sha256", "von-neumann+sha256"):
            if self._pending_hash_input_bits.size:
                out = np.concatenate([self._pending_hash_input_bits, out])
                self._pending_hash_input_bits = np.array([], dtype=np.uint8)

            digests_bits = []
            while out.size >= self.hash_block_bits:
                block_bits = out[:self.hash_block_bits]
                out = out[self.hash_block_bits:]
                block_bytes = _bits_to_bytes(block_bits)
                digest = hashlib.sha256(block_bytes).digest()
                digests_bits.append(_bytes_to_bits(digest))

            self._pending_hash_input_bits = out
            if digests_bits:
                return np.concatenate(digests_bits)
            return np.array([], dtype=np.uint8)

        return out

    def read_entropy_bytes(self, target_bytes: int) -> bytes:
        if target_bytes <= 0:
            return b""

        target_bits = target_bytes * 8
        while self._pending_output_bits.size < target_bits:
            raw_bits = self.source.read_raw_lsb_bits()
            whitened = self._apply_whitening(raw_bits)
            if whitened.size:
                self._pending_output_bits = np.concatenate([self._pending_output_bits, whitened])

        out_bits = self._pending_output_bits[:target_bits]
        self._pending_output_bits = self._pending_output_bits[target_bits:]
        return _bits_to_bytes(out_bits)


def read_entropy_chunk(source: str, chunk_size: int, random_dev_fd=None) -> bytes:
    if source == "urandom":
        return os.urandom(chunk_size)

    if source == "random":
        if random_dev_fd is None:
            raise RuntimeError("/dev/random is not available on this platform.")
        return os.read(random_dev_fd, chunk_size)

    raise ValueError(f"Unsupported source: {source}")


def collect_entropy(
        output_file: Path,
        total_bytes: int,
        source: str,
        overwrite: bool,
        mic_sample_rate: int = 48_000,
        mic_channels: int = 1,
        mic_block_frames: int = 4096,
        mic_lsb_index: int = 0,
        whitening: str = "none",
        hash_block_bytes: int = 4096,
        mic_device: str | None = None,
) -> tuple[int, str]:
    if total_bytes <= 0:
        raise ValueError("--bytes must be > 0")

    if output_file.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_file}. Use --overwrite to replace it."
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    hasher = hashlib.sha256()
    written = 0
    chunk_size = 1024 * 1024

    random_dev_fd = None
    mic_pipeline = None
    with ExitStack() as stack:
        if source == "random":
            if os.name == "nt":
                raise RuntimeError("source=random is not supported on Windows")
            random_dev_fd = os.open("/dev/random", os.O_RDONLY)
            stack.callback(os.close, random_dev_fd)
        elif source == "mic":
            mic_source = stack.enter_context(
                MicrophoneEntropySource(
                    sample_rate=mic_sample_rate,
                    channels=mic_channels,
                    block_frames=mic_block_frames,
                    lsb_index=mic_lsb_index,
                    device=mic_device,
                )
            )
            mic_pipeline = MicrophoneEntropyPipeline(
                source=mic_source,
                whitening=whitening,
                hash_block_bytes=hash_block_bytes,
            )

        with open(output_file, "wb") as f:
            while written < total_bytes:
                remaining = total_bytes - written
                to_read = min(chunk_size, remaining)
                if source == "mic":
                    if mic_pipeline is None:
                        raise RuntimeError("Microphone pipeline is not initialized.")
                    chunk = mic_pipeline.read_entropy_bytes(to_read)
                else:
                    chunk = read_entropy_chunk(source, to_read, random_dev_fd=random_dev_fd)
                if not chunk:
                    raise RuntimeError("Entropy source returned no data.")

                f.write(chunk)
                hasher.update(chunk)
                written += len(chunk)

    return int(written), str(hasher.hexdigest())


def build_default_output_path() -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("inputs") / f"{ts}.bin"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect entropy and store it into a .bin file.")
    parser.add_argument("--out", default=str(build_default_output_path()), help="Output .bin file path")
    parser.add_argument(
        "--bytes",
        type=int,
        default=1_000_000,
        help="How many bytes to collect (default: 1,000,000)",
    )
    parser.add_argument(
        "--source",
        choices=["urandom", "random", "mic"],
        default="urandom",
        help="Entropy source (default: urandom)",
    )
    parser.add_argument(
        "--whitening",
        choices=["none", "von-neumann", "sha256", "von-neumann+sha256"],
        default="none",
        help="Debias/conditioning mode (used mainly with --source mic)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48_000,
        help="Microphone sample rate for --source mic (default: 48000)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Microphone channels for --source mic (default: 1)",
    )
    parser.add_argument(
        "--mic-block-frames",
        type=int,
        default=4096,
        help="Microphone block size in frames for --source mic",
    )
    parser.add_argument(
        "--lsb-index",
        type=int,
        default=0,
        help="Which low-order bit to extract from int16 sample (0..3, default: 0)",
    )
    parser.add_argument(
        "--hash-block-bytes",
        type=int,
        default=4096,
        help="Input block size for SHA-256 conditioning modes",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional microphone device id/name for --source mic",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        out_path = resolve_output_path(args.out)
        written, digest = collect_entropy(
            output_file=out_path,
            total_bytes=args.bytes,
            source=args.source,
            overwrite=args.overwrite,
            mic_sample_rate=args.sample_rate,
            mic_channels=args.channels,
            mic_block_frames=args.mic_block_frames,
            mic_lsb_index=args.lsb_index,
            whitening=args.whitening,
            hash_block_bytes=args.hash_block_bytes,
            mic_device=args.device,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Entropy collection finished.")
    print(f"Source: {args.source}")
    print(f"Output: {out_path}")
    print(f"Bytes:  {written}")
    print(f"SHA256: {digest}")
    if args.source == "mic":
        print(
            "Mic params: "
            f"sample_rate={args.sample_rate}, channels={args.channels}, "
            f"block_frames={args.mic_block_frames}, lsb_index={args.lsb_index}, "
            f"whitening={args.whitening}, hash_block_bytes={args.hash_block_bytes}, device={args.device}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
