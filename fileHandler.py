import numpy as np


def save_bits_to_binary(filepath: str, bits: np.ndarray):
    byte_array = np.packbits(bits.astype(np.uint8))

    with open(filepath, "wb") as f:
        f.write(byte_array.tobytes())
