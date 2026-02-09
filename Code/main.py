"""
Main python file, so far mainly used for testing the functionality of the project.
Later to be developed to be the main runnable file
"""


def primitive_monobit_test(sequence: list) -> bool:
    """
    Primitively cheks if sequance of 1,0 is uniform
    :param sequence: binary sequence
    :return: is this sequence uniform?
    """
    ones_count = 0
    zeroes_count = 0

    for bit in sequence:
        if bit == 1:
            ones_count += 1
        else:
            zeroes_count += 1

    return ones_count == zeroes_count


def normalize_sequence(sequence: list, u_max: int) -> list:
    """
    Normalizes sequence given a sequance and a universe maximum
    :param sequence: sequnce of intigers
    :param u_max: universe maximum
    :return: normalized sequence to [0,1)
    """
    result = []

    for i in range(0, len(sequence)):
        if not (0 <= sequence[i] < 1):
            n_value = sequence[i] / (u_max + 1)
        else:
            n_value = sequence[i]
        result[i] = n_value

    return result
