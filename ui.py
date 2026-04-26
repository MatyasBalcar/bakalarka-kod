"""
Tento modul ridi uzivatelske rozhrani
"""


def print_tests(tests):
    for test_index, test in enumerate(tests):
        print(f"{test_index}. {test}")


def print_generators(generators):
    print("Dostupné generátory:")
    for i, name in enumerate(generators.keys()):
        print(f"{i}. {name}")


def get_generator_with_index(generators):
    selected_input = input(
        "Enter generator id to run in single-source mode: eg. 0\nPress [Enter] to run all generators separately\n")
    if selected_input.strip() == "":
        return None, None

    selected_index = int(selected_input.strip())
    generator_names = list(generators.keys())
    if selected_index < 0 or selected_index >= len(generator_names):
        raise Exception("Generator index out of range")

    generator_name = generator_names[selected_index]
    return generator_name, generators[generator_name]
