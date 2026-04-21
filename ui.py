"""
Tento modul ridi uzivatelske rozhrsni
"""

def print_tests(tests):
    i = 0
    for test in tests:
        print(f"{i}. {test}")
        i += 1


def print_generators(generators):
    print("Dostupné generátory:")
    for i, name in enumerate(generators.keys()):
        print(f"{i}. {name}")


def get_generator_with_index(generators):
    data = input(
        "Enter generator id to run in cherry-pick mode: eg. 0\nPress [Enter] to run all generators separately\n")
    if data.strip() == "":
        return None, None

    index = int(data.strip())
    names = list(generators.keys())
    if index < 0 or index >= len(names):
        raise Exception("Generator index out of range")

    name = names[index]
    return name, generators[name]


def get_tests_with_indexes(tests):
    data = input("Enter id's of tests to use: eg. 0,1,3\nPress [Enter] to run all tests")
    indexes = []
    if data.strip() == "":
        indexes = range(len(tests))
    else:
        indexes = list(map(int, data.split(",")))
    if (min(indexes) < 0) or (max(indexes) > (len(tests) - 1)):
        raise Exception("Index out of range")
    return [tests[i] for i in indexes]
