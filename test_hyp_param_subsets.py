import itertools


def subsets(input_set):
    """Generate all subsets of a given set."""
    # List to store all subsets
    all_subsets = []

    # Loop over all possible subset lengths
    for r in range(len(input_set) + 1):
        # Add all combinations of the current length to the list of all subsets
        all_subsets.extend(itertools.combinations(input_set, r))

    return all_subsets


# Example usage
input_set = {1, 2, 3}
result = subsets(input_set)
print(result)