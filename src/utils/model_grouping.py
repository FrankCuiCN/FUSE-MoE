from collections import defaultdict


def model_grouping(model_names):
    # Initialize groupings
    group_by_type = defaultdict(list)
    group_by_size = defaultdict(list)
    # Process each model name
    for name in model_names:
        if "-" not in name:
            continue  # Skip invalid format
        model_type, model_size = name.rsplit("-", 1)

        group_by_type[model_type].append(name)
        group_by_size[model_size].append(name)
    # Return a tuple of model divisions
    return (group_by_type, group_by_size)


def model_size_key(size_str):
    if size_str.endswith("M"):
        return int(size_str[:-1]) * 1_000_000
    elif size_str.endswith("B"):
        return int(size_str[:-1]) * 1_000_000_000
    else:
        return float("inf")
