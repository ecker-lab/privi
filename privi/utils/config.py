def parse_value(val: str):
    """Try to convert a string to int, then float, then bool, otherwise leave as str."""
    # int
    try:
        return int(val)
    except ValueError:
        pass
    # float
    try:
        return float(val)
    except ValueError:
        pass
    # bool
    low = val.lower()
    if low in ("true", "false"):
        return low == "true"
    # fallback
    return val

def apply_overrides(cfg: dict, overrides: list[tuple[str, str]]) -> None:
    """
    Apply a list of (key_path, value_str) overrides in-place.

    - key_path: dot-separated keys, e.g. "data.path.train"
    - value_str: string; will be coerced to int/float/bool/str

    Raises:
        KeyError: if any part of the path doesn’t exist in cfg.
    """
    for key_path, val_str in overrides:
        parts = key_path.split(".")
        sub = cfg
        # traverse to the parent of the leaf
        for p in parts[:-1]:
            if p not in sub or not isinstance(sub[p], dict):
                raise KeyError(f"No such nested key: {'.'.join(parts[:parts.index(p)+1])}")
            sub = sub[p]
        leaf = parts[-1]
        if leaf not in sub:
            raise KeyError(f"No such key to override: {key_path}")
        # parse and set
        sub[leaf] = parse_value(val_str)