def label_with_values(vals: dict[str, float], n_digits: int=2) -> str:
    """Makes a text label with parameter values, for example
    {'a': 42, 'b': 1.123} -> "a=42,b=1.12"
    """
    
    parts = (f"{k}={v:.{n_digits}g}" for k, v in vals.items())
    res = ", ".join(parts)
    return res
