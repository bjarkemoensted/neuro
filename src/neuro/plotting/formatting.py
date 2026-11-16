def label_with_values(vals: dict[object, object], n_digits: int=2) -> str:
    """Makes a text label with parameter values, for example
    {'a': 42, 'b': 1.123} -> "a=42,b=1.12"
    """

    formatted = (
        (
            str(k),
            f"{v:.{n_digits}g}" if isinstance(v, float) else str(v)
        )
        for k, v in vals.items()
    )
    parts = (f"{k}={v}" for k, v in formatted)
    res = ", ".join(parts)
    return res
