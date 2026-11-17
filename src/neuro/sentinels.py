class Sentinel:
    pass


class NONE(Sentinel):
    """For a parameter which wasn't passed,
    allowing 'normal' None to have some special meaning"""
    pass
