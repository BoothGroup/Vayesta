try:
    import pygnme
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("pygnme is required for variational embedding approaches.") from e
