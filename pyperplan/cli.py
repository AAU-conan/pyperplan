import argparse
from typing import Optional


_cli_register = {}


class CLIClassConstructor:
    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        # If any of the args were given as classes instead of instances, we need to instantiate them here with no cli_arguments
        self.cli_args = (o() if isinstance(o, CLIClassWrapper) else o for o in args)
        self.cli_kwargs = {k: (v() if isinstance(v, CLIClassWrapper) else v) for k, v in kwargs.items()}

    def __call__(self, *args, **kwargs):

        return self.cls(*args, *self.cli_args, **kwargs, **self.cli_kwargs)


class CLIClassWrapper:
    def __init__(self, cls):
        self.cls = cls

    def __call__(self, *args, **kwargs):
        return CLIClassConstructor(self.cls, *args, **kwargs)


def cli_register(cli_alias: Optional[str] = None):
    """
    A decorator to register a class as a CLI command.
    """
    if cli_alias is not None and not isinstance(cli_alias, str):
        raise ValueError("Looks like you forgot the parentheses on @cli_register()")

    def decorator(cls):
        assert cls.__name__ not in _cli_register, f"Class {cls.__name__} is already registered."
        _cli_register[cls.__name__] = CLIClassWrapper(cls)
        if cli_alias is not None:
            assert cli_alias not in _cli_register, f"CLI alias {cli_alias} is already registered."
            _cli_register[cli_alias] = CLIClassWrapper(cls)
        return cls

    return decorator


def cli_constructor(expected_type: type, allow_none: bool = True):
    """
    This function returns a function that constructs the object specified by the command line argument.

    The argument is a string, e.g. "DominancePruning(database=AllPreviousDatabase(),do_stuff=True)"
    We then evaluate this string as a python expression.
    """

    def constructor(arg: str):
        try:
            obj = eval(arg, {"__builtins__": {}}, _cli_register)  # type: ignore
        except NameError as e:
            raise argparse.ArgumentTypeError(f"Unknown name {e.name}, available names: {list(_cli_register.keys())}")
        except Exception as e:
            raise argparse.ArgumentTypeError(f"Could not parse argument: {arg}, error {e}")
        if obj is None:
            if allow_none:
                return None
            else:
                raise argparse.ArgumentTypeError(f"Expected type {expected_type.__name__}, got None")
        if isinstance(obj, CLIClassWrapper):
            # The class itself was given, so we need to instantiate it without arguments
            obj = obj()

        # Check that it is of the expected type
        if not issubclass(obj.cls, expected_type):
            raise argparse.ArgumentTypeError(f"Expected type {expected_type.__name__}, got {type(obj.cls).__name__}")
        return obj

    return constructor
