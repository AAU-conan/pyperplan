import importlib
import pkgutil

import pyperplan


def import_submodules(package):
    """Import all submodules under a package."""
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        results[name] = importlib.import_module(name)
    return results


for loader, name, is_pkgs in pkgutil.walk_packages(pyperplan.__path__, pyperplan.__name__ + "."):
    if is_pkgs and "test" not in name and not name.startswith("pyperplan.translate"):
        import_submodules(name)
