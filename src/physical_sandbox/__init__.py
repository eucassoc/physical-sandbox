"""physical-sandbox: overlay robots on a real site at real scale.

In a notebook, ``compare(folder, a, b)`` gives the gaps between two motions on a work folder the stage screen
left (see ``physical_sandbox.notebook``). It is loaded on first use, so importing the package stays light."""

__version__ = "0.1.3"

__all__ = ["__version__", "compare", "Gaps"]


def __getattr__(name: str):
    if name in ("compare", "Gaps"):
        from . import notebook

        return getattr(notebook, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
