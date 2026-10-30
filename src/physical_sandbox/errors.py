"""Errors for bad input (a wrong name, an episode that is not there): the ``psbx`` command reports these with
exit code 2. Each also is the built-in type it replaced, so code that caught that type still works."""


class InputError(ValueError):
    """The caller gave something that cannot be used."""


class InputKeyError(InputError, KeyError):
    """A name that is not there (a feature, a camera)."""

    def __str__(self):  # KeyError would show the message in quotes
        return str(self.args[0]) if self.args else ""


class InputIndexError(InputError, IndexError):
    """A number out of range (an episode)."""
