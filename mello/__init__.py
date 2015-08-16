"""

=======

A Python package for fitting generalized linear models

Modules
-------
experiment      - For holding relevant experimental data
features        - For generating features in the linear term
objective       - For calling the objective + gradient

For more information, see the accompanying README.md

"""

__all__ = [
    'experiment',
    'features',
    'objective'
    ]

from .experiment import *
from .features import *
from .objectives import *

__version__ = '0.0'
