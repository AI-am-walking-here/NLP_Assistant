"""Project-local Python startup customizations.

Keep this file narrow: it is imported automatically when ``PYTHONPATH=src``.
"""

from __future__ import annotations

import warnings


warnings.filterwarnings(
    "ignore",
    message=(
        r"_check_is_size will be removed in a future PyTorch release "
        r"along with guard_size_oblivious\.\s+Use _check\(i >= 0\) instead\."
    ),
    category=FutureWarning,
)
