"""Allow ``python -m umbral ...`` (same as the ``umbral`` command)."""

import sys

from umbral.cli import main

sys.exit(main())
