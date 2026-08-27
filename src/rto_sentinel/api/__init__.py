"""The HTTP layer.

Route handlers marshal and delegate. They do not compute risk, they do not choose
thresholds, and they do not decide policy - all of that lives in
``rto_sentinel.decision`` and is reached through a dependency.

``tests/architecture/test_layering.py`` asserts that no router imports an ML
library directly, so "no ML logic in a route handler" is a checked property
rather than a convention that erodes.
"""

from rto_sentinel.api.main import create_app

__all__ = ["create_app"]
