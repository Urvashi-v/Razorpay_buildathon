"""Rung 5: rung 4 plus a text encoder over address strings.

NOT IMPLEMENTED, and disabled in ``config/models/ladder.yaml``. Deliberately not
registered in ``models/registry.py`` either, so a configuration typo fails at
lookup rather than much later.

SPEC section 05 makes this rung conditional: attempted only if time permits, and
promoted only if it beats rung 4 on **net rupees** - not on AUC. A rung that wins
a ranking metric and loses the cost metric has not earned production.

THE FAIRNESS PROBLEM IS SHARPER HERE THAN ANYWHERE ELSE ON THE LADDER
=====================================================================
An encoder over raw address text can learn regional language and transliteration
patterns. That is a protected-attribute proxy arriving by a route the refused-
feature list does not cover, because it never names a forbidden column - it
learns the same information from free text.

The existing address features avoid this by construction: they measure structural
completeness (is there a house number, does the city match the pincode) and never
fluency. A text encoder erases that distinction.

If this rung is ever built, the fairness audit gates it, and the bar is higher
than for any other rung: it must show a material rupee gain **and** no disparate
impact, because the mechanism by which it could cause harm is one this project's
other defences do not catch.
"""

from __future__ import annotations
