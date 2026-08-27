"""RTO Sentinel - a cost-calibrated return-to-origin risk scorer for Indian COD commerce.

Layering (see ARCHITECTURE.md for the full contract)::

    data -> features -> models -> decision -> api -> console
                                     |
                                     +--> agents (downstream, optional, never authoritative)

The one rule that governs everything else: the ML model produces a calibrated
probability, the deterministic decision engine converts that probability into an
action using an explicit rupee cost model, and the LLM layer only ever describes
a decision that has already been made.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
