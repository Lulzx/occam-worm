# `occamworm.embodiment`

Embodiment interfaces and a reference body (§13). `interfaces.py` holds the typed boundaries, `body.py` a planar resistive-force-theory worm, `motor.py` the fixed NMJ map and the decoder-only control, and `loop.py` the closed/open-loop runner, the randomized-neural control and the behaviour endpoints.

**Invariant:** a frozen neural program is connected only through `NeuralStepper.step`, and the motor map has no fitted parameters unless they are declared in its provenance (§13.5).

Tickets: [OW-016](../../../docs/planning/tickets/OW-016.md) · Spec: [§13](../../../docs/science/EMBODIMENT.md)
