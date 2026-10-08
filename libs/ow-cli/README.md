# `ow-cli`

The `occamworm` command-line front end. Command names and paths in the spec are acceptance contracts.

**Invariant:** Test labels are never reachable from any command that searches or fits.

Spec: [§14.5](../../docs/engineering/ARCHITECTURE.md)

Status: the `ow` binary provides `rule inspect|canon`, `sim run|conformance` and `search enumerate` ([WRL_SYNTAX.md](../../docs/language/WRL_SYNTAX.md) §11). The remaining commands of the spec are not implemented. Build: [REPRODUCIBILITY.md](../../docs/engineering/REPRODUCIBILITY.md#local-build-and-test).
