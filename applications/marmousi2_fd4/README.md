# Historical A2-M9e4-FD4 lane

The unchanged Taylor-FD4 configuration describes the historical forward/data
bridge at core `f315d157c2e817480990c97a3de45c4a23e465db`.
Its CUDA preflight rejected the physical zero-mu water layer; it is historical
limitation evidence, not an accepted CUDA-fluid migration result.

The accepted physical-water result is the separate CPU `marmousi2_fluid2` case
at `f5ae2f3506f4ea8024b96d28bcbc3fc2470bb31d`. This publication copy of
`case.yaml` does not move historical data or rewrite their provenance.
See [the consolidated guide](../README.md); no CUDA fluid support or FD8/FD4
numerical equivalence is claimed.
