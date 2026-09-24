# The image the Forgejo CI runs the test suite in (.forgejo/workflows/ci.yml).
#
# julia 1.11.9 is the version Manifest.toml was resolved with; a different
# minor re-resolves the environment and tests something other than what ships.
# The official image is multi-arch (linux/amd64 and linux/arm64), so the same
# tag serves both CI architectures.
#
# build-essential is for Expect.jl, the one package in the dependency graph
# that runs `make` when it builds. Everything heavy -- Cairo for CairoMakie,
# NetCDF/HDF5 for NCDatasets, Enzyme's LLVM -- ships as _jll binary artifacts
# and needs nothing from the system.
FROM julia:1.11.9

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
