# Notes for AI coding assistants

This repository is research code for winding counts and phase-field surfaces in PHerc. Paris 4. Most scripts in `src/` need data that is not in the repository (CT, surface prediction and Lasagna arrays, several GB) and have hard-coded paths (`docs/PATHS.md`). Do not try to run the whole pipeline.

Two checks run on a laptop CPU with only numpy and torch:

1. `python examples/count_windings.py` downloads the 3 MB weights of the winding-count network from Hugging Face and prints the network's count and confidence for 9 sample point pairs with known counts (expected: 7 of 9 right).
2. `python src/compare_pcu.py` downloads the published per-pair rows of the PCU certificate from its repository and compares them with `results/pairs_oof.npz` (our per-pair predictions). Expected first lines: `PCU pairs 3966; matched to our pairs 3625` and, for 1 winding, PCU 70.7% coverage at 99.58% precision against 86.8% at 99.75% for the combined certifier.

If the user wants to run more, read `README.md` (terms, results, limits) and `docs/PATHS.md` first, and ask which box or slab they have data for.
