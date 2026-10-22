"""Repository paths. Outputs go to results/ and benchmark/ unless TAPIR_RESULTS or
TAPIR_BENCHMARK point elsewhere (used to verify scripts without touching the stored results)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NETWORK_DIR = os.path.join(ROOT, "networks")
NET_2014 = os.path.join(NETWORK_DIR, "kida.uva.2014", "kida.uva.2014.dat")
NET_2024 = os.path.join(NETWORK_DIR, "kida.uva.2024", "gas_reactions_kida.uva.2024.in")
SPECIES_2014 = os.path.join(NETWORK_DIR, "kida.uva.2014", "gas_species.dat")
SPECIES_2024 = os.path.join(NETWORK_DIR, "kida.uva.2024", "gas_species_kida.uva.2024.in")
NETWORKS = {"2014": NET_2014, "2024": NET_2024}

RESULTS = os.environ.get("TAPIR_RESULTS", os.path.join(ROOT, "results"))
BENCHMARK = os.environ.get("TAPIR_BENCHMARK", os.path.join(ROOT, "benchmark"))
CASES = os.path.join(BENCHMARK, "cases.json")
DATA_DIR = os.path.join(ROOT, "data")
IONIZATION_ENERGIES = os.path.join(DATA_DIR, "ionization_energies.csv")


def result(name):
    """Path of an output file in RESULTS (created if needed)."""
    os.makedirs(RESULTS, exist_ok=True)
    return os.path.join(RESULTS, name)


def relative(path):
    """Path relative to the repository root, for files written to disk."""
    return os.path.relpath(path, ROOT)
