"""Tests for the wrong-charge-partner rule of tapir.rates and data/ionization_energies.csv.
Usage: pytest  (or, without pytest: python -m tests.test_ie_filter)"""
import csv
from tests._compat import pytest, main
from tapir.darkcloud import load_rows
from tapir.flux import kinetic_row
from tapir.generator import load_cases
from tapir.knockout import NET
from tapir.paths import IONIZATION_ENERGIES
from tapir.rates import (RateEstimator, ie_flag, ie_validation, charge_partner, ionization_energies,
                         WRONG_CHARGE_PARTNER, IE_WRONG_PARTNER_MAX, IE_WRONG_PARTNER_EXCLUDE)

ROWS24 = load_rows(NET)


def rx(lab):
    reac, prod = lab.split(" -> ")
    return dict(reac=reac.split(" + "), prod=prod.split(" + "))


def test_table_rows_have_a_nist_source():
    with open(IONIZATION_ENERGIES) as fh:
        rows = list(csv.DictReader(l for l in fh if not l.startswith("#")))
    assert len(rows) >= 50
    for r in rows:
        assert r["url"].startswith("https://webbook.nist.gov/cgi/cbook.cgi?ID=") and r["cas"], r
        assert float(r["ie_eV"]) > 0 and r["method"]
    assert len({r["species"] for r in rows}) == len(rows)


@pytest.mark.parametrize("lab", ("c-C3H2 + HCO+ -> CH2CCH + CO+", "C2H4+ + CO -> C2H4 + CO+"))
def test_flagged(lab):
    assert ie_flag(rx(lab)) == WRONG_CHARGE_PARTNER


@pytest.mark.parametrize("hidden", [h for _, h in load_cases()])
def test_hidden_reactions_of_benchmark_b_not_flagged(hidden):
    assert ie_flag(hidden) is None


def test_n2_h3plus_not_flagged():
    assert ie_flag(rx("N2 + H3+ -> H2 + N2H+")) is None


def test_applies_only_with_both_ie():
    ie = {"CO": 14.0}                                       # IE(CH2CCH) missing: not applicable
    assert charge_partner(rx("c-C3H2 + HCO+ -> CH2CCH + CO+"), ie) is None
    ie = {"CO": 14.0, "CH2CCH": 13.5}                       # within the 1 eV margin
    assert ie_flag(rx("c-C3H2 + HCO+ -> CH2CCH + CO+"), ie) is None


def test_validation_decides_flag_only():
    """The decision in tapir.rates: the rule fails on >= 2 % of fast KIDA ion-neutral reactions,
    so candidates are flagged, not excluded."""
    n, flagged = ie_validation(ROWS24)
    assert n > 1000
    assert len(flagged) / n >= IE_WRONG_PARTNER_MAX
    assert IE_WRONG_PARTNER_EXCLUDE is False


def test_flag_only_keeps_rate():
    c = rx("c-C3H2 + HCO+ -> CH2CCH + CO+")
    assert RateEstimator(ROWS24).estimate(c) == (kinetic_row(c), "capture_guess")


if __name__ == "__main__":
    main(__file__, globals())
