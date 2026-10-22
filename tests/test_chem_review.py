"""Tests for results/discovery/chem_review.json (chemical review of the discovery hypotheses).
Usage: pytest  (or, without pytest: python -m tests.test_chem_review)"""
import json, os, re
from tests._compat import pytest, main
from tapir.paths import RESULTS

DIR = os.path.join(RESULTS, "discovery")
REVIEW = os.path.join(DIR, "chem_review.json")
VERDICTS = ("implausible", "plausible", "uncertain")
FILES = ("CH2CCH", "CH2CCH_upstream")
DOI_RE = re.compile(r"10\.\d{4,9}/\S+")


def review():
    return json.load(open(REVIEW))


def test_loads_with_reviewer_and_summary():
    d = review()
    assert d["reviewer"].startswith("chemical review by the team chemist")
    assert d["summary"].startswith("No candidate survives chemical review.")
    assert d["reactions"]


def test_fields():
    for r in review()["reactions"]:
        assert set(r) == {"reaction", "file", "verdict", "reason", "refs"}, r
        assert r["verdict"] in VERDICTS and r["file"] in FILES and r["reason"]
        assert isinstance(r["refs"], list) and all(DOI_RE.fullmatch(x) for x in r["refs"]), r["refs"]


def test_reactions_exist_in_their_discovery_json():
    hyps = {f: {h["reaction"] for h in json.load(open(os.path.join(DIR, f + ".json")))["hypotheses"]} for f in FILES}
    missing = [(r["file"], r["reaction"]) for r in review()["reactions"] if r["reaction"] not in hyps[r["file"]]]
    assert missing == [], missing


def test_no_duplicates():
    keys = [(r["file"], r["reaction"]) for r in review()["reactions"]]
    assert len(keys) == len(set(keys))


if __name__ == "__main__":
    main(__file__, globals())
