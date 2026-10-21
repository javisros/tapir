"""Tests for tapir.generator_llm and tapir.llm_client with fake clients (no call to LiteLLM).
Usage: pytest  (or, without pytest: python -m tests.test_generator_llm)"""
import json, os, tempfile, types
import httpx2, openai
from collections import Counter
from tests._compat import pytest, main
from tapir import chem
from tapir.darkcloud import load_rows
from tapir.knockout import NET
from tapir.generator import load_cases
from tapir.generator_llm import LLMGenerator, REASONS, SCHEMA, validate
from tapir.llm_client import LLMClient, BudgetExceeded, FormatError

ROWS24 = load_rows(NET)
T = "H2CS"
HIDDEN = dict(load_cases())[T]                    # S + CH3 -> H + H2CS


class FakeClient:
    """Same interface as LLMClient; returns `proposals` and counts calls."""
    model = "fake"

    def __init__(self, proposals):
        self.proposals, self.stats = proposals, Counter(calls=0)

    def ask(self, prompt, schema, validate):
        self.stats["calls"] += 1; self.prompt = prompt
        return validate(dict(candidates=self.proposals))


def p(reac, prod, cls="neutral-neutral"):
    return dict(reactants=reac, products=prod, reaction_class=cls, rationale="test")


def gen(proposals):
    return LLMGenerator(ROWS24, hidden=[HIDDEN], client=FakeClient(proposals),
                        cache_dir=tempfile.mkdtemp())


def run(proposals):
    g = gen(proposals)
    return g, g.candidates(T)


# a valid novel candidate: the best analogy candidate for the same target and hidden set
VALID = gen([]).base.candidates(T)[0]
IN_NET = next(r for r in gen([]).base.net if T in r["prod"] and len(r["prod"]) > 1
              and set(r["reac"] + r["prod"]) <= gen([]).allowed)
# valid novel formations (analogy candidates for T with two or more products, hidden excluded)
FORMS = [c for c in gen([]).base.candidates(T) if len(c["prod"]) > 1 and T not in c["reac"]
         and not chem.equivalent(c, HIDDEN)]
# a valid novel destruction: an analogy candidate for another target consuming T
DESTROY = next(c for c in gen([]).base.candidates("H2CSH+")
               if T in c["reac"] and T not in c["prod"])


def test_valid_candidate_and_format():
    g, cs = run([p(VALID["reac"], VALID["prod"])])
    assert len(cs) == 1 and sum(g.last["discarded"].values()) == 0
    c = cs[0]
    assert chem.canonical_key(c) == chem.canonical_key(VALID)
    # fields used by benchmark_a/b, flux and rank, plus the LLM ones
    for k in ("reac", "prod", "id", "score", "family", "derivations", "fam"):
        assert k in c, k
    assert set(VALID) - {"score", "family", "derivations", "fam"} <= set(c)
    assert c["source"] == "llm" and c["family"] == "LLM" and c["rationale"] == "test"
    assert c["id"] == -1 and isinstance(c["score"], float)


def test_order_kept_as_score():
    g, cs = run([p(["CH3", "XX"], [T, "H"]), p(VALID["reac"], VALID["prod"]),
                 p(IN_NET["reac"], IN_NET["prod"])])
    assert [c["score"] for c in cs] == [1.0]          # rank among the kept candidates


@pytest.mark.parametrize("reason,proposal", [
    ("unknown_species", p(["XYZ", "CH3"], [T, "H"])),
    ("unknown_species", p(["S", "CH3"], [T, "Photon"])),
    ("not_conserved", p(["S", "CH4"], [T, "H"])),
    ("no_target", p(["S", "CH3"], ["HCS", "H2"])),
    ("hidden", p(["CH3", "S"], [T, "H"])),       # reordered: canonical key, not the order
    ("already_in_network", p(IN_NET["reac"][::-1], IN_NET["prod"])),
])
def test_discard_reason(reason, proposal):
    g, cs = run([proposal])
    assert cs == []
    assert g.last["discarded"] == Counter({r: int(r == reason) for r in REASONS})


def test_duplicate():
    g, cs = run([p(VALID["reac"], VALID["prod"]), p(VALID["reac"][::-1], VALID["prod"][::-1])])
    assert len(cs) == 1 and g.last["discarded"]["duplicate"] == 1


def test_hidden_kept_without_drop_hidden():
    g = LLMGenerator(ROWS24, hidden=[HIDDEN], client=FakeClient([p(["CH3", "S"], [T, "H"])]),
                     cache_dir=tempfile.mkdtemp(), drop_hidden=False)
    assert [chem.canonical_key(c) for c in g.candidates(T)] == [chem.canonical_key(HIDDEN)]


def test_prompt_without_hidden():
    prompt = gen([]).prompt(T)
    assert "S + CH3 -> H + H2CS" not in prompt and "CH3 + S" not in prompt
    assert T in prompt and "30" in prompt and "Photon" not in prompt.split("Allowed")[1]


def test_cache_avoids_second_call():
    g = gen([p(VALID["reac"], VALID["prod"])])
    first = g.candidates(T)
    assert g.client.stats["calls"] == 1 and not g.last["cached"]
    assert len(os.listdir(g.cache_dir)) == 1
    g2 = LLMGenerator(ROWS24, hidden=[HIDDEN], client=FakeClient([]), cache_dir=g.cache_dir)
    assert g2.candidates(T) == first
    assert g2.client.stats["calls"] == 0 and g2.last["cached"]


def test_validate():
    assert validate(dict(candidates=[p(["S", "CH3"], [T, "H"])]))
    for bad in ([], dict(candidates=[]), dict(candidates=[p([], [T])]),
                dict(candidates=[p(["S"], [T], cls="radiative association")])):
        with pytest.raises(ValueError):
            validate(bad)


# --- second iteration: net change, stabilization, truncation, external context -----------------

def test_valid_formation_and_destruction():
    g, cs = run([p(VALID["reac"], VALID["prod"]), p(DESTROY["reac"], DESTROY["prod"])])
    assert [chem.canonical_key(c) for c in cs] == [chem.canonical_key(VALID), chem.canonical_key(DESTROY)]
    assert sum(g.last["discarded"].values()) == 0


@pytest.mark.parametrize("reason,proposal", [
    ("no_net_change", p([T, "H2"], [T, "H", "H"])),
    ("null_reaction", p(["CH", T], [T, "CH"])),
    ("null_reaction", p(["CH", "CO"], ["CO", "CH"])),          # null even without the target
    ("requires_stabilization", p(["CH2", "S"], [T])),
    ("requires_stabilization", p(["H2CS+", "e-"], [T], "dissociative recombination")),
])
def test_trivial_and_association_discarded(reason, proposal):
    g, cs = run([proposal])
    assert cs == [] and g.last["discarded"] == Counter({r: int(r == reason) for r in REASONS})


BAD = p(["XYZ", "CH3"], [T, "H"])                  # unknown species


def test_limit_applies_after_filtering():
    g, cs = run([BAD] * 30 + [p(VALID["reac"], VALID["prod"])])      # valid proposal is raw #31
    assert [chem.canonical_key(c) for c in cs] == [chem.canonical_key(VALID)]
    assert (g.last["returned_raw"], g.last["processed_raw"], g.last["valid_before_limit"],
            g.last["considered"]) == (31, 31, 1, 1)


def test_more_than_30_valid_keeps_first_30():
    g, cs = run([BAD] + [p(c["reac"], c["prod"]) for c in FORMS[:35]])
    assert [chem.canonical_key(c) for c in cs] == [chem.canonical_key(c) for c in FORMS[:30]]
    assert [c["score"] for c in cs] == [float(30 - i) for i in range(30)]
    assert (g.last["returned_raw"], g.last["processed_raw"], g.last["valid_before_limit"],
            g.last["considered"]) == (36, 36, 35, 30)


def test_raw_safety_limit():
    g, cs = run([BAD] * 34 + [p(VALID["reac"], VALID["prod"])])      # 35 raw: all processed
    assert len(cs) == 1 and g.last["processed_raw"] == 35 and not g.last["raw_truncated"]
    g, cs = run([BAD] * 65 + [p(VALID["reac"], VALID["prod"])])      # 66 raw: only 60 processed
    assert cs == [] and g.last["returned_raw"] == 66 and g.last["processed_raw"] == 60
    assert g.last["raw_truncated"]
    assert g.last["discarded"]["unknown_species"] == 60 and g.last["valid_before_limit"] == 0


@pytest.mark.parametrize("effect,kept,wrong", [("increase", "VALID", "DESTROY"),
                                               ("decrease", "DESTROY", "VALID"),
                                               (None, "both", None)])
def test_needed_effect(effect, kept, wrong):
    g = gen([p(VALID["reac"], VALID["prod"]), p(DESTROY["reac"], DESTROY["prod"]),
             p([T, "H2"], [T, "H", "H"])])
    cs = g.candidates(T, needed_effect=effect)
    goal = dict(increase="increase the abundance", decrease="decrease the abundance")
    assert goal.get(effect, "change the abundance") in g.client.prompt
    want = [VALID, DESTROY] if kept == "both" else [globals()[kept]]
    assert [chem.canonical_key(c) for c in cs] == [chem.canonical_key(c) for c in want]
    assert g.last["discarded"]["no_net_change"] == 1
    assert g.last["discarded"]["wrong_direction"] == (0 if wrong is None else 1)
    assert g.last["needed_effect"] == effect


@pytest.mark.parametrize("reason,proposal", [
    ("already_in_network", p(IN_NET["reac"], IN_NET["prod"])),   # formation of T, in the network
    ("hidden", p(HIDDEN["reac"], HIDDEN["prod"])),                # formation of T, hidden
])
def test_reason_has_priority_over_wrong_direction(reason, proposal):
    g = gen([proposal])
    assert g.candidates(T, needed_effect="decrease") == []
    assert g.last["discarded"] == Counter({r: int(r == reason) for r in REASONS})


def test_needed_effect_invalid():
    with pytest.raises(ValueError):
        gen([]).prompt(T, needed_effect="formation")


def test_external_context():
    form, dest = ["X1 + Y1 -> Z1 + W1", "X2 + Y2 -> Z2 + W2"], ["H2 formation on grains"]
    g = gen([p(VALID["reac"], VALID["prod"])])
    g.candidates(T, context=(form, dest))
    assert all(f"  {r}\n" in g.client.prompt for r in form + dest)
    assert gen([]).prompt(T, (form, dest)) == g.client.prompt
    assert gen([]).prompt(T) != g.client.prompt                  # fallback context differs
    strict = LLMGenerator(ROWS24, hidden=[HIDDEN], client=FakeClient([]), allow_fallback_context=False)
    with pytest.raises(ValueError):
        strict.prompt(T)
    with pytest.raises(ValueError):                               # leak: hidden in the context
        strict.prompt(T, (["CH3 + S -> H2CS + H"], []))


def test_cache_key_depends_on_config():
    d = tempfile.mkdtemp()
    a = LLMGenerator(ROWS24, hidden=[HIDDEN], client=FakeClient([p(VALID["reac"], VALID["prod"])]),
                     cache_dir=d)
    a.candidates(T)
    b = LLMGenerator(ROWS24, hidden=[HIDDEN], client=FakeClient([p(VALID["reac"], VALID["prod"])]),
                     cache_dir=d)
    b.client.reasoning_effort = "high"
    b.candidates(T)
    assert b.client.stats["calls"] == 1 and len(os.listdir(d)) == 2


# --- llm_client with a fake openai SDK ---------------------------------------------------------

def fake_sdk(contents):
    """openai.OpenAI-like object answering `contents` in turn; records the requests."""
    sent = []

    def create(**kw):
        sent.append(kw)
        if isinstance(contents[len(sent) - 1], Exception):
            raise contents[len(sent) - 1]
        msg = types.SimpleNamespace(content=contents[len(sent) - 1])
        usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=usage)
    sdk = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    return sdk, sent


GOOD = json.dumps(dict(candidates=[p(["S", "CH3"], [T, "H"])]))


def test_client_retries_once_on_bad_format():
    sdk, sent = fake_sdk(["not json", GOOD])
    c = LLMClient(model="m", max_calls=5, sdk=sdk)
    assert c.ask("q", SCHEMA, validate)[0]["products"] == [T, "H"]
    assert c.stats == Counter(calls=2, format_failures=1, prompt_tokens=20, completion_tokens=10,
                              total_tokens=30)
    assert sent[0]["response_format"]["type"] == "json_schema"
    with pytest.raises(FormatError):
        LLMClient(model="m", sdk=fake_sdk(["{}", "{}"])[0]).ask("q", SCHEMA, validate)


def test_client_falls_back_to_json_object():
    req = httpx2.Request("POST", "http://fake/v1/chat/completions")
    rejected = openai.BadRequestError("json_schema not supported", body=None,
                                      response=httpx2.Response(400, request=req))
    sdk, sent = fake_sdk([rejected, GOOD, GOOD])
    c = LLMClient(model="m", sdk=sdk)
    c.ask("q", SCHEMA, validate); c.ask("q", SCHEMA, validate)
    assert [s["response_format"]["type"] for s in sent] == ["json_schema", "json_object", "json_object"]
    assert c.stats["calls"] == 3 and c.stats["format_failures"] == 0


def test_client_json_validate_failed_is_a_format_failure():
    req = httpx2.Request("POST", "http://fake/v1/chat/completions")
    failed = openai.BadRequestError("GroqException - json_validate_failed", body=None,
                                    response=httpx2.Response(400, request=req))
    sdk, sent = fake_sdk([failed, GOOD])
    c = LLMClient(model="m", sdk=sdk)
    assert c.ask("q", SCHEMA, validate)
    assert c.stats["calls"] == 2 and c.stats["format_failures"] == 1 and c.schema_ok
    assert sent[1]["response_format"]["type"] == "json_schema"


def test_client_call_cap():
    c = LLMClient(model="m", max_calls=1, sdk=fake_sdk(["{}", GOOD])[0])
    with pytest.raises(BudgetExceeded):
        c.ask("q", SCHEMA, validate)
    assert c.stats["calls"] == 1


if __name__ == "__main__":
    main(__file__, globals())
