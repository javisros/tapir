"""Tests for tapir.literature and scripts.check_literature with a fake Bond and a fake LLM (no call to
Valency or LiteLLM). Usage: pytest  (or, without pytest: python -m tests.test_literature)"""
import json, math, os, sys, tempfile, threading, types
from http.server import BaseHTTPRequestHandler, HTTPServer
from tests._compat import pytest, main
from tapir import literature as L
from tapir.literature import (Literature, BondClient, BondError, LeakError, guard, key_str, kida_comparison,
                              uncertainty_factor, NO_EVIDENCE)
from scripts import check_literature as CL

DOI_A, DOI_B = "10.1063/1.4772651", "10.1016/S0009-2614(99)00380-3"
BOND_TEXT = (f"Chastaing et al. measured C + C2H4 between 15 and 295 K: k = 4.3 x 10^-10 cm3 s-1 at 15 K. "
             f"doi:{DOI_B}\nModel of TMC-1 including C + C2H4 -> CH2CCH + H. https://doi.org/{DOI_A}")
C_C2H4 = dict(reaction="C + C2H4 -> H + CH2CCH", reac=["C", "C2H4"], prod=["H", "CH2CCH"])


def discovery(n=3):
    hyps = [dict(rank=i + 1, reaction=f"A{i} + B -> CH2CCH + C{i}", reac=[f"A{i}", "B"], prod=["CH2CCH", f"C{i}"],
                 literature=None) for i in range(n)]
    existing = [dict(C_C2H4, kida_id=3491, k_kida=3.933e-10, F=1.4),
                dict(reaction="H + CH2CCH -> CH3CCH + Photon", reac=["H", "CH2CCH"], prod=["CH3CCH", "Photon"],
                     kida_id=6495, k_kida=1.643e-11, F=10.0)]
    return dict(species="CH2CCH", mode="formation", search="direct", X_obs=8.7e-9, hypotheses=hyps,
                existing=existing, disclaimer="Ranked hypotheses, not confirmed reactions.")


class FakeBond:
    """search(query, n) -> {"text", "refs"} with the refs extracted as BondClient does; fail=True raises."""
    def __init__(self, text=BOND_TEXT, fail=False):
        self.text, self.fail, self.queries = text, fail, []

    def search(self, query, n):
        self.queries.append(query)
        if self.fail:
            raise BondError("connection refused")
        return dict(text=self.text, refs=L.find_refs(self.text))


class FakeLLM:
    """Same interface as llm_client.LLMClient; returns `answer` (validated) and counts calls."""
    model = "fake"

    def __init__(self, answer):
        self.answer, self.calls = answer, 0

    def ask(self, prompt, schema, validate):
        self.calls += 1
        return validate(json.loads(json.dumps(self.answer)))


def ref(r, same_channel=True):
    return dict(ref=r, same_channel=same_channel)


def answer(**kw):
    a = dict(status="measured", refs=[ref(f"https://doi.org/{DOI_B}")], k_lit=4.3e-10, k_lit_T=15,
             k_lit_ref=f"doi:{DOI_B}", k_lit_quote="k = 4.3 x 10^-10 cm3 s-1 at 15 K", note="Measured down to 15 K.")
    a.update(kw)
    return a


def lit(bond=None, llm=None, cache=None, **kw):
    cache = cache or os.path.join(tempfile.mkdtemp(), "literature_cache.json")
    return Literature(bond=bond or FakeBond(), client=llm or FakeLLM(answer()), cache_path=cache, **kw)


# --- guard --------------------------------------------------------------------------------------

@pytest.mark.parametrize("module", ("scripts.benchmark_b", "scripts.benchmark_a", "scripts.bench_llm"))
def test_guard_refuses_benchmark(module):
    bond, llm = FakeBond(), FakeLLM(answer())
    src = os.path.join(tempfile.mkdtemp(), "CH2CCH.json")
    json.dump(discovery(), open(src, "w"))
    sys.modules[module] = types.ModuleType(module)
    try:
        with pytest.raises(LeakError):
            lit(bond, llm).check_reaction(C_C2H4, discovery())
        with pytest.raises(LeakError):
            CL.main([src], literature=lit(bond, llm))
    finally:
        del sys.modules[module]
    assert bond.queries == [] and llm.calls == 0


@pytest.mark.parametrize("data", ({"cases": []}, [1, 2], {**discovery(), "hypotheses": "x"},
                                  {k: v for k, v in discovery().items() if k != "X_obs"}))
def test_guard_refuses_non_discovery(data):
    bond = FakeBond()
    with pytest.raises(ValueError):
        lit(bond).check_reaction(C_C2H4, data)
    assert bond.queries == []


def test_guard_accepts_stored_discovery():
    path = os.path.join(os.path.dirname(L.CACHE), "discovery", "CH2CCH.json")
    if os.path.exists(path):
        guard(json.load(open(path)))
    guard(discovery())


# --- classification and anti-invention ---------------------------------------------------------

def test_query_in_plain_words():
    assert L.query(C_C2H4) == ("gas-phase reaction of atomic carbon (C) with ethylene (C2H4) producing atomic "
                               "hydrogen (H) and propargyl radical (CH2CCH): rate coefficient at low temperature")
    assert L.query(dict(reac=["H", "CH2CCH"], prod=["CH3CCH", "Photon"])).startswith(
        "radiative association of atomic hydrogen (H) with propargyl radical (CH2CCH) forming propyne")
    assert "->" not in L.query(dict(reac=["C3H5+", "e-"], prod=["CH2CCH", "H", "H"]))


def test_short_query_uses_generic_formulas():
    assert L.queries(C_C2H4)[1] == "C + C2H4 reaction rate constant (atomic carbon with ethylene)"
    assert L.queries(dict(reac=["H", "CH2CCH"], prod=["CH3CCH", "Photon"]))[1].startswith("H + C3H3 reaction")


def test_effort_fallback_on_format_error():
    """FormatError at reasoning effort "high" -> one retry at "medium", then back to "high"."""
    class FormatError(Exception):
        pass

    class Flaky(FakeLLM):
        reasoning_effort = "high"
        efforts = []

        def ask(self, prompt, schema, validate):
            self.efforts.append(self.reasoning_effort)
            if self.reasoning_effort == "high":
                raise FormatError("json_validate_failed")
            return super().ask(prompt, schema, validate)
    llm = Flaky(answer())
    x = lit(llm=llm)
    assert x.check_reaction(C_C2H4, discovery())["status"] == "measured"
    assert llm.efforts == ["high", "medium"] and llm.reasoning_effort == "high" and x.stats["effort_fallbacks"] == 1


def test_measured_with_verified_k():
    e = lit().check_reaction(C_C2H4, discovery())
    assert e["status"] == "measured" and e["refs"] == [ref(DOI_B.lower())]
    assert e["k_lit"] == 4.3e-10 and e["k_lit_T"] == 15 and e["discarded_refs"] == []
    assert e["k_lit_ref"] == DOI_B.lower() and e["k_lit_same_channel"] is True


def test_refs_not_from_bond_are_discarded():
    invented = "10.1086/999999"
    refs = [ref(invented), ref(DOI_A), ref("https://example.org/paper")]
    e = lit(llm=FakeLLM(answer(refs=refs, k_lit_ref=invented))).check_reaction(C_C2H4, discovery())
    assert e["refs"] == [ref(DOI_A)] and e["discarded_refs"] == [invented, "https://example.org/paper"]
    assert e["k_lit"] is None and e["k_lit_discarded"] == 4.3e-10        # its reference was not from Bond


def test_status_without_bond_ref_becomes_no_evidence():
    e = lit(llm=FakeLLM(answer(refs=[ref("10.1086/999999")]))).check_reaction(C_C2H4, discovery())
    assert e["status"] == NO_EVIDENCE and e["refs"] == [] and e["unverified_status"] == "measured"


@pytest.mark.parametrize("status", ("measured", "computed", "proposed"))
def test_other_channel_only_is_not_evidence(status):
    """Same reactants, other products: the reference is kept, flagged, and does not count."""
    e = lit(llm=FakeLLM(answer(status=status, refs=[ref(DOI_B, False)]))).check_reaction(C_C2H4, discovery())
    assert e["status"] == NO_EVIDENCE and e["unverified_status"] == status
    assert e["refs"] == [ref(DOI_B.lower(), False)] and e["k_lit_same_channel"] is False


def test_one_same_channel_ref_is_enough():
    e = lit(llm=FakeLLM(answer(refs=[ref(DOI_B, False), ref(DOI_A)]))).check_reaction(C_C2H4, discovery())
    assert e["status"] == "measured" and e["refs"] == [ref(DOI_B.lower(), False), ref(DOI_A)]


@pytest.mark.parametrize("k, quote", ((4.3e-10, "k = 4.3e-10 at 15 K"),          # not verbatim
                                      (5.0e-10, "k = 4.3 x 10^-10 cm3 s-1 at 15 K")))  # number not in the quote
def test_k_lit_not_in_bond_text_is_discarded(k, quote):
    e = lit(llm=FakeLLM(answer(k_lit=k, k_lit_quote=quote))).check_reaction(C_C2H4, discovery())
    assert e["k_lit"] is None and e["k_lit_discarded"] == k and e["status"] == "measured"


def test_no_refs_from_bond_gives_none_without_llm():
    llm = FakeLLM(answer())
    e = lit(FakeBond(text="No results."), llm).check_reaction(C_C2H4, discovery())
    assert e["status"] == NO_EVIDENCE and e["refs"] == [] and llm.calls == 0


def test_bond_failure_gives_unchecked_and_is_not_cached():
    cache = os.path.join(tempfile.mkdtemp(), "c.json")
    llm = FakeLLM(answer())
    x = lit(FakeBond(fail=True), llm, cache)
    e = x.check_reaction(C_C2H4, discovery())
    q = x.check_question("Q?", discovery())
    assert e["status"] == "unchecked" and e["refs"] == [] and e["k_lit"] is None and "Bond" in e["error"]
    assert q["status"] == "unchecked" and q["articles"] == []
    assert llm.calls == 0 and x.cache == dict(reactions={}, questions={})
    e2 = lit(FakeBond(), FakeLLM(answer()), cache).check_reaction(C_C2H4, discovery())    # retried later
    assert e2["status"] == "measured"


def test_llm_failure_gives_unchecked():
    class Broken(FakeLLM):
        def ask(self, *a):
            raise RuntimeError("Rate limit reached in organization `org_01abcXYZ`")
    e = lit(llm=Broken(None)).check_reaction(C_C2H4, discovery())
    assert e["status"] == "unchecked" and "org_01abcXYZ" not in e["error"] and "org_<redacted>" in e["error"]


# --- cache ---------------------------------------------------------------------------------------

def test_cache_by_canonical_key():
    cache = os.path.join(tempfile.mkdtemp(), "c.json")
    bond, llm = FakeBond(), FakeLLM(answer())
    x = lit(bond, llm, cache)
    e = x.check_reaction(C_C2H4, discovery())
    swapped = dict(reac=["C2H4", "C"], prod=["CH2CCH", "H"])          # same canonical key
    assert x.check_reaction(swapped, discovery()) == e and len(bond.queries) == 1 and llm.calls == 1
    x.save()
    assert key_str(C_C2H4) in json.load(open(cache))["reactions"]
    bond2, llm2 = FakeBond(), FakeLLM(answer())
    assert lit(bond2, llm2, cache).check_reaction(C_C2H4, discovery()) == e
    assert bond2.queries == [] and llm2.calls == 0
    lit(bond2, llm2, cache, refresh=True).check_reaction(C_C2H4, discovery())
    assert len(bond2.queries) == 1


# --- context questions ---------------------------------------------------------------------------

def test_question_articles_only_from_bond():
    arts = dict(articles=[dict(ref=f"doi:{DOI_A}", finding="Gas-phase model of TMC-1.", stance="NEUTRAL"),
                          dict(ref="10.1086/999999", finding="Invented.", stance="SUPPORTS"),
                          dict(ref=DOI_A, finding="Duplicate.", stance="NEUTRAL")])
    q = lit(llm=FakeLLM(arts)).check_question(CL.CONTEXT_QUESTIONS["CH2CCH"][1], discovery())
    assert q["status"] == "found" and q["articles"] == [dict(ref=DOI_A, finding="Gas-phase model of TMC-1.",
                                                           stance="NEUTRAL")]
    assert q["discarded_refs"] == ["10.1086/999999"]
    invented = dict(articles=[dict(ref="10.1086/999999", finding="Invented.", stance="SUPPORTS")])
    q = lit(llm=FakeLLM(invented)).check_question(CL.CONTEXT_QUESTIONS["CH2CCH"][1], discovery())
    assert q["status"] == NO_EVIDENCE and q["articles"] == []


# --- script --------------------------------------------------------------------------------------

# --- comparison with KIDA ------------------------------------------------------------------------

def klit(k=4.3e-10, T=15, same_channel=True):
    return dict(k_lit=k, k_lit_T=T, k_lit_same_channel=same_channel)


def test_kida_comparison_c_c2h4():
    """C + C2H4 -> H + CH2CCH (KIDA 3491): F0 = 1.4, g = 0, k_KIDA(15 K) = 3.82e-10, read from the network."""
    c = kida_comparison(klit(), 3491)
    assert c["result"] == "compatible" and (c["F0"], c["g"]) == (1.4, 0.0) and c["F_T"] == 1.4
    assert abs(c["k_kida_T"] - 3.1e-10 * (15 / 300) ** -0.07) < 1e-15 and c["extrapolated"] is False
    assert kida_comparison(klit(k=1e-9), 3491)["result"] == "incompatible"


@pytest.mark.parametrize("entry, reason", ((klit(T=298), "> 50 K"), (klit(T=None), "no temperature"),
                                           (klit(same_channel=False), "same channel"), (klit(k=None), "no k_lit")))
def test_kida_not_comparable(entry, reason):
    c = kida_comparison(entry, 3491)
    assert c["result"] == "not_comparable" and reason in c["reason"]


def test_kida_factor_depends_on_temperature():
    """F(T) = F0 exp(|g (1/T - 1/300)|): with g = 50, a factor 100 is within F(10 K) (~250) but not F0 = 2."""
    assert uncertainty_factor(2.0, 0.0, 10) == 2.0 and uncertainty_factor(2.0, 50.0, 300) == 2.0
    assert abs(uncertainty_factor(2.0, -50.0, 10) - 2.0 * math.exp(50 * (0.1 - 1 / 300))) < 1e-9
    row = dict(reac=["X", "Y"], prod=["Z", "W"], A=1e-10, B=0.0, C=0.0, itype=4, tmin=10.0, tmax=300.0,
               formula=3, id=1)
    L._KIDA["fake.dat"] = {1: [(row, 2.0, 50.0)]}
    try:
        assert kida_comparison(klit(k=1e-8, T=10), 1, "fake.dat")["result"] == "compatible"
        L._KIDA["fake.dat"] = {1: [(row, 2.0, 0.0)]}
        assert kida_comparison(klit(k=1e-8, T=10), 1, "fake.dat")["result"] == "incompatible"
    finally:
        del L._KIDA["fake.dat"]


def test_script_three_uses():
    d = tempfile.mkdtemp()
    src, out = os.path.join(d, "in.json"), os.path.join(d, "out.json")
    json.dump(discovery(12), open(src, "w"))
    x = lit(cache=os.path.join(d, "c.json"))
    CL.main([src, "--top", "10", "--out", out, "--cache", x.cache_path], literature=x)
    res = json.load(open(out))
    assert sum(h["literature"] is not None for h in res["hypotheses"]) == 10
    lits = {e["reaction"]: e.get("literature") for e in res["existing"]}
    assert lits["C + C2H4 -> H + CH2CCH"]["kida_comparison"]["result"] == "compatible"
    assert len(res["literature_context"]["questions"]) == 3
    assert json.load(open(src))["hypotheses"][0]["literature"] is None          # input untouched with --out
    assert os.path.exists(x.cache_path)


# --- MCP client against a local fake Bond server -------------------------------------------------

class FakeMCP(BaseHTTPRequestHandler):
    token, seen = "secret", []

    def log_message(self, *a):
        pass

    def do_POST(self):
        msg = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeMCP.seen.append((msg["method"], self.headers.get("Mcp-Session-Id"), msg.get("params")))
        if self.headers.get("Authorization") != f"Bearer {self.token}":
            self.send_response(401); self.end_headers(); return
        if "id" not in msg:
            self.send_response(202); self.end_headers(); return
        result = {"initialize": dict(protocolVersion="2025-06-18", capabilities={}, serverInfo=dict(name="bond")),
                  "tools/list": dict(tools=[dict(name="get_paper", inputSchema={}),
                                            dict(name="search_papers", inputSchema=dict(
                                                type="object", required=["query"],
                                                properties=dict(query=dict(type="string"),
                                                                max_results=dict(type="integer"))))]),
                  "tools/call": dict(content=[dict(type="text", text=BOND_TEXT)], isError=False)}[msg["method"]]
        body = f"event: message\ndata: {json.dumps(dict(jsonrpc='2.0', id=msg['id'], result=result))}\n\n".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Mcp-Session-Id", "s-1")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def fake_server():
    srv = HTTPServer(("127.0.0.1", 0), FakeMCP)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/mcp"


def test_bond_client_mcp():
    srv, url = fake_server()
    FakeMCP.seen = []
    try:
        res = BondClient(url, "secret", timeout=5).search("C + C2H4", 5)
    finally:
        srv.shutdown()
    assert res["refs"] == [DOI_B.lower(), DOI_A] and "Chastaing" in res["text"]
    methods = [m for m, _, _ in FakeMCP.seen]
    assert methods == ["initialize", "notifications/initialized", "tools/list", "tools/call"]
    assert FakeMCP.seen[-1][1] == "s-1"                                       # session id sent back
    assert FakeMCP.seen[-1][2] == dict(name="search_papers", arguments=dict(query="C + C2H4", max_results=5))


def test_bond_papers_compact_with_titles():
    """semantic_search_papers is preferred; its papers become compact text, titles come from Bond."""
    papers = dict(papers=[dict(title="Low-T kinetics of C + C2H4", doi=DOI_B, url="https://arxiv.org/abs/9901.00001",
                               authors=["A", "B", "C", "D"], first_submitted="1999-01-01",
                               abstract="k = 4.3 x 10^-10 cm3 s-1 at 15 K."),
                          dict(title="No identifiers", abstract="...")])
    calls = []

    class Fake(BondClient):
        def _post(self, method, params=None, notify=False):
            calls.append((method, params))
            if method == "tools/list":
                return dict(tools=[dict(name="search_by_title"), dict(name="semantic_search_papers",
                            inputSchema=dict(properties=dict(query={}, limit={})))])
            if method == "tools/call":
                return dict(content=[dict(type="text", text=json.dumps(papers))], structuredContent=papers)
            return {}
    res = Fake("http://x", "t").search(["q", "q2"], 4)
    assert calls[-1][1] == dict(name="semantic_search_papers", arguments=dict(query="q2", limit=4))
    assert res["refs"] == [DOI_B.lower(), "https://arxiv.org/abs/9901.00001"]
    assert res["titles"][DOI_B.lower()] == "Low-T kinetics of C + C2H4" and res["text"].count("Low-T kinetics") == 1
    assert "A, B, C et al. (1999)" in res["text"]
    e = lit(FakeBond(text=res["text"])).check_reaction(C_C2H4, discovery())
    assert e["status"] == "measured" and e["k_lit"] == 4.3e-10
    bond = FakeBond(text=res["text"]); bond.search = lambda q, n: res
    assert lit(bond).check_reaction(C_C2H4, discovery())["refs"][0]["title"] == "Low-T kinetics of C + C2H4"


def test_bond_client_bad_token_gives_unchecked():
    srv, url = fake_server()
    try:
        with pytest.raises(BondError):
            BondClient(url, "wrong", timeout=5).search("C + C2H4", 5)
        e = lit(BondClient(url, "wrong", timeout=5)).check_reaction(C_C2H4, discovery())
    finally:
        srv.shutdown()
    assert e["status"] == "unchecked" and "401" in e["error"]


def test_bond_client_requires_configuration():
    with pytest.raises(BondError):
        BondClient(None, "x")
    with pytest.raises(BondError):
        BondClient("http://127.0.0.1:9/mcp", "")


if __name__ == "__main__":
    main(__file__, globals())
