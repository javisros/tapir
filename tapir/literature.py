"""
literature.py - Literature check of discovery results with Valency MCP Bond (T3).

Bond SEARCHES, the LLM only CLASSIFIES what Bond returned:
  reaction  -> literature = {status, refs: [{ref, same_channel}], k_lit, k_lit_T, k_lit_ref,
               k_lit_same_channel, k_lit_quote, note, ...}
               status: measured | proposed | computed | no_evidence_found (Bond found nothing usable,
               which does not mean that nothing exists), or "unchecked" if Bond (or the LLM) failed
               same_channel: the reference is about these reactants AND these products (a generic
               formula of the same composition counts, e.g. C3H3 for CH2CCH); same reactants with other
               products, or a total rate without products, is not the same channel
  question  -> up to MAX_ARTICLES articles {ref, finding, stance: SUPPORTS | CONTRADICTS | NEUTRAL}

Anti-invention rules, applied in code after the LLM answer (never left to the prompt):
  - a reference is kept only if it is a DOI or URL found in the text Bond returned; the others are
    dropped and counted (discarded_refs);
  - k_lit is kept only if k_lit_quote is a verbatim excerpt of Bond's text and contains that number,
    and k_lit_ref is one of the kept references (k_lit_same_channel is that reference's same_channel);
  - status measured/proposed/computed needs at least one kept reference with same_channel = true,
    otherwise it becomes "no_evidence_found" (the LLM's status is kept in unverified_status): a
    measurement of other products of the same reactants does not count as "measured";
  - Bond returns no reference at all -> status "no_evidence_found" without asking the LLM.
Failures are never cached ("unchecked" is retried on the next run).

Discovery mode only: guard() refuses to run when a benchmark module is loaded (scripts.benchmark_*,
scripts.bench_llm: Bond would see the hidden reactions) or when the input is not a discovery JSON.

Configuration (.env or environment):
  BOND_MCP_URL       Streamable HTTP endpoint of the Bond MCP server
  BOND_TOKEN         bearer token (VALENCY_TOKEN accepted as an alias)
  BOND_SEARCH_TOOL   tool to call (default: the first tool whose name contains "search")
The LLM is tapir.llm_client.LLMClient (its own LITELLM_* / TAPIR_LLM_* variables).

KIDA comparison (kida_comparison): k_lit against KIDA at the temperature T of k_lit, with the
temperature-dependent uncertainty factor of KIDA, F(T) = F0 * exp(|g * (1/T - 1/300)|), F0 and g read
from the network file (discovery.kida_uncertainty) -> "compatible" if |log10(k_lit / k_KIDA(T))| <=
log10 F(T), "incompatible" otherwise, "not_comparable" if there is no k_lit, it is not for the same
channel, it has no temperature or T > T_LOW.

Cache: results/literature_cache.json, {"reactions": {canonical key: entry}, "questions": {text: entry}}.
"""
import datetime, json, math, os, re, sys, urllib.error, urllib.request
from .chem import canonical_key
from .darkcloud import parse_kida, rate, W24_GAS
from .discovery import kida_uncertainty, row_key
from .knockout import NET, label
from .paths import RESULTS

CACHE = os.path.join(RESULTS, "literature_cache.json")
NO_EVIDENCE = "no_evidence_found"
STATUSES = ("measured", "proposed", "computed", NO_EVIDENCE)
COMPARISONS = ("compatible", "incompatible", "not_comparable")
STANCES = ("SUPPORTS", "CONTRADICTS", "NEUTRAL")
MAX_REFS = 3                                       # refs per reaction
MAX_ARTICLES = 5                                   # articles per context question
N_RESULTS = 8                                      # results asked to Bond per search
MAX_TEXT = 15000                                   # characters of Bond's answer shown to the LLM
K_RTOL = 0.02                                      # k_lit must match a number of the quote within 2 %
T_LOW = 50.0                                       # K: a k_lit above this is not compared with KIDA
BENCHMARK_PREFIXES = ("scripts.benchmark_", "scripts.bench_llm")
DISCOVERY_KEYS = ("species", "mode", "search", "X_obs", "hypotheses", "disclaimer")
GRAIN_HYPOTHESIS = ("C2H4 (ethylene) and therefore CH2CCH (propargyl radical) in TMC-1 are formed, at least "
                    "partly, by chemistry on grain surfaces, which a gas-phase-only model does not include.")
# Common names for the Bond queries (KIDA notation is not what papers use); missing species keep the formula.
NAMES = {"CH2CCH": "propargyl radical", "C2H4": "ethylene", "CH3CCH": "propyne (methylacetylene)",
         "CH2CCH2": "allene", "c-C3H2": "cyclopropenylidene", "l-C3H2": "propadienylidene",
         "C": "atomic carbon", "H": "atomic hydrogen", "O": "atomic oxygen", "N": "atomic nitrogen",
         "CCH": "ethynyl radical", "CH3": "methyl radical", "C2H2": "acetylene", "C4H2": "diacetylene",
         "HCO+": "formyl cation", "H3+": "H3+", "HCO": "formyl radical", "HCCCHO": "propynal",
         "Photon": "photon (radiative association)", "e-": "electron (dissociative recombination)"}


class LeakError(RuntimeError):
    pass


class BondError(RuntimeError):
    pass


def guard(data):
    """Raises LeakError if a benchmark is loaded, ValueError if data is not a discovery JSON."""
    loaded = sorted(m for m in sys.modules if m.startswith(BENCHMARK_PREFIXES))
    if loaded:
        raise LeakError(f"literature check refused: benchmark modules loaded {loaded} (discovery mode only)")
    missing = [k for k in DISCOVERY_KEYS if not isinstance(data, dict) or k not in data]
    if missing or not isinstance(data["hypotheses"], list):
        raise ValueError(f"not a discovery JSON (missing {missing or ['hypotheses list']})")


def key_str(row):
    """Canonical key as text: sorted reactants -> sorted products."""
    reac, prod = canonical_key(row)
    return " + ".join(reac) + " -> " + " + ".join(prod)


def words(row):
    """'C + C2H4 -> H + CH2CCH' -> 'atomic carbon + ethylene -> atomic hydrogen + propargyl radical'"""
    name = lambda s: NAMES.get(s, s)
    return " + ".join(map(name, row["reac"])) + " -> " + " + ".join(map(name, row["prod"]))


# --- references and numbers found in Bond's text ------------------------------------------------

REF_CHARS = r"[^\s\"'<>\[\]{},;]+"
DOI_RE = re.compile(r"10\.\d{4,9}/" + REF_CHARS)
URL_RE = re.compile(r"https?://" + REF_CHARS)
REF_RE = re.compile(rf"{URL_RE.pattern}|{DOI_RE.pattern}")
DOI_PREFIXES = ("doi:", "https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/")
SUPERSCRIPT = str.maketrans("⁻⁺⁰¹²³⁴⁵⁶⁷⁸⁹−–", "-+0123456789--")
NUM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:(?:[x×*·]|\\times)\s*10\s*(?:\^|\*\*)?\s*\{?\s*([-+]?\s*\d+)\s*\}?"
                    r"|[eE]\s*([-+]?\d+)|\(\s*(-\d+)\s*\))")


def clean(s):
    """Without trailing punctuation or an unbalanced closing parenthesis (DOIs may contain '(...)')."""
    s = s.strip().rstrip(".,:")
    while s.endswith(")") and s.count("(") < s.count(")"):
        s = s[:-1].rstrip(".,:")
    return s


def norm_ref(s):
    """DOI (lower case, without doi: or doi.org prefix) or URL; None if s is neither."""
    s = clean(s)
    for p in DOI_PREFIXES:
        if s.lower().startswith(p):
            s = s[len(p):].strip()
            break
    if DOI_RE.fullmatch(s):
        return s.lower()
    return s if URL_RE.fullmatch(s) else None


def find_refs(text):
    """DOIs and URLs in Bond's text, in order of appearance, normalised and without duplicates."""
    out = []
    for m in REF_RE.finditer(text):
        r = norm_ref(m.group(0))
        if r and r not in out:
            out.append(r)
    return out


def numbers(text):
    """Values in scientific notation in text: 1.2e-10, 1.2 x 10^-10, 1.2×10⁻¹⁰, 1.2(-10)."""
    out = []
    for m in NUM_RE.finditer(text.translate(SUPERSCRIPT)):
        exp = next(g for g in m.groups()[1:] if g is not None)
        out.append(float(m.group(1)) * 10 ** int(exp.replace(" ", "")))
    return out


def squash(s):
    return " ".join(s.translate(SUPERSCRIPT).split())


def verified_k(k, quote, text):
    """True if quote is a verbatim excerpt of text and contains k (within K_RTOL)."""
    if not (isinstance(k, (int, float)) and k > 0 and quote and squash(quote) in squash(text)):
        return False
    return any(abs(v - k) <= K_RTOL * k for v in numbers(quote))


# --- Bond MCP client ----------------------------------------------------------------------------

class BondClient:
    """Minimal MCP client (Streamable HTTP, JSON-RPC 2.0) with a bearer token. search(query, n) ->
    {"text", "refs"}; any failure raises BondError."""
    PROTOCOL = "2025-06-18"

    def __init__(self, url, token, tool=None, timeout=60):
        if not url or not token:
            raise BondError("BOND_MCP_URL and BOND_TOKEN are required")
        self.url, self.token, self.tool_name, self.timeout = url, token, tool, timeout
        self.session, self.tool, self._id = None, None, 0

    @classmethod
    def from_env(cls):
        from dotenv import load_dotenv
        load_dotenv()
        return cls(os.getenv("BOND_MCP_URL"), os.getenv("BOND_TOKEN") or os.getenv("VALENCY_TOKEN"),
                   os.getenv("BOND_SEARCH_TOOL"))

    def _post(self, method, params=None, notify=False):
        msg = dict(jsonrpc="2.0", method=method, **({"params": params} if params is not None else {}))
        if not notify:
            self._id += 1
            msg["id"] = self._id
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                   "Authorization": f"Bearer {self.token}"}
        if self.session:
            headers.update({"Mcp-Session-Id": self.session, "MCP-Protocol-Version": self.PROTOCOL})
        req = urllib.request.Request(self.url, json.dumps(msg).encode(), headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                self.session = resp.headers.get("Mcp-Session-Id") or self.session
                body, ctype = resp.read().decode("utf-8", "replace"), resp.headers.get("Content-Type", "")
        except (urllib.error.URLError, OSError, ValueError) as e:      # HTTPError is a URLError
            raise BondError(f"{method}: {e}") from e
        if notify:
            return None
        try:
            if "text/event-stream" in ctype:      # SSE: the JSON-RPC answer is in a data: line
                msgs = [json.loads(ln[5:]) for ln in body.splitlines() if ln.startswith("data:") and ln[5:].strip()]
            else:
                msgs = [json.loads(body)] if body.strip() else []
        except ValueError as e:
            raise BondError(f"{method}: invalid JSON ({e})") from e
        answer = next((m for m in msgs if m.get("id") == msg["id"]), None)
        if answer is None:
            raise BondError(f"{method}: no answer")
        if "error" in answer:
            raise BondError(f"{method}: {answer['error']}")
        return answer["result"]

    def connect(self):
        if self.tool:
            return
        self._post("initialize", dict(protocolVersion=self.PROTOCOL, capabilities={},
                                      clientInfo=dict(name="tapir", version="0.1.0")))
        self._post("notifications/initialized", notify=True)
        tools = self._post("tools/list").get("tools", [])
        pick = ([t for t in tools if t["name"] == self.tool_name] if self.tool_name
                else [t for t in tools if "search" in t["name"].lower()] or tools)
        if not pick:
            raise BondError(f"no tool {self.tool_name or 'search'} among {[t['name'] for t in tools]}")
        self.tool = pick[0]

    def arguments(self, query, n):
        """Arguments for the tool from its input schema: the query and, if there is one, the limit."""
        schema = self.tool.get("inputSchema") or {}
        props, required = schema.get("properties", {}), schema.get("required", [])
        qk = next((k for k in ("query", "q", "question", "search", "text", "prompt") if k in props),
                  next((k for k in required if props.get(k, {}).get("type") == "string"), "query"))
        args = {qk: query}
        nk = next((k for k in ("limit", "max_results", "top_k", "num_results", "k", "n", "count", "size")
                   if k in props), None)
        if nk:
            args[nk] = n
        return args

    def search(self, query, n=N_RESULTS):
        self.connect()
        res = self._post("tools/call", dict(name=self.tool["name"], arguments=self.arguments(query, n)))
        text = "\n".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
        if res.get("structuredContent") is not None:
            text += "\n" + json.dumps(res["structuredContent"], ensure_ascii=False)
        if res.get("isError"):
            raise BondError(f"tool error: {text[:300]}")
        return dict(text=text, refs=find_refs(text))


# --- LLM classification -------------------------------------------------------------------------

NULLABLE_NUM = dict(type=["number", "null"])
REACTION_SCHEMA = dict(
    type="object", additionalProperties=False,
    required=["status", "refs", "k_lit", "k_lit_T", "k_lit_ref", "k_lit_quote", "note"],
    properties=dict(status=dict(type="string", enum=list(STATUSES)),
                    refs=dict(type="array", items=dict(
                        type="object", additionalProperties=False, required=["ref", "same_channel"],
                        properties=dict(ref=dict(type="string"), same_channel=dict(type="boolean")))),
                    k_lit=NULLABLE_NUM, k_lit_T=NULLABLE_NUM, k_lit_ref=dict(type="string"),
                    k_lit_quote=dict(type="string"), note=dict(type="string")))
ARTICLES_SCHEMA = dict(
    type="object", additionalProperties=False, required=["articles"],
    properties=dict(articles=dict(type="array", items=dict(
        type="object", additionalProperties=False, required=["ref", "finding", "stance"],
        properties=dict(ref=dict(type="string"), finding=dict(type="string"),
                        stance=dict(type="string", enum=list(STANCES)))))))

RULES = """Use ONLY the search results below, never your own knowledge. If they do not answer, say so.
References: copy DOIs or URLs exactly as they appear in the search results."""


def reaction_prompt(row, text):
    return f"""You classify literature search results about one gas-phase reaction.
{RULES}

Reaction (KIDA notation): {label(row)}
In words: {words(row)}
Question: has the rate coefficient of this reaction been measured or computed, ideally at low
temperature (10-50 K, dark interstellar clouds)?

Answer with JSON:
- status: "measured" if a result reports a laboratory measurement of the rate coefficient of this
  channel; "computed" if a result reports a theoretical calculation of it; "proposed" if a result only
  proposes or uses this channel (e.g. in an astrochemical model or network); "{NO_EVIDENCE}" if no result
  is about this channel.
- refs: up to {MAX_REFS} results that justify the status, each with
  ref: its DOI or URL; same_channel: true only if it is about these reactants AND these products
  (products may be written in words or as a formula of the same composition, e.g. C3H3 for CH2CCH);
  false if it gives other products for the same reactants, or only a total rate without products.
  [] if none.
- k_lit: rate coefficient in cm3 s-1 at the lowest temperature reported, only if the number is written
  in the results; otherwise null.
- k_lit_T: temperature in K of k_lit, or null.
- k_lit_ref: the DOI or URL of the result that gives k_lit (one of refs); "" if k_lit is null.
- k_lit_quote: the exact text of the results that contains k_lit, copied verbatim; "" if k_lit is null.
- note: one short sentence on what the results say about this reaction.

Search results:
{text[:MAX_TEXT]}"""


def question_prompt(question, text):
    return f"""You classify literature search results.
{RULES}

Question: {question}
Hypothesis: {GRAIN_HYPOTHESIS}

For each distinct article in the results (at most {MAX_ARTICLES}, most relevant first) give:
- ref: its DOI or URL, copied from the results (skip articles without one);
- finding: one sentence with what the article says that is relevant to the question;
- stance towards the hypothesis: "SUPPORTS" (it proposes or finds grain-surface formation of C2H4 or
  CH2CCH, or attributes missing CH2CCH to absent grain chemistry), "CONTRADICTS" (it explains CH2CCH in
  TMC-1 with gas-phase chemistry alone, or finds grain routes inefficient) or "NEUTRAL" (otherwise).
Answer with JSON: {{"articles": [{{"ref": "...", "finding": "...", "stance": "SUPPORTS|CONTRADICTS|NEUTRAL"}}]}}

Search results:
{text[:MAX_TEXT]}"""


def validate_reaction(a):
    if not (isinstance(a, dict) and a.get("status") in STATUSES and isinstance(a.get("refs"), list)
            and all(isinstance(r, dict) and isinstance(r.get("ref"), str) and isinstance(r.get("same_channel"), bool)
                    for r in a["refs"])
            and all(isinstance(a.get(k), str) for k in ("k_lit_ref", "k_lit_quote", "note"))
            and all(a.get(k) is None or isinstance(a[k], (int, float)) for k in ("k_lit", "k_lit_T"))):
        raise ValueError(f"malformed reaction answer: {a!r}"[:300])
    return a


def validate_articles(a):
    arts = a.get("articles") if isinstance(a, dict) else None
    if not isinstance(arts, list) or not all(
            isinstance(x, dict) and isinstance(x.get("ref"), str) and isinstance(x.get("finding"), str)
            and x.get("stance") in STANCES for x in arts):
        raise ValueError(f"malformed articles answer: {a!r}"[:300])
    return arts


def keep_bond_refs(refs, allowed):
    """refs: [{ref, same_channel}] -> (those found in Bond's answer, normalised and without duplicates;
    the refs that are not)"""
    kept, dropped = {}, []
    for r in refs:
        n = norm_ref(r["ref"])
        if n not in allowed:
            dropped.append(r["ref"])
        elif n not in kept:
            kept[n] = dict(ref=n, same_channel=r["same_channel"])
    return list(kept.values()), dropped


_KIDA = {}


def kida_rows(path=NET):
    """{KIDA id: [(row, F0, g)]} of a kida.uva file (all temperature ranges), read once."""
    if path not in _KIDA:
        unc, out = kida_uncertainty(path), {}
        for r in parse_kida(path):
            out.setdefault(r["id"], []).append((r, *unc.get(row_key(r), (None, None))))
        _KIDA[path] = out
    return _KIDA[path]


def uncertainty_factor(F0, g, T):
    """KIDA's temperature-dependent uncertainty factor F(T) = F0 * exp(|g * (1/T - 1/300)|)."""
    return F0 * math.exp(abs(g * (1.0 / T - 1.0 / 300.0)))


def kida_comparison(lit, kida_id, path=NET):
    """{result: compatible | incompatible | not_comparable, reason, T, k_lit, k_kida_T, F0, g, F_T,
    log10_ratio, extrapolated} for the literature entry `lit` of the KIDA reaction kida_id."""
    k, T = lit.get("k_lit"), lit.get("k_lit_T")
    out = dict(result="not_comparable", reason=None, T=T, k_lit=k, k_kida_T=None, F0=None, g=None, F_T=None,
               log10_ratio=None, extrapolated=None)
    rows = kida_rows(path).get(kida_id, [])
    if not k:
        out["reason"] = "no k_lit"
    elif not lit.get("k_lit_same_channel"):
        out["reason"] = "k_lit is not for the same channel (same products)"
    elif T is None:
        out["reason"] = "no temperature for k_lit"
    elif T > T_LOW:
        out["reason"] = f"k_lit at {T:g} K > {T_LOW:g} K"
    elif not rows:
        out["reason"] = f"KIDA id {kida_id} not in {os.path.basename(path)}"
    if out["reason"]:
        return out
    inside = [x for x in rows if x[0]["tmin"] <= T <= x[0]["tmax"]]
    row, F0, g = (inside or sorted(rows, key=lambda x: x[0]["tmin"]))[0]
    if not F0 or F0 <= 0 or g is None:
        out["reason"] = "no KIDA uncertainty factor"
        return out
    k_kida = float(rate(row, T, W24_GAS["zeta"], W24_GAS["Av"], 1.0))
    F_T, ratio = uncertainty_factor(F0, g, T), math.log10(k / k_kida)
    out.update(result="compatible" if abs(ratio) <= math.log10(F_T) + 1e-12 else "incompatible",
               k_kida_T=k_kida, F0=F0, g=g, F_T=F_T, log10_ratio=ratio, extrapolated=not inside)
    return out


def today():
    return datetime.date.today().isoformat()


class Literature:
    """bond: object with search(query, n) -> {"text", "refs"} (default BondClient.from_env());
    client: object with .ask(prompt, schema, validate) (default llm_client.LLMClient(max_calls))."""

    def __init__(self, bond=None, client=None, cache_path=CACHE, refresh=False, max_calls=None):
        self._bond, self._client, self.max_calls = bond, client, max_calls
        self.cache_path, self.refresh = cache_path, refresh
        self.cache = dict(reactions={}, questions={})
        if os.path.exists(cache_path):
            self.cache.update(json.load(open(cache_path)))
        self.stats = dict(bond_calls=0, llm_calls=0, cached=0, unchecked=0, discarded_refs=0)

    @property
    def bond(self):
        if self._bond is None:
            self._bond = BondClient.from_env()
        return self._bond

    @property
    def client(self):
        if self._client is None:
            from .llm_client import LLMClient
            self._client = LLMClient(**({"max_calls": self.max_calls} if self.max_calls else {}))
        return self._client

    def save(self):
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        with open(self.cache_path, "w") as fh:
            json.dump(self.cache, fh, indent=1, ensure_ascii=False)

    def _search(self, query):
        """Bond's answer, or (None, error) if Bond fails."""
        try:
            self.stats["bond_calls"] += 1
            return self.bond.search(query, N_RESULTS), None
        except Exception as e:                       # BondError, missing configuration, bad JSON...
            return None, f"Bond: {type(e).__name__}: {e}"[:300]

    def _ask(self, prompt, schema, validate):
        """LLM answer, or (None, error)."""
        try:
            self.stats["llm_calls"] += 1
            return self.client.ask(prompt, schema, validate), None
        except Exception as e:                       # FormatError, BudgetExceeded, connection...
            return None, f"LLM: {type(e).__name__}: {e}"[:300]

    def _unchecked(self, query, error):
        self.stats["unchecked"] += 1
        return dict(status="unchecked", refs=[], k_lit=None, k_lit_T=None, k_lit_ref=None,
                    k_lit_same_channel=None, k_lit_quote="", note="",
                    backend="valency-bond", query=query, checked=today(), error=error)

    def check_reaction(self, row, data):
        """literature entry for one reaction (dict with reac, prod) of the discovery JSON `data`."""
        guard(data)
        key = key_str(row)
        if not self.refresh and key in self.cache["reactions"]:
            self.stats["cached"] += 1
            return self.cache["reactions"][key]
        query = f"{label(row)} ({words(row)}) reaction rate coefficient low temperature"
        found, err = self._search(query)
        if err:
            return self._unchecked(query, err)
        entry = dict(status=NO_EVIDENCE, refs=[], k_lit=None, k_lit_T=None, k_lit_ref=None,
                     k_lit_same_channel=None, k_lit_quote="", note="", backend="valency-bond", query=query,
                     checked=today(), error=None, n_bond_refs=len(found["refs"]), discarded_refs=[])
        if found["refs"]:
            a, err = self._ask(reaction_prompt(row, found["text"]), REACTION_SCHEMA, validate_reaction)
            if err:
                return self._unchecked(query, err)
            refs, dropped = keep_bond_refs(a["refs"], set(found["refs"]))
            refs = refs[:MAX_REFS]
            entry.update(status=a["status"], refs=refs, note=a["note"], discarded_refs=dropped)
            k_ref = next((r for r in refs if r["ref"] == norm_ref(a["k_lit_ref"] or "")), None)
            if k_ref and verified_k(a["k_lit"], a["k_lit_quote"], found["text"]):
                entry.update(k_lit=float(a["k_lit"]), k_lit_T=a["k_lit_T"], k_lit_ref=k_ref["ref"],
                             k_lit_same_channel=k_ref["same_channel"], k_lit_quote=a["k_lit_quote"])
            elif a["k_lit"] is not None:
                entry["k_lit_discarded"] = a["k_lit"]
            if entry["status"] != NO_EVIDENCE and not any(r["same_channel"] for r in refs):
                entry.update(status=NO_EVIDENCE, unverified_status=a["status"])
            self.stats["discarded_refs"] += len(dropped)
        else:
            entry["note"] = "Bond returned no DOI or URL for this query."
        self.cache["reactions"][key] = entry
        return entry

    def check_question(self, question, data):
        """{question, status (found | no_evidence_found | unchecked), articles: [{ref, finding, stance}], ...} for one context question."""
        guard(data)
        if not self.refresh and question in self.cache["questions"]:
            self.stats["cached"] += 1
            return self.cache["questions"][question]
        found, err = self._search(question)
        if err:
            self.stats["unchecked"] += 1
            return dict(question=question, status="unchecked", articles=[], checked=today(), error=err)
        entry = dict(question=question, status=NO_EVIDENCE, articles=[], backend="valency-bond", checked=today(),
                     error=None, n_bond_refs=len(found["refs"]), discarded_refs=[])
        if found["refs"]:
            arts, err = self._ask(question_prompt(question, found["text"]), ARTICLES_SCHEMA, validate_articles)
            if err:
                self.stats["unchecked"] += 1
                return dict(question=question, status="unchecked", articles=[], checked=today(), error=err)
            allowed, seen = set(found["refs"]), set()
            for x in arts:
                r = norm_ref(x["ref"])
                if r in allowed and r not in seen:
                    seen.add(r)
                    entry["articles"].append(dict(ref=r, finding=x["finding"], stance=x["stance"]))
                elif r not in allowed:
                    entry["discarded_refs"].append(x["ref"])
            entry["articles"] = entry["articles"][:MAX_ARTICLES]
            entry["status"] = "found" if entry["articles"] else NO_EVIDENCE
            self.stats["discarded_refs"] += len(entry["discarded_refs"])
        self.cache["questions"][question] = entry
        return entry
