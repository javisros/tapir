"""
generator_llm.py - LLM generator of candidate reactions, with the interface of Generator.candidates.

The LLM only PROPOSES. Leak-proofing, closed world and novelty are those of generator.Generator,
built with the same arguments: the prompt is made from its network after hiding (hidden reaction and
strict equivalents removed), and every proposal is filtered with its species, keys and hidden set.
Limits: at most MAX_RAW proposals are processed (safety limit against oversized answers); all of
them are filtered, and the first n surviving candidates in the LLM order are kept (scientific limit,
the number asked for).
Direction: needed_effect "increase" (net formation of the target), "decrease" (net destruction) or
None (either), given by the caller; it sets the prompt and is checked on delta_target =
count in products - count in reactants.
Discards are counted by reason in `last["discarded"]` (first failing reason, in this order):
  unknown_species         a species outside the network after hiding (CR, CRP, Photon, grains included)
  not_conserved           chem.conserves fails
  null_reaction           same multiset of reactants and products (canonical key)
  no_target               the target is on neither side
  no_net_change           delta_target == 0
  requires_stabilization  two or more reactants and a single product (association; radiative
                          association is excluded, three-body is negligible in a dark cloud)
  hidden                  strict equivalent of a hidden reaction (only if drop_hidden)
  already_in_network      strict equivalent of a reaction of the network after hiding (only if novelty)
  wrong_direction         valid and new, but delta_target has the sign opposite to needed_effect
  duplicate               same canonical key as an earlier accepted proposal
Score: rank among the kept candidates (first = highest), i.e. the LLM order. It is the generator's
priority, NOT the scientific ranking that the simulation produces later.

Context: the main formation and destruction reactions of the target, passed by the caller
(benchmarks: knockout_demo.budget on the network after hiding). Without it, a FALLBACK ranks the
network reactions by rate coefficient at 10 K, for smoke/manual tests only; benchmarks build the
generator with allow_fallback_context=False.

Answers are cached in results/llm_cache/<sha256(model, reasoning effort, schema, prompt)>.json.
"""
import hashlib, json, os
from collections import Counter
from .chem import canonical_key, conserves
from .darkcloud import rate, W24_GAS
from .generator import Generator, FAMILIES
from .knockout import label
from .paths import RESULTS

N_CANDIDATES = 30                                 # scientific limit: valid candidates kept
MAX_RAW = 60                                      # safety limit: raw proposals processed
N_CONTEXT = 5                                     # formation / destruction reactions in the prompt
CLASSES = ("ion-neutral", "dissociative recombination", "neutral-neutral")
REASONS = ("unknown_species", "not_conserved", "null_reaction", "no_target", "no_net_change",
           "requires_stabilization", "hidden", "already_in_network", "wrong_direction", "duplicate")
EFFECTS = {"increase": ("increase the abundance of {t} (net formation)",
                        "{t} appears more times among products than among reactants"),
           "decrease": ("decrease the abundance of {t} (net destruction)",
                        "{t} appears more times among reactants than among products"),
           None: ("change the abundance of {t} (net formation or destruction)",
                  "the number of {t} differs between reactants and products")}
CACHE_DIR = os.path.join(RESULTS, "llm_cache")

SCHEMA = dict(
    type="object", additionalProperties=False, required=["candidates"],
    properties=dict(candidates=dict(type="array", items=dict(
        type="object", additionalProperties=False,
        required=["reactants", "products", "reaction_class", "rationale"],
        properties=dict(reactants=dict(type="array", items=dict(type="string")),
                        products=dict(type="array", items=dict(type="string")),
                        reaction_class=dict(type="string"),     # CLASSES, checked by validate
                        rationale=dict(type="string"))))))


def build_prompt(target, species, formation, destruction, n=N_CANDIDATES, needed_effect=None):
    """species: allowed names; formation/destruction: reaction labels, most important first;
    needed_effect: a key of EFFECTS."""
    lines = lambda rs: "\n".join(f"  {r}" for r in rs) or "  (none)"
    goal, check = (x.format(t=target) for x in EFFECTS[needed_effect])
    return f"""Dark molecular cloud, T = 10 K, gas phase only. Target species: {target}

Propose up to {n} valid new reactions that {goal}, most plausible first.
Only barrierless channels fast at 10 K: ion-neutral, dissociative recombination (ion + e- ->
two or more fragments), radical-radical or radical-neutral without barrier.
Before returning each candidate, check that:
- atoms and charge are conserved;
- {check} (no null reactions);
- there are at least two products (no association to a single product, no radiative association);
- every species is in the allowed list (no photons, cosmic rays or grains);
- it is not one of the network reactions listed below.
Return fewer candidates rather than invalid ones.

Allowed species (exact names, electron = e-):
{" ".join(species)}

Main formation reactions in the network:
{lines(formation)}
Main destruction reactions in the network:
{lines(destruction)}

Answer only with JSON:
{{"candidates": [{{"reactants": [...], "products": [...], "reaction_class": "{'" | "'.join(CLASSES)}", "rationale": "one short sentence"}}]}}"""


def validate(answer):
    """The list of proposals, or ValueError if the answer does not follow SCHEMA."""
    cands = answer.get("candidates") if isinstance(answer, dict) else None
    if not isinstance(cands, list) or not cands:
        raise ValueError("no 'candidates' list")
    for c in cands:
        if not (isinstance(c, dict)
                and all(isinstance(c.get(k), list) and c[k]
                        and all(isinstance(s, str) for s in c[k]) for k in ("reactants", "products"))
                and c.get("reaction_class") in CLASSES and isinstance(c.get("rationale"), str)):
            raise ValueError(f"malformed candidate: {c!r}"[:300])
    return cands


def parse_label(s):
    """Inverse of knockout.label (and of the names in knockout_demo.budget)."""
    reac, prod = s.split(" -> ")
    return dict(reac=reac.split(" + "), prod=prod.split(" + "))


class LLMGenerator:
    def __init__(self, network_rows, extra_sources=(), hidden=(), novelty=True, families=FAMILIES,
                 client=None, n=N_CANDIDATES, cache_dir=CACHE_DIR, drop_hidden=True,
                 allow_fallback_context=True):
        """Same arguments as Generator (extra_sources and families only affect the analogy
        templates, not the LLM). client: object with .model and .ask(prompt, schema, validate);
        by default llm_client.LLMClient()."""
        self.base = Generator(network_rows, extra_sources, hidden, novelty, families)
        self.species = sorted(s for s in self.base.species if not s.startswith("GRAIN"))
        self.allowed = set(self.species) | {"e-"}
        if client is None:
            from .llm_client import LLMClient
            client = LLMClient()
        self.client, self.n, self.cache_dir, self.drop_hidden = client, n, cache_dir, drop_hidden
        self.allow_fallback_context = allow_fallback_context
        self.last = None

    def fallback_context(self, target, n=N_CONTEXT):
        """FALLBACK for smoke/manual tests, not for benchmarks: the two-body gas-phase formation
        and destruction reactions of the target with the largest rate coefficient at 10 K, from
        the network after hiding. ([labels], [labels])"""
        two_body = [r for r in self.base.net if len(r["reac"]) == 2
                    and all(s in self.allowed for s in r["reac"] + r["prod"])]
        k = lambda r: rate(r, W24_GAS["T"], W24_GAS["zeta"], W24_GAS["Av"], 1.0)
        top = lambda rs: [label(r) for r in sorted(rs, key=k, reverse=True)[:n]]
        return (top([r for r in two_body if target in r["prod"]]),
                top([r for r in two_body if target in r["reac"]]))

    def prompt(self, target, context=None, needed_effect=None):
        """context: (formation, destruction) labels, most important first; None = fallback.
        needed_effect: "increase", "decrease" or None (see the module docstring)."""
        if needed_effect not in EFFECTS:
            raise ValueError(f"needed_effect must be one of {list(EFFECTS)}: {needed_effect!r}")
        if context is None:
            if not self.allow_fallback_context:
                raise ValueError("context required (allow_fallback_context=False)")
            context = self.fallback_context(target)
        leak = [s for s in context[0] + context[1]          # budget() also names a non-reaction
                if " -> " in s and canonical_key(parse_label(s)) in self.base.hidden_keys]
        if leak:
            raise ValueError(f"hidden reaction in the context: {leak}")
        return build_prompt(target, self.species, *context, n=self.n, needed_effect=needed_effect)

    def _ask(self, prompt):
        """(proposals, cached)"""
        effort = getattr(self.client, "reasoning_effort", None)
        h = hashlib.sha256(json.dumps([self.client.model, effort, SCHEMA, prompt]).encode()).hexdigest()
        path = os.path.join(self.cache_dir, h + ".json")
        if os.path.exists(path):
            return json.load(open(path))["candidates"], True
        before = dict(self.client.stats)
        cands = self.client.ask(prompt, SCHEMA, validate)
        usage = {k: v - before.get(k, 0) for k, v in self.client.stats.items()}
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(dict(model=self.client.model, reasoning_effort=effort, prompt=prompt,
                           usage=usage, candidates=cands), f, indent=1)
        return cands, False

    def candidates(self, target, context=None, needed_effect=None):
        """Filtered LLM candidates for `target`, in the LLM order (score descending).
        context, needed_effect: see prompt()."""
        proposals, cached = self._ask(self.prompt(target, context, needed_effect))
        cands = self.filter(target, proposals, needed_effect)
        self.last["cached"] = cached
        return cands

    def filter(self, target, proposals, needed_effect=None):
        """First n candidates surviving the filters, from the first MAX_RAW validated proposals;
        counts in self.last."""
        processed = proposals[:MAX_RAW]
        out, discarded, rejected = {}, Counter({r: 0 for r in REASONS}), []
        for p in processed:
            row = dict(reac=[s.strip() for s in p["reactants"]], prod=[s.strip() for s in p["products"]])
            k = canonical_key(row)
            if not all(s in self.allowed for s in row["reac"] + row["prod"]):
                why = "unknown_species"
            elif not conserves(row):
                why = "not_conserved"
            elif k[0] == k[1]:
                why = "null_reaction"
            elif target not in row["reac"] + row["prod"]:
                why = "no_target"
            elif (delta := row["prod"].count(target) - row["reac"].count(target)) == 0:
                why = "no_net_change"
            elif len(row["reac"]) >= 2 and len(row["prod"]) == 1:
                why = "requires_stabilization"
            elif self.drop_hidden and k in self.base.hidden_keys:
                why = "hidden"
            elif self.base.novelty and k in self.base.net_keys:
                why = "already_in_network"
            elif ((needed_effect == "increase" and delta < 0)
                  or (needed_effect == "decrease" and delta > 0)):
                why = "wrong_direction"
            elif k in out:
                why = "duplicate"
            else:
                out[k] = dict(row, id=-1, family="LLM", source="llm",
                              reaction_class=p["reaction_class"], rationale=p["rationale"])
                continue
            discarded[why] += 1
            rejected.append((why, row))
        kept = list(out.values())[:self.n]
        for i, c in enumerate(kept):
            c["score"] = float(len(kept) - i)    # generator priority, not the simulation ranking
            c["fam"], c["derivations"] = {"LLM": c["score"]}, [("LLM", c["reaction_class"], 0, 0)]
        self.last = dict(target=target, cached=None, needed_effect=needed_effect,
                         returned_raw=len(proposals), processed_raw=len(processed),
                         raw_truncated=len(proposals) > MAX_RAW,
                         valid_before_limit=len(out), considered=len(kept), discarded=discarded,
                         rejected=rejected)
        return kept
