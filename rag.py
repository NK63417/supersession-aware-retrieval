#!/usr/bin/env python3
"""
A grounded RAG with temporal-authority reranking — corpus-agnostic.

The whole pipeline in one file:

    corpus(markdown) → chunk by ## heading → embed (local, model2vec) → store (numpy)
      → embed query → rank by cosine + recency + authority → grounded generate (Claude)

Why these choices:
  • Embeddings are LOCAL (model2vec, no torch) — the documents never leave the
    machine. A private corpus stays private.
  • The "vector store" is a numpy array + brute-force cosine. For a few hundred
    chunks that's instant AND transparent — you can read the retrieval math, which
    IS the fundamental. Swap in a real vector DB only when scale demands it.
  • Generation is CITE-OR-REFUSE: answer only from retrieved chunks, cite the
    source, otherwise return the corpus's exact refusal string. That discipline is
    verify-before-act as code — what Bedrock Guardrails later automates.
  • Retrieval is TIME-AWARE: a living document set is not a bag of timeless facts.
    When a newer document supersedes an older one, the current answer wins by
    construction. Cite-or-refuse guarantees faithfulness to the retrieved context;
    it does NOT guarantee that context is still true. Ranking has to.

A CORPUS IS DATA, NOT CODE. Everything corpus-specific (root, includes, the
authority/"current state" file, the label and refusal string) lives in a profile,
so pointing this engine at a different body of documents — ops runbooks, policy
docs — is a config entry in corpora.json, not a fork.

Usage:
    python rag.py corpora                      # list profiles + index status
    python rag.py index                        # build / refresh the index
    python rag.py retrieve "..."               # top-k chunks (no API key needed)
    python rag.py ask "..."                    # grounded answer + citations
    python rag.py ask "..." --deep             # escalate to the deep model
    python rag.py --corpus runbooks ask "..."  # same engine, different corpus
"""
import os, re, glob, json, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "minishlab/potion-base-8M"            # small static embedding model

# ---- reranking weights (defaults; overridable PER CORPUS) ------------------
# Fallbacks for a corpus profile that sets none. They were grid-searched against the
# author's personal-notes corpus (not published) — best recall while holding
# current-wins at 100% — and they do NOT transfer. A corpus with a different date
# spread, or different prose, needs its own values found the same way.
#
# Why prose matters, learned the hard way on a second corpus: a SUPERSEDING document
# is often semantically SELF-PENALIZING. The text that marks it as the replacement
# ("replaces the v1 procedure", "do not restart…") is negation and meta-commentary,
# which pushes it *away* from a plain how-to query — while the naive older doc reads
# as the cleaner match. The cosine gap can therefore run the wrong way by more than
# recency can close, and that corpus needs a materially higher w_rec.
# Tune with eval.py against labelled cases; never by eye.
DEFAULT_W_REC = 0.10        # recency: freshest doc gets ~+w_rec, decaying with age
DEFAULT_W_AUTH = 0.05       # authority: the corpus's override doc gets a small bump
DEFAULT_HALFLIFE_DAYS = 21  # recency weight halves every N days

# ---- corpus profiles -------------------------------------------------------
# The engine is corpus-agnostic; a corpus is data, not code. Everything specific
# to a body of documents lives in a profile, so pointing the same pipeline at a
# different one (ops runbooks, policy docs) is a config entry — not a fork.
#
# A profile declares:
#   root      — the corpus directory
#   include   — glob(s) of content files, relative to root
#   extra     — individual files outside `include` that ARE knowledge
#   exclude   — basenames to drop (meta/navigation files, not knowledge)
#   authority — files that OVERRIDE others when they disagree (the temporal-
#               authority signal; see retrieve()). Usually the "current state" doc.
#   label     — how the corpus is described to the model in the system prompt
#   refusal   — the exact string returned when the answer isn't in the corpus
BUILTIN_CORPORA = {
    # The synthetic L2 support knowledge base (SKB) in corpora/support/: runbooks, resolved
    # tickets and schema references for a fictional billing product, with three planted
    # supersession cases. Weights grid-searched against eval.support.jsonl: best recall
    # and MRR while holding current-wins at 100%. Far stronger than the fallbacks above
    # — slower revision cadence, and its correcting docs are negation-heavy, so the
    # recency term must exceed a larger similarity penalty (0.081 and 0.156; see README).
    "support": {
        "root":          os.path.join(HERE, "corpora", "support"),
        "include":       ["**/*.md"],
        "authority":     ["CURRENT.md"],           # the current-state doc wins ties
        "label":         "the L2 application-support knowledge base "
                         "(runbooks, resolved tickets, and schema reference)",
        "refusal":       "Not in the knowledge base.",
        "w_rec":         0.40,
        "halflife_days": 120,
    },
}
# Optional user profiles (see corpora.example.json); merged over the built-ins so a
# new corpus needs no code change.
CORPORA_FILE = os.path.join(HERE, "corpora.json")


class Corpus:
    """One body of documents + the rules for reading it."""

    def __init__(self, name, spec):
        self.name      = name
        self.root      = spec["root"]
        self.include   = spec.get("include", ["**/*.md"])
        self.extra     = spec.get("extra", [])
        self.exclude   = set(spec.get("exclude", []))
        self.authority = set(spec.get("authority", []))
        self.label     = spec.get("label", f"the {name} corpus")
        self.refusal   = spec.get("refusal", "Not in the corpus.")
        # reranking weights — per-corpus; see the DEFAULT_* notes on why they don't transfer
        self.w_rec     = spec.get("w_rec", DEFAULT_W_REC)
        self.w_auth    = spec.get("w_auth", DEFAULT_W_AUTH)
        self.halflife  = spec.get("halflife_days", DEFAULT_HALFLIFE_DAYS)

    @property
    def index_dir(self):                            # per-corpus, so they coexist
        return os.path.join(HERE, ".index", self.name)

    @property
    def vec_path(self):
        return os.path.join(self.index_dir, "vectors.npy")

    @property
    def meta_path(self):
        return os.path.join(self.index_dir, "chunks.json")

    def files(self):
        """Every content file in the corpus, de-duplicated, in stable order."""
        out = []
        for pattern in self.include:
            out += glob.glob(os.path.join(self.root, pattern), recursive=True)
        out = [f for f in sorted(set(out)) if os.path.basename(f) not in self.exclude]
        out += [p for p in (os.path.join(self.root, n) for n in self.extra)
                if os.path.exists(p) and p not in out]
        return out

    def is_authority(self, path):
        return os.path.basename(path) in self.authority


def load_corpora():
    corpora = {k: dict(v) for k, v in BUILTIN_CORPORA.items()}
    if os.path.exists(CORPORA_FILE):
        for name, spec in json.load(open(CORPORA_FILE, encoding="utf-8")).items():
            merged = dict(corpora.get(name, {}))
            merged.update(spec)
            merged.setdefault("root", HERE)
            if not os.path.isabs(merged["root"]):     # relative roots resolve from the repo root
                merged["root"] = os.path.normpath(os.path.join(HERE, merged["root"]))
            corpora[name] = merged
    return corpora


DEFAULT_CORPUS = "support"
CORPUS = Corpus(DEFAULT_CORPUS, BUILTIN_CORPORA[DEFAULT_CORPUS])


def use_corpus(name):
    """Select the active corpus. Everything downstream reads module-level CORPUS."""
    global CORPUS
    corpora = load_corpora()
    if name not in corpora:
        raise SystemExit(f"Unknown corpus {name!r} — available: {sorted(corpora)}")
    CORPUS = Corpus(name, corpora[name])
    return CORPUS

# ---- embedding (local; nothing leaves the machine) ------------------------
_model = None
def get_model():
    global _model
    if _model is None:
        from model2vec import StaticModel
        _model = StaticModel.from_pretrained(MODEL_NAME)
    return _model

def embed(texts):
    """list[str] -> np.ndarray (n, dim) float32"""
    return np.asarray(get_model().encode(texts), dtype=np.float32)

# ---- chunking: one chunk per ## / ### section, tagged with its page --------
_FRONTMATTER = re.compile(r"^---\n.*?\n---\n", re.DOTALL)

_DATE = re.compile(r'(\d{4}-\d{2}-\d{2})')

def parse_page(path):
    raw = open(path, encoding="utf-8").read()
    title = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r'^title:\s*"?(.*?)"?\s*$', raw, re.MULTILINE)   # prefer frontmatter title
    if m:
        title = m.group(1).strip()
    body = _FRONTMATTER.sub("", raw).strip()
    return title, body, raw

def page_signals(path, raw):
    """Two temporal-authority signals per page, captured at index time:
      • updated  — the page's own freshness (frontmatter `last-updated`, else a
                   `*Last updated:*` line, else `created`). ISO date str.
      • authority — the corpus's declared override file(s): when an authority doc
                   disagrees with an ordinary page, the authority doc is right.
                   Top tier lets ground truth break ties against stale pages."""
    fm = _FRONTMATTER.match(raw)
    front = fm.group(0) if fm else ""
    m = (re.search(r'^last-updated:\s*"?(\d{4}-\d{2}-\d{2})', front, re.M)
         or re.search(r'[Ll]ast[ -]updated:?\s*\*?\s*(\d{4}-\d{2}-\d{2})', raw)
         or re.search(r'^created:\s*"?(\d{4}-\d{2}-\d{2})', front, re.M)
         or _DATE.search(raw))
    updated = m.group(1) if m else "2026-01-01"        # floor for undated pages
    authority = 2 if CORPUS.is_authority(path) else 1
    return updated, authority

def chunk_page(path):
    title, body, raw = parse_page(path)
    updated, authority = page_signals(path, raw)
    rel = os.path.relpath(path, CORPUS.root)
    out = []
    for part in re.split(r'\n(?=#{2,3}\s)', body):     # split before ## / ### headings
        part = part.strip()
        if len(part) < 20:                              # skip near-empty fragments
            continue
        hm = re.match(r'#{2,3}\s+(.*)', part)
        heading = hm.group(1).strip() if hm else "(intro)"
        # what we embed includes the page title + heading, so a chunk carries its context
        embed_text = f"{title} — {heading}\n{part}"
        out.append({"source": rel, "title": title, "heading": heading,
                    "text": part, "embed_text": embed_text,
                    "updated": updated, "authority": authority})
    return out

# ---- index: chunk everything, embed once, save ----------------------------
def build_index():
    files = CORPUS.files()
    if not files:
        raise SystemExit(f"No files matched for corpus {CORPUS.name!r} under {CORPUS.root}")
    chunks = []
    for f in files:
        chunks.extend(chunk_page(f))
    print(f"  [{CORPUS.name}] {len(files)} pages → {len(chunks)} chunks")
    vectors = embed([c["embed_text"] for c in chunks])
    os.makedirs(CORPUS.index_dir, exist_ok=True)
    np.save(CORPUS.vec_path, vectors)
    meta = [{k: c[k] for k in ("source", "title", "heading", "text", "updated", "authority")}
            for c in chunks]
    json.dump(meta, open(CORPUS.meta_path, "w"), ensure_ascii=False)
    print(f"  saved → {os.path.relpath(CORPUS.vec_path, HERE)} {vectors.shape} + chunks.json")

def load_index():
    if not os.path.exists(CORPUS.vec_path):
        raise SystemExit(f"No index for corpus {CORPUS.name!r} — run:  python rag.py index --corpus {CORPUS.name}")
    return np.load(CORPUS.vec_path), json.load(open(CORPUS.meta_path))

# ---- retrieval: cosine + temporal-authority reranking ---------------------
# When two chunks are close on meaning, prefer the CURRENT one. Cosine alone is
# time-blind: it happily ranks a superseded page above the ruling that replaced it
# (the "a citation is not a fact" failure). These weights add two small signals on
# top of cosine — deliberately small, so they only decide near-ties (exactly where
# supersession bugs live) and leave clear matches untouched.
def _date_ord(s):
    try:
        return __import__("datetime").date.fromisoformat(s).toordinal()
    except Exception:
        return __import__("datetime").date(2026, 1, 1).toordinal()

def retrieve(query, k=5, weighted=True):
    vecs, meta = load_index()
    q = embed([query])[0]
    vn = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)   # normalize rows
    qn = q / (np.linalg.norm(q) + 1e-9)
    cos = vn @ qn                                                       # cosine = dot of unit vectors

    if not weighted:
        scores = cos
    else:
        # Recency as EXPONENTIAL DECAY, not linear min-max: age is measured from
        # the newest page in the corpus, and weight halves every HALFLIFE_DAYS.
        # Linear min-max fails here because a few ancient pages stretch the scale
        # so all recent dates squash together — killing resolution exactly among
        # the recent pages where supersession happens (a Jun ruling vs a Jul one).
        ords = np.array([_date_ord(m.get("updated", "2026-01-01")) for m in meta], dtype=np.float64)
        age = ords.max() - ords                                         # days older than newest
        recency = 0.5 ** (age / CORPUS.halflife)                        # newest -> 1.0, decays
        authority = np.array([1.0 if m.get("authority", 1) >= 2 else 0.0 for m in meta])
        scores = cos + CORPUS.w_rec * recency + CORPUS.w_auth * authority

    top = np.argsort(-scores)[:k]
    return [(float(scores[i]), meta[i]) for i in top]

# ---- api key: .env (gitignored) -> environment -----------------------------
def load_env():
    """Read KEY=value lines from .env into os.environ. No dependency, and a
    real env var always wins so CI / a one-off export can override the file."""
    path = os.path.join(HERE, ".env")
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

# ---- model routing: match the model to the task, not the reverse ----------
# The orchestration point most RAG demos skip: don't send every call to the most
# expensive model. Here the routing is deliberately simple because the workload is.
#   • tier 0  — retrieve()  : NO model at all (local embeddings). The cheapest
#                             routing decision is "does this even need an LLM?".
#   • "fast"  — grounded synthesis over already-retrieved context. This is
#               extractive-with-citations, not open reasoning — a small model
#               does it well and ~15× cheaper. Default.
#   • "deep"  — opt-in for genuinely hard synthesis (conflicting sources,
#               multi-hop). Reserve the expensive model for where it earns it.
MODELS = {
    "fast": "claude-haiku-4-5-20251001",
    "deep": "claude-opus-4-8",
}
DEFAULT_TIER = "fast"

def pick_model(tier):
    if tier not in MODELS:
        raise SystemExit(f"Unknown tier {tier!r} — choose from {list(MODELS)}")
    return MODELS[tier]

def _call_with_retry(client, **kw):
    """The API can return a transient 429/529 under load. Retry with exponential
    backoff so a capacity blip isn't a crash — a shipped tool shouldn't traceback
    on a condition that resolves itself in four seconds."""
    import anthropic
    transient = (anthropic.OverloadedError, anthropic.RateLimitError,
                 anthropic.APIConnectionError, anthropic.InternalServerError)
    for attempt in range(5):
        try:
            return client.messages.create(**kw)
        except transient as e:
            if attempt == 4:
                raise
            wait = 2 ** attempt
            print(f"  … API busy ({type(e).__name__}) — retry {attempt+1}/4 in {wait}s")
            time.sleep(wait)

def ask(query, k=5, tier=DEFAULT_TIER):
    hits = retrieve(query, k)
    context = "\n\n".join(
        f"[{i+1}] ({h['source']} :: {h['heading']})\n{h['text']}"
        for i, (_, h) in enumerate(hits))
    system = (
        f"You answer ONLY from the provided context, which comes from {CORPUS.label}. "
        "Cite the source with a bracket like [1] for every claim. "
        f"If the answer is not contained in the context, reply exactly: '{CORPUS.refusal}' "
        "Never use outside knowledge.")
    load_env()
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("No ANTHROPIC_API_KEY — put it in .env (see .env.example) or export it.")
    client = anthropic.Anthropic()
    model = pick_model(tier)
    resp = _call_with_retry(
        client,
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}],
    )
    answer = "".join(b.text for b in resp.content if b.type == "text")
    return answer, hits, model

# ---- CLI ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Corpus-agnostic grounded RAG with temporal-authority reranking")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS,
                    help=f"which corpus profile to use (default: {DEFAULT_CORPUS})")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index", help="build/refresh the index")
    sub.add_parser("corpora", help="list available corpus profiles")
    r = sub.add_parser("retrieve", help="show top-k chunks"); r.add_argument("query"); r.add_argument("-k", type=int, default=5)
    a = sub.add_parser("ask", help="grounded answer w/ citations"); a.add_argument("query"); a.add_argument("-k", type=int, default=5)
    a.add_argument("--deep", action="store_true", help=f"escalate to the deep model ({MODELS['deep']}) for hard synthesis; default is fast ({MODELS['fast']})")
    args = ap.parse_args()

    if args.cmd == "corpora":
        for name, spec in sorted(load_corpora().items()):
            c = Corpus(name, spec)
            built = "indexed" if os.path.exists(c.vec_path) else "not indexed"
            print(f"  {name:<12} {len(c.files()):>4} files  ({built})  → {c.root}")
        return

    use_corpus(args.corpus)

    if args.cmd == "index":
        build_index()
    elif args.cmd == "retrieve":
        for score, h in retrieve(args.query, args.k):
            print(f"  {score:.3f}  {h['source']} :: {h['heading']}")
    elif args.cmd == "ask":
        answer, hits, model = ask(args.query, args.k, tier="deep" if args.deep else DEFAULT_TIER)
        print(answer + "\n\nsources:")
        for i, (score, h) in enumerate(hits):
            print(f"  [{i+1}] {h['source']} :: {h['heading']}  ({score:.3f})")
        print(f"\n[model: {model}]")

if __name__ == "__main__":
    main()
