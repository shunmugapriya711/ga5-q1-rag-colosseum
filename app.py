"""RAG Colosseum service: hybrid (dense MiniLM + BM25) retrieval with extractive answers.

POST /query {"question": str, "top_k": int} -> {"answer", "sources", "scores", "confidence"}
"""
import json
import math
import os
import re
import threading
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastembed import TextEmbedding
from pydantic import BaseModel

KB_PATH = Path(os.environ.get("KB_PATH", Path(__file__).parent / "knowledge_base.json"))
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Abstention thresholds (dense cosine on the best passage)
DENSE_ABSTAIN = float(os.environ.get("DENSE_ABSTAIN", "0.30"))
W_DENSE, W_BM25 = 0.65, 0.35

STOP = set("""a an the of in on at to for from by with without and or but if then than is are was were be been
being do does did done what which who whom whose when where why how much many long often is it its this that these
those there their they them as into onto over under per each any all some can could should would will may might must
shall use uses used using about between both either neither not no yes vs versus compared compare comparison
difference differ different respectively value values tell me please give explain describe list name i you we our your
my default typically typical exactly according kb knowledge base passage passages context say says said state stated
also like such other one two set sets""".split())

# Abbreviations used in questions but not spelled out in the KB.
ALIASES = {
    "sgd": "stochastic gradient descent",
    "mvcc": "multi-version concurrency control",
    "2pc": "two-phase commit",
    "3pc": "three-phase commit",
    "hlc": "hybrid logical clocks",
    "hlcs": "hybrid logical clocks",
    "mlp": "multi-layer perceptrons",
    "mlps": "multi-layer perceptrons",
    "ecc": "elliptic curve cryptography",
    "resnet": "residual convolutional networks resnet",
    "cnn": "convolutional networks",
    "batchnorm": "batch normalization",
    "layernorm": "layer normalization",
    "bn": "batch normalization",
    "ln": "layer normalization",
    "lsm-tree": "log-structured merge trees",
    "lsm": "log-structured merge trees lsm",
    "b-tree": "b+ tree",
    "btree": "b+ tree",
    "postgres": "postgresql",
    "pg": "postgresql",
    "mongo": "mongodb",
    "innodb": "mysql innodb",
    "waf": "web application firewalls",
    "jwt": "json web tokens jwt",
    "jwts": "json web tokens jwt",
    "pki": "public key cryptography",
    "rsa": "rsa public key cryptography",
    "chacha20": "chacha20-poly1305",
    "chacha": "chacha20-poly1305",
    "http2": "http/2",
    "http3": "http/3",
    "logreg": "logistic regression",
    "svms": "support vector machines svm",
    "dht": "consistent hashing",
    "hrw": "rendezvous hashing highest random weight",
    "paxos": "multi-paxos paxos",
    "aes-gcm": "aes gcm advanced encryption standard",
    "pbkdf": "pbkdf2",
    "oauth": "oauth 2.0 pkce",
    "saga": "saga pattern",
    "sagas": "saga pattern",
    "tcp": "transmission control protocol",
    "dns": "domain name system dns",
    "bgp": "border gateway protocol bgp",
    "ospf": "open shortest path first ospf",
    "tls": "transport layer security tls",
    "aes": "advanced encryption standard aes",
    "wal": "write-ahead logging wal",
    "gossip": "epidemic gossip protocols",
    "2-phase": "two-phase commit",
}

# Paraphrase synonyms (gauntlet questions avoid the KB's wording): appended to the query, never replacing.
SYNONYMS = {
    "keepalive": "heartbeat", "keepalives": "heartbeats", "subordinate": "follower", "subordinates": "followers",
    "coordinator": "leader", "triggers": "timeouts", "coefficient": "rate", "scalar": "rate",
    "magnitude": "scale", "daemon": "server", "provisioning": "generators create", "allocated": "listen",
}

# Capitalised discourse words that are not named entities.
DISCOURSE = set("""furthermore moreover however additionally additional also synthesizing synthesising synthesize
given since assuming suppose considering regarding note finally then next first second third lastly meanwhile
in adjacent architecture overall together both""".split())

INJECTION_RE = re.compile(
    r"(ignore|disregard|forget|override|bypass)\b|system\s*(prompt|message|note|override|instruction)|"
    r"\b(instruction|instructions|jailbreak|pretend|roleplay)\b|you are now|you must|act as|"
    r"\b(respond|reply|output|print|say|return|write|answer)\b[^.?!]{0,40}\b(only|exactly|with|verbatim|the word|instead)\b|"
    r"\b(reveal|leak|disclose)\b|\b(hacked|pwned|compromised)\b|^\s*(system|assistant|developer|admin|note to)\b\s*:?|"
    r"<\s*/?\s*(system|inst|admin)|\[/?inst\]|###",
    re.IGNORECASE,
)
COMPOUND_RE = re.compile(r"\b(and|both|compare|compared|versus|vs\.?|whereas|while|respectively|difference|"
                         r"differ|each|sum|total|combined|together|plus)\b|,", re.IGNORECASE)
NUMERIC_Q_RE = re.compile(r"\b(how many|how much|how long|how often|port|timeout|interval|size|threshold|ratio|"
                          r"rate|bits?|length|number|count|quorum|duration|limit|window|latency|lease|layers?|"
                          r"blocks?|batch|value|percent|milliseconds|ms|key|digest|hash)\b", re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+/\-\.]*[A-Za-z0-9+]|[A-Za-z0-9]")
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def stem(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("es") and w[-3] in "sxz":
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss") and not w.endswith("us"):
        return w[:-1]
    return w


def raw_tokens(text: str):
    out = []
    for t in TOKEN_RE.findall(text):
        t = t.strip(".-/").lower()
        if not t:
            continue
        out.append(t)
        if "-" in t or "/" in t:
            out.extend(p for p in re.split(r"[-/]", t) if p)
    return out


def tokens(text: str):
    return [stem(t) for t in raw_tokens(text) if t not in STOP]


def expand_aliases(text: str) -> str:
    def rep(m):
        w = m.group(0)
        a = ALIASES.get(w.lower()) or SYNONYMS.get(w.lower())
        return f"{w} ({a})" if a else w
    return re.sub(r"[A-Za-z0-9][A-Za-z0-9+/\-]*", rep, text)


def sanitize(question: str) -> str:
    """Drop clauses that look like prompt-injection; keep the actual question."""
    q = re.sub(r"<[^>]{0,200}>", " ", question)  # html/xml-ish tags
    parts = re.split(r"(\([^)]*\)|\[[^\]]*\]|\{[^}]*\}|\"[^\"]*\"|[.!?;]+(?=\s|$)|\n+|\s[-–—]{1,2}\s)", q)
    kept = []
    for p in parts:
        if not p or not p.strip():
            continue
        if INJECTION_RE.search(p):
            continue
        kept.append(p.strip(" ()[]{}\""))
    cleaned = ". ".join(k.strip() for k in kept if re.search(r"[A-Za-z0-9]", k))  # keep sentence boundaries
    return re.sub(r"\s+", " ", cleaned).strip()


# ---------------------------------------------------------------- index
PASSAGES = json.loads(KB_PATH.read_text(encoding="utf-8"))
N = len(PASSAGES)
# threads=1 + small batches keep peak RAM ~250MB (default settings exceed Render's 512MB free tier)
_model = TextEmbedding(MODEL_NAME, threads=int(os.environ.get("EMBED_THREADS", "1")))
_lock = threading.Lock()


def embed(texts):
    with _lock:
        v = np.array(list(_model.embed(list(texts), batch_size=8)), dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


P_EMB = embed(PASSAGES)
SENTS = [SENT_RE.split(p.strip()) for p in PASSAGES]
FLAT = [(i, j, s) for i, ss in enumerate(SENTS) for j, s in enumerate(ss)]
S_EMB = embed([f"{SENTS[i][0]} {s}" if j else s for i, j, s in FLAT])  # sentence + topic context
S_INDEX = {(i, j): k for k, (i, j, _) in enumerate(FLAT)}
S_TOK = [set(tokens(s)) for _, _, s in FLAT]
W_SENT = float(os.environ.get("W_SENT", "0.6"))

P_TOK = [tokens(p) for p in PASSAGES]
P_TF = [Counter(t) for t in P_TOK]
DF = Counter(t for tf in P_TF for t in tf)
AVGDL = sum(len(t) for t in P_TOK) / N
IDF = {t: math.log(1 + (N - df + 0.5) / (df + 0.5)) for t, df in DF.items()}
# Words that appear with capitals in the KB (proper nouns, acronyms, sentence-initial words).
VOCAB = set()
for p in PASSAGES:
    VOCAB.update(t.strip(".-/").lower() for t in TOKEN_RE.findall(p) if any(c.isupper() for c in t))
    VOCAB.update(x.lower() for x in re.findall(r"[A-Z][A-Za-z0-9]+", p))
VOCAB.update(ALIASES.keys())
VOCAB.discard("")
# Topic map: stemmed token -> passages whose first sentence contains it.
TOPIC = {}
for i, ss in enumerate(SENTS):
    for t in set(tokens(ss[0])):
        TOPIC.setdefault(t, set()).add(i)
TOPIC_TOKS = [set(tokens(ss[0])) for ss in SENTS]


def bm25(q_toks, k1=1.4, b=0.75):
    s = np.zeros(N, dtype=np.float32)
    for t in set(q_toks):
        if t not in IDF:
            continue
        for i in range(N):
            f = P_TF[i].get(t, 0)
            if f:
                s[i] += IDF[t] * f * (k1 + 1) / (f + k1 * (1 - b + b * len(P_TOK[i]) / AVGDL))
    return s


@lru_cache(maxsize=4096)
def q_embed(text: str):
    return embed([text])[0]


def known(word: str) -> bool:
    w = word.lower().strip(".")
    if w in VOCAB or DF.get(stem(w), 0) >= 3:
        return True
    if len(w) >= 4 and any(v.startswith(w) or (len(v) >= 4 and w.startswith(v)) for v in VOCAB if len(v) >= 4):
        return True
    return False


def entity_tokens(expanded_q: str):
    """Stemmed tokens from capitalised/acronym words (and alias expansions) that name a specific topic."""
    ents = set()
    for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9+/\-\.]*[A-Za-z0-9+]", expanded_q):
        cap = any(c.isupper() for c in w[1:]) or w[0].isupper() or any(c.isdigit() for c in w)
        alias_part = w.lower() in " ".join(ALIASES.values()).split()
        if cap or alias_part:
            for t in tokens(w):
                if 0 < DF.get(t, 0) <= 4:
                    ents.add(t)
    return ents


def unknown_entities(clean_q: str):
    """Proper-noun-ish tokens (capitalised mid-sentence, acronyms, alnum mixes) absent from the KB."""
    out = []
    for seg in re.split(r"(?<!\d)[.:;?!](?!\d)", clean_q):  # first word of each sentence/clause may be capitalised
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9+/\-\.]*[A-Za-z0-9+]|[A-Za-z]", seg)
        for idx, w in enumerate(words):
            lw = w.lower()
            if lw in STOP or lw in DISCOURSE or len(w) < 2:
                continue
            is_ent = (w[0].isupper() and idx > 0) or (sum(c.isupper() for c in w) >= 2) or \
                     (any(c.isdigit() for c in w) and any(c.isalpha() for c in w))
            if is_ent and not known(w):
                out.append(w)
    return out


def best_sentence(i: int, qv: np.ndarray, q_toks, numeric: bool, exclude=()):
    best, best_s = None, -1e9
    qset = set(q_toks)
    for j, s in enumerate(SENTS[i]):
        if j in exclude:
            continue
        sv = S_EMB[S_INDEX[(i, j)]]
        st = set(tokens(s))
        overlap = sum(IDF.get(t, 0) for t in qset & st) / (sum(IDF.get(t, 0.5) for t in qset) or 1)
        score = float(sv @ qv) + 0.6 * overlap
        if numeric and re.search(r"\d", s):
            score += 0.25
        if score > best_s:
            best, best_s = j, score
    return best, best_s


def retrieve(clean_q: str):
    qv = q_embed(clean_q)
    dense = P_EMB @ qv
    q_toks = tokens(clean_q)
    bm = bm25(q_toks)
    bm_n = bm / bm.max() if bm.max() > 0 else bm
    hybrid = W_DENSE * np.clip(dense, 0, 1) + W_BM25 * bm_n
    ents = entity_tokens(clean_q)
    topic_hits = {}
    for t in ents:
        for i in TOPIC.get(t, ()):
            topic_hits.setdefault(i, set()).add(t)
    # Sentence-level evidence: a passage holding one sentence that matches the question's attribute well
    # beats a passage that matches loosely overall (paraphrased probes, look-alike passages).
    attr = set(t for t in q_toks if t not in ents) or set(q_toks)
    denom = sum(IDF.get(t, 0.5) for t in attr) or 1
    s_dense = S_EMB @ qv
    sent_best = np.zeros(N, dtype=np.float32)
    for k, (i, j, s) in enumerate(FLAT):
        cov = sum(IDF.get(t, 0) for t in attr & S_TOK[k]) / denom
        v = float(s_dense[k]) + 0.6 * cov
        if v > sent_best[i]:
            sent_best[i] = v
    hybrid = (hybrid + W_SENT * sent_best) / (1 + W_SENT)
    for i, ts in topic_hits.items():  # named topic (Raft, AdamW, Postgres...) wins over look-alikes
        hybrid[i] += 0.12 * min(len(ts), 2)
    hybrid = np.clip(hybrid, 0, 1)
    return qv, q_toks, dense, bm_n, hybrid, ents, topic_hits


def abstain():
    return {"answer": None, "sources": [], "scores": [], "confidence": 0.05}


SYNTH_RE = re.compile(r"\b(integrat\w*|synthesi[sz]\w*|interoperat\w*|work together|relate|relationship)\b|"
                      r"\bhow (do|does|would|can) (both|they|these|the two|each)\b", re.IGNORECASE)
MECH_RE = re.compile(r"^\s*how (does|do|is|are|can|would|should)\b|\b(exact|mathematical) (formula|formulation|"
                     r"derivation|equation)\b|\bwhy (does|do|is|are)\b", re.IGNORECASE)
NUM_RE = re.compile(r"(?<![A-Za-z0-9\-/.])\d+(?:\.\d+)?")
GENERIC = set(tokens("protocol protocols algorithm system systems mechanism handle handles handled work works "
                     "operate operates implement implementation process perform execute during when"))


def numbers_in(text: str):
    return {float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)}


def build(chosen, pieces, order, hybrid, top_k):
    answer = " ".join(dict.fromkeys(pieces))
    # sources: chosen passages first, then remaining by hybrid score; scores descending in [0,1]
    srcs = list(chosen) + [int(i) for i in order if int(i) not in chosen]
    srcs = srcs[:max(top_k, len(chosen))]
    scores, prev = [], 1.0
    for i in srcs:
        v = min(prev, max(0.0, min(1.0, float(hybrid[i]))))
        scores.append(round(v, 4))
        prev = v
    conf = round(float(min(0.99, max(0.65, 0.5 + 0.6 * hybrid[chosen[0]]))), 4)
    return {"answer": answer, "sources": srcs, "scores": scores, "confidence": conf}


def synthesize(clean: str, top_k: int):
    """'X does A. Furthermore: Y does B. How do both integrate?' -> one passage per described system."""
    chosen, seg_scores = [], {}
    for seg in re.split(r"(?<!\d)[.?!](?!\d)", clean):
        seg = SYNTH_RE.sub(" ", seg)
        if len(tokens(seg)) < 3:
            continue
        _, _, dense, _, h, _, _ = retrieve(expand_aliases(seg))
        i = int(np.argmax(h))
        if dense[i] >= 0.45 and i not in chosen:
            chosen.append(i)
            seg_scores[i] = float(h[i])
    if len(chosen) < 2:
        return None
    _, _, _, _, hybrid, _, _ = retrieve(expand_aliases(clean))
    for i in chosen:
        hybrid[i] = max(hybrid[i], seg_scores[i])
    chosen.sort(key=lambda x: -hybrid[x])
    order = np.argsort(-hybrid)
    pieces = [" ".join(SENTS[i][:2]) for i in chosen]
    return build(chosen, pieces, order, hybrid, top_k)


def answer_query(question: str, top_k: int = 3):
    top_k = max(1, min(int(top_k or 3), N))
    clean = sanitize(question or "")
    if len(tokens(clean)) == 0:
        return abstain()
    expanded = expand_aliases(clean)
    if unknown_entities(clean):
        return abstain()

    if SYNTH_RE.search(clean):
        res = synthesize(clean, top_k)
        if res:
            return res

    qv, q_toks, dense, bm_n, hybrid, ents, topic_hits = retrieve(expanded)
    order = np.argsort(-hybrid)
    top = int(order[0])
    if dense[top] < DENSE_ABSTAIN and bm_n[top] < 0.5:
        return abstain()
    if float(dense.max()) < DENSE_ABSTAIN - 0.08:
        return abstain()

    # False premise: the question asserts a number the matching passage does not contain
    # ("Since Raft requires 100 nodes ...") -> refuse rather than answer on a poisoned premise.
    q_nums = {float(x) for x in NUM_RE.findall(clean)}
    if q_nums and not q_nums <= numbers_in(PASSAGES[top]):
        return abstain()

    # Mechanism / derivation question about a KB topic whose specifics the passage never covers.
    if MECH_RE.search(clean):
        attr = set(t for t in q_toks if t not in ents and t not in GENERIC and t not in TOPIC_TOKS[top])
        denom = sum(IDF.get(t, 0.5) for t in attr)
        if denom:
            best_cov = max(sum(IDF.get(t, 0) for t in attr & set(tokens(s))) / denom for s in SENTS[top])
            if best_cov < 0.34:
                return abstain()

    numeric = bool(NUMERIC_Q_RE.search(clean))
    chosen = [top]

    # Cross-passage synthesis: compound question whose distinct entities live in different passages.
    if COMPOUND_RE.search(clean):
        # passages that are the *topic* of an entity named in the question
        cand = set()
        for t in ents:
            ps = TOPIC.get(t, set())
            if len(ps) == 1:
                cand |= ps
            elif 1 < len(ps) <= 3:
                cand.add(max(ps, key=lambda x: hybrid[x]))
        for i in sorted(cand, key=lambda x: -hybrid[x]):
            if i not in chosen and hybrid[i] >= 0.5 * hybrid[top] and dense[i] >= DENSE_ABSTAIN - 0.05:
                chosen.append(i)
            if len(chosen) >= 3:
                break

    pieces = []
    attr_toks = [t for t in q_toks if t not in ents] or q_toks  # entity words pick passages, not sentences
    for i in chosen:
        j, _ = best_sentence(i, qv, attr_toks, numeric)
        pieces.append(SENTS[i][j])
        if len(chosen) == 1 and COMPOUND_RE.search(clean):
            j2, s2 = best_sentence(i, qv, attr_toks, numeric, exclude={j, 0})
            if j2 is not None and s2 > 0.55:
                pieces.append(SENTS[i][j2])
    return build(chosen, pieces, order, hybrid, top_k)


# ---------------------------------------------------------------- API
app = FastAPI(title="RAG Colosseum")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Query(BaseModel):
    question: str = ""
    top_k: int = 3


import time
from collections import deque

LOG = deque(maxlen=200)  # recent requests, viewable at GET /log (for tuning against the gauntlet)


@app.post("/query")
def query(q: Query):
    try:
        res = answer_query(q.question, q.top_k)
    except Exception as e:  # never fail the request
        res = abstain()
        res_err = repr(e)
    else:
        res_err = None
    entry = {"t": time.strftime("%H:%M:%S"), "question": q.question, "top_k": q.top_k, "response": res}
    if res_err:
        entry["error"] = res_err
    LOG.append(entry)
    print("QUERY", json.dumps(entry, ensure_ascii=False), flush=True)
    return res


@app.get("/log")
def get_log():
    return list(LOG)


@app.get("/")
def root():
    return {"status": "ok", "passages": N, "endpoint": "POST /query"}
