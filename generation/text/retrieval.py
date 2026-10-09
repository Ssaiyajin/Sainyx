"""Answer lookup for the text chat.

The 57M character-level model cannot store facts reliably, so questions are
answered from text we already have: the hand-written Q&A pairs and the
Wikipedia intros collected by ``build_qa_store.py``. This module needs no
torch and no network, so it is cheap to test and quick to load.

Matching goes in two steps:
1. Entity match: strip "who is / what is / tell me about", then look the rest
   up as a title or alias (exact, token subset, then small typos).
2. Passage match: for other questions ("who created Goku"), score stored
   sentences by how much of the question's weighted vocabulary they cover.
If neither is confident enough, ``answer`` returns None and the caller decides
what to do instead of guessing.
"""
import ast
import difflib
import json
import math
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_STORE_PATH = os.path.join(_ROOT, "data", "text", "qa_store.json")
DEFAULT_QA_SOURCE = os.path.join(_ROOT, "generation", "text", "build_qa.py")
DEFAULT_CURATED_PATH = os.path.join(_ROOT, "data", "text", "curated_qa.json")

STOPWORDS = frozenset("""
a an the is are was were be been am do does did of in on at to for from by with and or but
who what which when where why how whom whose it its this that these those there their they
he she his her him them i you me my your we our us about tell explain describe please can could
would should will as into than then so if not no yes also more some any
""".split())

_LEAD = re.compile(
    r"^(?:(?:please|can you|could you) )*"
    r"(?:(?:tell me|explain|describe|talk about)(?: more)?(?: about)?"
    r"|who is|who was|who are|who were|whos"
    r"|what is|what was|what are|what were|whats"
    r"|do you know(?: about| who| what)?"
    r"|info on|information on|information about|about) "
)
_ARTICLE = re.compile(r"^(?:a|an|the) ")

ABBREV = {"dr", "mr", "mrs", "ms", "vs", "st", "jr", "sr", "no", "inc", "co", "ltd"}


def normalize(text: str) -> str:
    """Lowercase, drop accents and punctuation, collapse spaces."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = text.replace("'", "").replace("\u2019", "")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


def strip_question(norm: str) -> str:
    """'who is the goku' style prefixes removed from an already-normalized string."""
    prev = None
    while prev != norm:
        prev = norm
        norm = _LEAD.sub("", norm, count=1)
        norm = _ARTICLE.sub("", norm, count=1)
    return norm.strip()


def _osa(a: str, b: str) -> int:
    """Edit distance where swapping two neighbouring letters (bulam -> bulma) counts as one typo."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[len(a)][len(b)]


def _max_typos(n: int) -> int:
    return 0 if n <= 3 else 1 if n <= 5 else 2 if n <= 9 else 3


def _stem(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def content_tokens(norm: str) -> List[str]:
    return [_stem(t) for t in norm.split() if t not in STOPWORDS]


# ---------- text cleanup helpers (also used by build_qa_store.py) ----------

def strip_parens(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\([^()]*\)", "", s)
    s = re.sub(r"\[[^\]]*\]", "", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def split_sentences(text: str) -> List[str]:
    out, start = [], 0
    for m in re.finditer(r"[.!?](?=\s+[A-Z\"'])", text):
        end = m.end()
        words = text[start:end].rstrip(".!?").split()
        w = words[-1] if words else ""
        if w.lower() in ABBREV or (len(w) == 1 and w.isupper()):
            continue
        out.append(text[start:end].strip())
        start = end
    rest = text[start:].strip()
    if rest:
        out.append(rest)
    return out


def lead_to_sentences(lead: str, limit: int = 8) -> List[str]:
    lead = strip_parens(lead.replace("\n", " "))
    return [s for s in split_sentences(lead) if len(s) >= 25][:limit]


def make_answer(sentences: List[str], max_chars: int = 420) -> Optional[str]:
    if not sentences:
        return None
    ans = sentences[0]
    if len(ans) > max_chars:
        return None
    if len(sentences) > 1 and len(ans) + 1 + len(sentences[1]) <= max_chars:
        ans = ans + " " + sentences[1]
    return ans


def short_title(title: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", title).strip()


# ---------- the store ----------

@dataclass
class Answer:
    text: str
    title: str
    score: float
    method: str     # "entity" | "passage"
    source: str     # "handwritten" | "curated" | "wikipedia"
    matched: str = ""         # name the question was matched to, for "Showing results for ..."
    corrected: bool = False   # True when the question had a typo that was fixed


class QAStore:
    """entries: dicts with title, answer, optional aliases / facts / source."""

    def __init__(self, entries: Iterable[dict]):
        self.entries = [e for e in entries if e.get("title") and e.get("answer")]
        self._keys = {}
        for i, e in enumerate(self.entries):
            for name in [e["title"], short_title(e["title"]), *e.get("aliases", [])]:
                k = strip_question(normalize(name))
                if k and k not in self._keys:       # earlier entries win (hand-written come first)
                    self._keys[k] = i
        self._key_list = list(self._keys)

        self._docs = []                              # (token Counter, entry index, text)
        for i, e in enumerate(self.entries):
            title_toks = content_tokens(normalize(short_title(e["title"])))
            if e.get("source") == "handwritten":
                toks = content_tokens(normalize(e["title"] + " " + e["answer"]))
                self._docs.append((Counter(toks), i, e["answer"]))
            for fact in e.get("facts", []):
                toks = title_toks + content_tokens(normalize(fact))
                self._docs.append((Counter(toks), i, fact))
        df = Counter()
        for toks, _, _ in self._docs:
            df.update(toks.keys())
        n = max(len(self._docs), 1)
        self._idf = {t: math.log(1 + n / c) for t, c in df.items()}
        self._idf_unknown = math.log(1 + n)

    def __len__(self):
        return len(self.entries)

    # -- step 1: entity lookup --
    def _display(self, entry_idx: int, key: str) -> str:
        e = self.entries[entry_idx]
        return key.title() if e.get("source") == "handwritten" else short_title(e["title"])

    def _entity(self, q: str) -> Optional[Answer]:
        if not q:
            return None
        hit, key, score, corrected = self._keys.get(q), q, 1.0, False
        qt = q.split()
        if hit is None and len(q) >= 4 and len(qt) <= 3:
            cands = [k for k in self._key_list if set(qt) <= set(k.split())]
            if cands:
                key = min(cands, key=lambda k: (len(k), k))
                hit, score = self._keys[key], 0.9
        if hit is None and len(q) >= 4:
            limit = _max_typos(len(q))
            scored = []
            for k in self._key_list:
                if abs(len(k) - len(q)) > limit or (len(q) <= 6 and k[0] != q[0]):
                    continue
                d = _osa(q, k)
                if d <= limit:
                    scored.append((d, -difflib.SequenceMatcher(None, q, k).ratio(), k))
            scored.sort()
            if scored and not (len(scored) > 1 and scored[1][:2] == scored[0][:2]
                               and self._keys[scored[1][2]] != self._keys[scored[0][2]]):
                key = scored[0][2]
                hit, corrected = self._keys[key], True
                score = 1 - scored[0][0] / max(len(q), len(key))
        if hit is None:
            return None
        e = self.entries[hit]
        return Answer(e["answer"], e["title"], score, "entity", e.get("source", "wikipedia"),
                      matched=self._display(hit, key), corrected=corrected)

    # -- step 2: passage lookup --
    def _passage(self, q: str, min_coverage: float = 0.75) -> Optional[Answer]:
        qtoks = list(dict.fromkeys(content_tokens(q)))
        if len(qtoks) < 2:
            return None
        weight = {t: self._idf.get(t, self._idf_unknown) for t in qtoks}
        total = sum(weight.values())
        best = None
        for toks, i, text in self._docs:
            matched = sum(w for t, w in weight.items() if t in toks)
            cov = matched / total
            if cov >= min_coverage and (best is None or (cov, matched) > best[0]):
                best = ((cov, matched), i, text)
        if best is None:
            return None
        e = self.entries[best[1]]
        text, name = best[2], short_title(e["title"])
        if normalize(name) not in normalize(text):
            text = f"{name}: {text}"
        return Answer(text, e["title"], best[0][0], "passage", e.get("source", "wikipedia"), matched=name)

    def answer(self, query: str) -> Optional[Answer]:
        q = strip_question(normalize(query))
        return self._entity(q) or self._passage(q)

    def compare(self, query: str):
        """'goku vs vegeta' / 'difference between goku and vegeta' -> (Answer, Answer) or None."""
        n = normalize(query)
        m = re.match(r"^(?:what is the )?(?:difference between|compare) (.+?) (?:and|vs|versus|with|to) (.+)$", n) \
            or re.match(r"^(.+?) (?:vs|versus) (.+)$", n)
        if not m:
            return None
        first, second = self._entity(strip_question(m.group(1))), self._entity(strip_question(m.group(2)))
        if first is None or second is None or first.matched == second.matched:
            return None
        return first, second

    def suggest(self, query: str, limit: int = 3) -> List[str]:
        """Close names for a question that found no answer, for 'Did you mean ...?'."""
        q = strip_question(normalize(query))
        if len(q) < 3:
            return []
        out = []
        for k in difflib.get_close_matches(q, self._key_list, n=8, cutoff=0.72):
            name = self._display(self._keys[k], k)
            if name not in out:
                out.append(name)
        return out[:limit]

    @classmethod
    def from_json(cls, path: str, extra_entries: Iterable[dict] = ()):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(list(extra_entries) + data.get("entries", []))


_GREETINGS = {"hi", "hello", "hey", "hiya", "yo", "sup", "good morning", "good evening", "good afternoon", "hello there", "hi there"}
_HELP = {"help", "what can you do", "what do you do", "who are you", "what are you", "how does this work", "what can i ask"}
_THANKS = {"thanks", "thank you", "thx", "ty", "thanks a lot", "cheers"}


def smalltalk(query: str, store: Optional[QAStore] = None) -> Optional[str]:
    """Fixed replies for greetings, thanks and 'what can you do'. No guessing involved."""
    n = normalize(query)
    if n in _THANKS:
        return "You're welcome. Ask me about another character, series or game whenever you like."
    if n in _GREETINGS or n in _HELP:
        examples = []
        if store is not None:
            examples = [name for key, name in [("goku", "Goku"), ("bulma", "Bulma"), ("naruto uzumaki", "Naruto Uzumaki"),
                                               ("elden ring", "Elden Ring")] if key in store._keys]
        tail = ""
        if len(examples) > 1:
            tail = f" Try asking about {', '.join(examples[:-1])} or {examples[-1]}."
        elif examples:
            tail = f" Try asking about {examples[0]}."
        if n in _GREETINGS:
            return "Hi! I answer questions about Dragon Ball, anime and games." + tail
        return ("I answer questions about Dragon Ball, anime and games using stored Wikipedia text. "
                "Ask about a character, series or game by name, or compare two with 'Goku vs Vegeta'." + tail)
    return None


def parse_qa_pairs(text: str):
    pairs = re.findall(r"Question:\s*(.+?)\nAnswer:\s*(.+?)(?=\n\nQuestion:|\Z)", text.strip(), flags=re.S)
    return list(dict.fromkeys((q.strip(), a.strip().replace("\n", " ")) for q, a in pairs))


def handwritten_entries(source_path: str = DEFAULT_QA_SOURCE) -> List[dict]:
    """Read the hand-written Q&A straight from the *_qa strings in build_qa.py.

    qa_pairs.txt is git-ignored and only exists after build_qa.py has been run, so the
    deployed app cannot rely on it. The file is parsed, not imported, because importing
    build_qa.py writes a 170 KB file as a side effect.
    """
    if not os.path.exists(source_path):
        return []
    with open(source_path, encoding="utf-8") as f:
        text = f.read()
    if source_path.endswith(".txt"):
        pairs = parse_qa_pairs(text)
    else:
        pairs = []
        for node in ast.parse(text).body:
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                    and any(isinstance(t, ast.Name) and t.id.endswith("_qa") for t in node.targets)):
                pairs += parse_qa_pairs(node.value.value)
        pairs = list(dict.fromkeys(pairs))
    return [{"title": q, "answer": a, "source": "handwritten"} for q, a in pairs]


def curated_entries(path: str = DEFAULT_CURATED_PATH) -> List[dict]:
    """Short checked answers for names Wikipedia only has as a section of a list page (Gogeta, Vegito, ...)."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [dict(e, source="curated") for e in json.load(f).get("entries", [])]


_default = None


def get_default_store(store_path: str = DEFAULT_STORE_PATH, qa_source: str = DEFAULT_QA_SOURCE,
                      curated_path: str = DEFAULT_CURATED_PATH) -> QAStore:
    """Priority order: hand-written, curated, then the Wikipedia store (when it has been built)."""
    global _default
    if _default is None:
        first = handwritten_entries(qa_source) + curated_entries(curated_path)
        _default = QAStore.from_json(store_path, first) if os.path.exists(store_path) else QAStore(first)
    return _default