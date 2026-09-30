"""rag_core.py — Pipeline RAG « Lois numériques du Bénin ».

Source unique du pipeline : utilisée par le notebook, l'application Streamlit et le CLI.
Les imports lourds (torch, faiss, google-genai) sont faits à la demande.
"""
import os
import re
import json
import time
import unicodedata
from pathlib import Path

import numpy as np

CONFIG = {
    "EMBEDDING_MODEL": "BAAI/bge-m3",                                    # multilingue, bon en français
    "CROSS_ENCODER_MODEL": "BAAI/bge-reranker-v2-m3",                    # re-ranking multilingue (défaut)
    "CROSS_ENCODER_BASELINE": "antoinelouis/crossencoder-me5-small-mmarcoFR",  # 1er essai, comparé en section 10
    "LLM_MODEL": os.getenv("GEMINI_MODEL", "gemini-flash-latest"),       # modèle principal
    "LLM_FALLBACKS": ["gemini-flash-lite-latest", "gemini-3.5-flash"],   # essayés si le principal est indisponible / épuisé
    "TEMPERATURE": 0.1,       # faible : on veut de la fidélité au texte de loi, pas de créativité
    "MAX_OUTPUT_TOKENS": 2048,  # marge large (les modèles « thinking » consomment aussi ce budget)
    "DEFAULT_MODE": "hybrid_rerank",  # dense | bm25 | hybrid | hybrid_rerank (fixé d'après l'évaluation, section 10)
    "K_CANDIDATES": 20,       # candidats récupérés avant fusion / re-ranking
    "K_FINAL": 4,             # extraits envoyés au LLM
    "RERANK_THRESHOLD": 0.05,  # seuil de pertinence (à CALIBRER, cf. notebook)
    "RRF_K": 60,              # constante de Reciprocal Rank Fusion
    "N_QUERY_VARIANTS": 3,    # reformulations générées en multi-query
}
LLM_LOG = []  # une entrée par appel LLM : latence, tokens (sert à l'analyse coût / latence)
from collections import Counter

CORPUS_REPORT = {}  # rapport de qualité d'extraction, rempli par load_corpus()

_COMMON = frozenset("de la le les des du et en un une que qui est dans pour par sur au aux ou ne pas plus se sa son ses "
                    "ce cette il elle a à l d".split())


def text_quality(text):
    """Indicateurs d'une extraction saine : part de mots français courants (~0,3 attendu) et caractères de contrôle (0 attendu)."""
    words = re.findall(r"[A-Za-zÀ-ÿ]+", text)
    ctrl = sum(1 for ch in text if ord(ch) < 32 and ch not in "\n\r\t")
    common = sum(w.lower() in _COMMON for w in words) / len(words) if words else 0.0
    return {"mots_courants": round(common, 3), "car_controle": ctrl}


def _decode_segment(seg):
    """Décode un segment dont les glyphes sont décalés (police sans table Unicode) : code réel = code lu + 29 (ASCII),
    + 62 pour les lettres accentuées ; U+0335 = apostrophe. Heuristique vérifiée sur des exemples réels du PDF de l'APDP."""
    if seg.isdigit():
        return seg
    if not any(ch.isalpha() for ch in seg) and all(ord(ch) >= 32 for ch in seg):
        return seg  # ponctuation imprimable authentique (« : », « , »…) : les glyphes corrompus sont des codes de contrôle
    lower = sum(ch.islower() and ch.isascii() for ch in seg)
    if lower and lower / len(seg) >= 0.4:
        return seg  # segment majoritairement normal : on n'y touche pas
    out = []
    for ch in seg:
        o = ord(ch)
        if ch.islower() and ch.isascii():
            out.append(ch)
        elif o == 0x335:
            out.append("’")
        elif 3 <= o <= 97:
            out.append(chr(o + 29))
        elif 160 <= o <= 200:
            out.append(chr(o + 62))
        else:
            out.append(ch)
    return "".join(out)


def repair_shifted_text(text):
    """Répare les lignes corrompues (repérées par le glyphe \\x03 = espace décalé). Retourne (texte, nb_lignes_réparées)."""
    out, n = [], 0
    for line in text.split("\n"):
        if "\x03" not in line:
            out.append(line)
            continue
        n += 1
        pieces = re.split(r"([ \x03]+)", line)
        out.append("".join(" " if ("\x03" in pc) else (pc if re.fullmatch(r"[ \t]+", pc) else _decode_segment(pc)) for pc in pieces))
    return "\n".join(out), n


def _drop_running_lines(text, min_count=30, min_len=25):
    """Supprime les en-têtes / pieds de page répétés (ex. « [546] LOI N°2017-20 DU 20 AVRIL 2018 PORTANT CODE… »)."""
    norm = lambda l: re.sub(r"\d+", "#", re.sub(r"\s+", " ", l.strip()))
    lines = text.split("\n")
    cnt = Counter(norm(l) for l in lines if l.strip())
    bad = {k for k, v in cnt.items() if v >= min_count and len(k) >= min_len and not k.lower().startswith("article")}
    kept = [l for l in lines if norm(l) not in bad]
    return "\n".join(kept), len(lines) - len(kept)


def _clean(text):
    """Nettoyage léger du texte extrait des PDF."""
    text = text.replace("\r", "")
    text = re.sub(r"-\n(?=[a-zà-ÿ])", "", text)  # mots coupés en fin de ligne
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_corpus(folder="corpus"):
    """Charge tous les .pdf / .txt / .md du dossier. Retourne {nom_du_fichier_sans_extension: texte}.
    Répare l'extraction corrompue, retire les en-têtes répétés, et remplit CORPUS_REPORT (contrôle qualité)."""
    laws = {}
    for p in sorted(Path(folder).iterdir()):
        suffix = p.suffix.lower()
        if suffix == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(p))
            text = "\n".join((pg.extract_text() or "") for pg in reader.pages)
        elif suffix in (".txt", ".md"):
            text = p.read_text(encoding="utf-8")
        else:
            continue
        text, n_rep = repair_shifted_text(text)
        text, n_drop = _drop_running_lines(text)
        text = _clean(text)
        laws[p.stem] = text
        CORPUS_REPORT[p.stem] = {"caracteres": len(text), "lignes_reparees": n_rep,
                                 "lignes_repetitives_supprimees": n_drop, **text_quality(text)}
    return laws
# En-tête d'article : « Article 12 », « Article premier », « Art. 5 bis » ... en début de ligne.
# ADAPTEZ cette regex au format réel de vos PDF si peu d'articles sont détectés.
ARTICLE_RE = re.compile(
    r"(?m)^[ \t]*(?:Article|ARTICLE|Art\.)[ \t]+(?P<num>premier|1er|\d+)(?P<suf>[ \t]*(?:bis|ter|quater))?\b[^\n]*"
)


def _norm_num(num, suf):
    n = "1" if num.lower() in ("premier", "1er") else num
    return n + (suf.strip().lower() if suf else "")


def _split_long(text, size, overlap):
    """Découpe un texte trop long en morceaux de ~size caractères avec recouvrement."""
    if len(text) <= size:
        return [text]
    parts, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind("\n", start + size // 2, end)
            if cut == -1:
                cut = text.rfind(" ", start + size // 2, end)
            if cut != -1:
                end = cut
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return parts


# Titres de structure (LIVRE / TITRE / CHAPITRE / SECTION…) : ils figurent à la fin du texte de l'article précédent
# dans le PDF ; on les retire du corps de l'article et on les réutilise comme CONTEXTE des articles qui suivent.
HEADING_RE = re.compile(r"(?m)^[ \t]*(?:LIVRE|TITRE|CHAPITRE|SECTION|SOUS-SECTION|PARTIE)\b.*$")


def chunk_by_article(law, text, max_chars=2500, overlap=250):
    """Stratégie A (proposée) : 1 chunk = 1 article de loi, avec métadonnées (loi, n° d'article) et contexte de structure."""
    matches = list(ARTICLE_RE.finditer(text))
    chunks, ctx = [], ""
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.start():end].strip()
        h = HEADING_RE.search(body)
        next_ctx = ctx
        if h:
            next_ctx = re.sub(r"\s+", " ", body[h.start():]).strip()[:300]
            body = body[:h.start()].rstrip()
        art = _norm_num(m.group("num"), m.group("suf"))
        meta = {"loi": law, "article": art, "strategy": "article"}
        for j, part in enumerate(_split_long(body, max_chars, overlap)):
            txt = part if j == 0 else f"Article {art} (suite)\n{part}"
            chunks.append({"text": txt, "meta": dict(meta), "ctx": ctx})
        ctx = next_ctx
    return chunks


def chunk_fixed(law, text, size=1500, overlap=300):
    """Stratégie B (baseline) : découpage en fenêtres de taille fixe, sans tenir compte des articles."""
    return [
        {"text": p, "meta": {"loi": law, "article": None, "strategy": "fixed"}}
        for p in _split_long(text, size, overlap)
    ]


def build_chunks(laws, strategy="article"):
    fn = chunk_by_article if strategy == "article" else chunk_fixed
    out = []
    for law, text in laws.items():
        out.extend(fn(law, text))
    return out
STOPWORDS = set(
    "le la les de du des un une et en a au aux ce ces que qui quoi dans par pour sur avec est sont l d j n s c qu "
    "il elle ils elles on se sa son ses leur leurs ou ne pas plus ni mais donc car y".split()
)


def tokenize(text):
    """Minuscules, sans accents, sans mots vides : sert à la recherche lexicale BM25."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return [t for t in re.findall(r"[a-z0-9]+", text) if t not in STOPWORDS and len(t) > 1]


class BM25:
    """BM25 Okapi minimal (recherche par mots-clés)."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b, self.N = k1, b, len(docs)
        self.dl = [len(d) for d in docs]
        self.avgdl = sum(self.dl) / max(self.N, 1)
        self.inverted, df = {}, {}
        for i, d in enumerate(docs):
            counts = {}
            for t in d:
                counts[t] = counts.get(t, 0) + 1
            for t, f in counts.items():
                self.inverted.setdefault(t, []).append((i, f))
                df[t] = df.get(t, 0) + 1
        self.idf = {t: float(np.log(1 + (self.N - n + 0.5) / (n + 0.5))) for t, n in df.items()}

    def scores(self, query_tokens):
        s = np.zeros(self.N)
        for t in set(query_tokens):
            for i, f in self.inverted.get(t, []):
                denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                s[i] += self.idf[t] * f * (self.k1 + 1) / denom
        return s


_EMB, _CE = None, {}


def get_embedder():
    global _EMB
    if _EMB is None:
        from sentence_transformers import SentenceTransformer
        _EMB = SentenceTransformer(CONFIG["EMBEDDING_MODEL"])
    return _EMB


def get_cross_encoder(name=None):
    name = name or CONFIG["CROSS_ENCODER_MODEL"]
    if name not in _CE:
        from sentence_transformers import CrossEncoder
        _CE[name] = CrossEncoder(name, max_length=512)
    return _CE[name]


def search_text(chunk):
    """Texte indexé (embeddings + BM25) : contexte de structure (chapitre / section) + texte de l'article."""
    return (chunk.get("ctx", "") + "\n" + chunk["text"]).strip()


class Index:
    """Vector store FAISS (recherche sémantique) + index BM25 (recherche lexicale) sur les mêmes chunks."""

    def __init__(self, chunks, embeddings=None):
        import faiss
        self.chunks = chunks
        if embeddings is None:
            embeddings = get_embedder().encode(
                [search_text(c) for c in chunks], normalize_embeddings=True, batch_size=16, show_progress_bar=True
            )
        self.emb = np.asarray(embeddings, dtype="float32")
        self.faiss = faiss.IndexFlatIP(self.emb.shape[1])  # produit scalaire = cosinus (vecteurs normalisés)
        self.faiss.add(self.emb)
        self.bm25 = BM25([tokenize(search_text(c)) for c in chunks])

    def save(self, folder="index"):
        Path(folder).mkdir(exist_ok=True)
        np.save(Path(folder) / "embeddings.npy", self.emb)
        (Path(folder) / "chunks.json").write_text(json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, folder="index"):
        chunks = json.loads((Path(folder) / "chunks.json").read_text(encoding="utf-8"))
        return cls(chunks, embeddings=np.load(Path(folder) / "embeddings.npy"))
def dense_search(idx, query, k):
    qv = get_embedder().encode([query], normalize_embeddings=True).astype("float32")
    _, ids = idx.faiss.search(qv, k)
    return [int(i) for i in ids[0] if i >= 0]


def bm25_search(idx, query, k):
    s = idx.bm25.scores(tokenize(query))
    return [int(i) for i in np.argsort(-s)[:k] if s[i] > 0]


def rrf(rankings, top=None, k_const=None):
    """Reciprocal Rank Fusion : fusionne plusieurs classements (score = somme de 1/(k+rang))."""
    k_const = k_const or CONFIG["RRF_K"]
    scores = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking):
            scores[i] = scores.get(i, 0.0) + 1.0 / (k_const + rank + 1)
    order = sorted(scores, key=scores.get, reverse=True)
    return order[:top] if top else order


def hybrid_search(idx, query, k):
    return rrf([dense_search(idx, query, k), bm25_search(idx, query, k)], top=k)


def rerank(idx, query, ids, top, model=None):
    """Re-ranking par cross-encoder : score chaque couple (question, extrait). Retourne [(id, score)]."""
    if not ids:
        return []
    scores = get_cross_encoder(model).predict([(query, idx.chunks[i]["text"]) for i in ids])
    ranked = sorted(zip(ids, scores), key=lambda x: -x[1])
    return [(i, float(s)) for i, s in ranked[:top]]


def retrieve(idx, question, mode="hybrid_rerank", k=None, variants=None, reranker=None):
    """mode ∈ {dense, bm25, hybrid, hybrid_rerank}. `variants` = reformulations (multi-query).
    Retourne une liste de (id_chunk, score_rerank_ou_None), meilleur en premier."""
    k = k or CONFIG["K_FINAL"]
    kc = CONFIG["K_CANDIDATES"]
    queries = [question] + list(variants or [])
    if mode == "dense":
        rankings = [dense_search(idx, q, kc) for q in queries]
    elif mode == "bm25":
        rankings = [bm25_search(idx, q, kc) for q in queries]
    else:
        rankings = [hybrid_search(idx, q, kc) for q in queries]
    fused = rrf(rankings, top=kc)
    if mode == "hybrid_rerank":
        return rerank(idx, question, fused, k, model=reranker)
    return [(i, None) for i in fused[:k]]
import hashlib

SYSTEM_PROMPT = """Tu es un assistant d'information juridique spécialisé dans le droit du numérique en République du Bénin.

RÈGLES :
1. Réponds UNIQUEMENT à partir des extraits de textes fournis dans le CONTEXTE. N'utilise aucune autre connaissance juridique.
2. Cite la source de chaque règle avec le format [Loi, Article N].
3. Si le contexte ne permet pas de répondre, réponds : « Je ne trouve pas de disposition correspondante dans les textes dont je dispose. » N'invente JAMAIS d'article, de sanction ou de procédure.
4. Structure ta réponse : **Ce que dit la loi** ; **Sanctions ou recours** (uniquement s'ils figurent dans le contexte) ; **Limites** (ce que les extraits ne permettent pas de conclure).
5. Utilise un langage simple, accessible à un non-juriste.
6. Termine par : « Information générale, qui ne remplace pas l'avis d'un avocat ou d'un professionnel du droit. »"""

REFUSAL = (
    "Je ne trouve pas de disposition correspondante dans les textes dont je dispose. "
    "Essayez de reformuler votre question ou consultez un professionnel du droit."
)


class QuotaExceeded(RuntimeError):
    """Plus aucun modèle utilisable (quota journalier épuisé ou modèles indisponibles)."""


CACHE_DIR = Path("llm_cache")  # cache disque : une requête identique ne consomme pas de quota deux fois
USE_CACHE = True
_CLIENT = None
_DEAD_MODELS = {}  # modèle -> raison (quota journalier épuisé / 404) pour la session en cours


def reset_dead_models():
    _DEAD_MODELS.clear()


def get_client():
    global _CLIENT
    if _CLIENT is None:
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Clé API absente : définissez la variable d'environnement GEMINI_API_KEY.")
        from google import genai
        _CLIENT = genai.Client(api_key=key)
    return _CLIENT


def list_models():
    """Modèles disponibles pour votre clé (n'utilise pas de quota de génération)."""
    return [m.name.replace("models/", "") for m in get_client().models.list()
            if "generateContent" in (m.supported_actions or [])]


def _retry_delay(msg):
    m = re.search(r"retry in ([\d.]+)s", msg) or re.search(r"retryDelay['\"]?: ?['\"]?(\d+)", msg)
    return float(m.group(1)) if m else None


def call_llm(prompt, system=None, kind="generation", temperature=None, max_tokens=None, retries=4):
    """Appel Gemini : cache disque, bascule automatique entre modèles, retry adapté à l'erreur, mesure latence / tokens.
    - quota JOURNALIER ou modèle introuvable (404) : ce modèle est écarté, on passe au suivant (LLM_FALLBACKS) ;
    - 503 / 429 « par minute » : on essaie le modèle suivant, puis on attend avant de recommencer ;
    - 400/401/403 : erreur de configuration (clé…), non réessayable ;
    - si tous les modèles sont écartés : QuotaExceeded."""
    temp = CONFIG["TEMPERATURE"] if temperature is None else temperature
    mx = max_tokens or CONFIG["MAX_OUTPUT_TOKENS"]
    ckey = hashlib.sha256(json.dumps([CONFIG["LLM_MODEL"], system, prompt, temp, mx], ensure_ascii=False).encode()).hexdigest()
    cfile = CACHE_DIR / f"{ckey}.json"
    if USE_CACHE and cfile.exists():
        rec = json.loads(cfile.read_text(encoding="utf-8"))
        LLM_LOG.append({**rec["stats"], "kind": kind, "attempt": 0, "cached": True})  # mesures d'origine conservées
        return rec["text"]

    from google.genai import types
    cfg = types.GenerateContentConfig(system_instruction=system, temperature=temp, max_output_tokens=mx)
    models = [CONFIG["LLM_MODEL"]] + [m for m in CONFIG.get("LLM_FALLBACKS", []) if m != CONFIG["LLM_MODEL"]]
    last_err = None
    for attempt in range(retries):
        alive = [m for m in models if m not in _DEAD_MODELS]
        if not alive:
            raise QuotaExceeded("Aucun modèle utilisable : " + " ; ".join(f"{m} ({r})" for m, r in _DEAD_MODELS.items()))
        wait = 0.0
        for model in alive:
            try:
                t0 = time.time()
                r = get_client().models.generate_content(model=model, contents=prompt, config=cfg)
                u = getattr(r, "usage_metadata", None)
                stats = {
                    "latency_s": time.time() - t0, "model": model,
                    "tokens_in": getattr(u, "prompt_token_count", 0) or 0,
                    "tokens_out": getattr(u, "candidates_token_count", 0) or 0,
                }
                LLM_LOG.append({**stats, "kind": kind, "attempt": attempt + 1, "cached": False})
                text = r.text or ""
                if USE_CACHE and text:
                    CACHE_DIR.mkdir(exist_ok=True)
                    cfile.write_text(json.dumps({"text": text, "stats": stats}, ensure_ascii=False), encoding="utf-8")
                return text
            except Exception as e:
                msg, code, last_err = str(e), getattr(e, "code", None), e
                if "PerDay" in msg:
                    _DEAD_MODELS[model] = "quota journalier épuisé"
                elif code == 404:
                    _DEAD_MODELS[model] = "modèle indisponible (404)"
                elif code in (400, 401, 403):
                    raise RuntimeError(f"Erreur de configuration (non réessayable) : {msg[:300]}") from e
                elif code == 429 or "RESOURCE_EXHAUSTED" in msg:
                    wait = max(wait, (_retry_delay(msg) or 10) + 1)
                else:  # 503 surcharge, réseau…
                    wait = max(wait, 2 ** attempt * 2)
        if wait:
            time.sleep(min(wait, 65))
    raise RuntimeError(f"Échec de l'appel LLM après {retries} tentatives : {last_err}")


def expand_query(question, n=None):
    """Multi-query : reformule la question pour imiter la FORMULATION des articles de loi.
    Écart de vocabulaire fréquent : le citoyen dit « photo / publier / Facebook », la loi dit
    « image », « porter atteinte à l'intimité de la vie privée », « sans le consentement de »."""
    n = n or CONFIG["N_QUERY_VARIANTS"]
    prompt = (
        f"Un citoyen pose cette question sur le droit du numérique au Bénin :\n« {question} »\n\n"
        f"Écris {n} requêtes de recherche, une par ligne, sans numérotation ni commentaire. Chacune doit imiter "
        "la formulation d'un article de code pénal ou de code du numérique francophone (vocabulaire juridique exact : "
        "« porter atteinte à », « sans le consentement de », « est puni de », « le fait de », notions d'infraction, "
        "de sanction, de données à caractère personnel, de vie privée, de système informatique…). "
        "Varie les angles : (1) l'infraction pénale possible, (2) la protection des données personnelles, "
        "(3) la sanction ou le recours."
    )
    try:
        text = call_llm(prompt, kind="query_expansion", temperature=0.3, max_tokens=512)
    except QuotaExceeded:
        raise
    except RuntimeError:
        return []
    lines = [re.sub(r"^[\s\-\*\d\.\)]+", "", ln).strip() for ln in text.splitlines()]
    return [ln for ln in lines if ln][:n]
def _label(meta):
    return f"{meta['loi']}, Article {meta['article']}" if meta.get("article") else f"{meta['loi']}, extrait"


def select_sources(idx, question, mode=None, multiquery=True, reranker=None):
    """Étapes SANS génération : (multi-query) → retrieval → seuil de pertinence.
    Retourne (reformulations, sources). Utilisable pour l'évaluation sans consommer de quota de génération."""
    mode = mode or CONFIG["DEFAULT_MODE"]
    variants = expand_query(question) if multiquery else []
    hits = retrieve(idx, question, mode, variants=variants, reranker=reranker)
    thr = CONFIG["RERANK_THRESHOLD"]
    if mode == "hybrid_rerank" and thr is not None:
        hits = [(i, s) for i, s in hits if s >= thr]  # document grading : on écarte le hors-sujet
    sources = [
        {"label": _label(idx.chunks[i]["meta"]), "loi": idx.chunks[i]["meta"]["loi"],
         "article": idx.chunks[i]["meta"].get("article"), "score": s, "text": idx.chunks[i]["text"]}
        for i, s in hits
    ]
    return variants, sources


def answer(idx, question, mode=None, multiquery=True, reranker=None):
    """Pipeline complet : sélection des sources → prompt augmenté → génération."""
    t0, log_start = time.time(), len(LLM_LOG)
    out = {"question": question, "variants": [], "sources": [], "refused": False, "error": None}
    try:
        out["variants"], out["sources"] = select_sources(idx, question, mode, multiquery, reranker)
        if not out["sources"]:
            out.update(answer=REFUSAL, refused=True)
        else:
            context = "\n\n---\n\n".join(f"[{s['label']}]\n{s['text']}" for s in out["sources"])
            prompt = f"CONTEXTE :\n{context}\n\nQUESTION : {question}\n\nRÉPONSE :"
            out["answer"] = call_llm(prompt, system=SYSTEM_PROMPT, kind="generation")
    except QuotaExceeded as e:
        out.update(answer="Quota de l'API épuisé pour le moment. Réessayez plus tard.", error=str(e))
    except RuntimeError as e:
        out.update(answer="Le service de génération est momentanément indisponible. Réessayez plus tard.",
                   error=str(e))
    out["latency_s"] = time.time() - t0
    out["llm_calls"] = LLM_LOG[log_start:]
    return out

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or input("Question : ")
    result = answer(Index.load("index"), q)
    print("\n" + result["answer"])
    print("\nSources :")
    for s in result["sources"]:
        print(f" - {s['label']} (score {s['score']})")
    print(f"\nLatence : {result['latency_s']:.1f}s")
