import os
import re
import asyncio
from typing import Optional, List, Dict, Any, Literal
from collections import defaultdict

from fastapi import FastAPI
from pydantic import BaseModel
from pinecone import Pinecone
from openai import AsyncOpenAI

# ======================================================
# APP
# ======================================================

app = FastAPI(title="Animal Knowledge Backend")


@app.get("/")
async def root():
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "ok"}


# ======================================================
# CONFIG
# ======================================================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "animal-kb")

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise RuntimeError("Missing API keys")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
openai = AsyncOpenAI(api_key=OPENAI_API_KEY)

EMBED_MODEL = "text-embedding-3-large"

DEFAULT_TOP_K = 12
MIN_SCORE = 0.15
RERANK_CANDIDATES = 12

ConfidenceLevel = Literal["high", "medium", "inferred", "unknown"]


# ======================================================
# REQUEST MODEL
# ======================================================

class QueryRequest(BaseModel):
    query: str
    top_k: int = DEFAULT_TOP_K
    debug: bool = False


# ======================================================
# NORMALIZATION
# ======================================================

def normalize_typos(text: str) -> str:
    replacements = {
        "lyon": "lion",
        "tigar": "tiger",
        "elefant": "elephant",
        "egle": "eagle",
        "shrk": "shark",
        "habbitat": "habitat",
        "contintent": "continent",
        "carnivor": "carnivore",
        "herbivor": "herbivore",
    }

    t = (text or "").lower()

    for typo, correct in replacements.items():
        t = t.replace(typo, correct)

    return t


def normalize(text: str) -> str:
    text = normalize_typos(text or "")
    text = text.lower()
    text = text.replace("-", " ")
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ======================================================
# QUERY REWRITE
# ======================================================

def rewrite_query(query: str) -> str:
    t = normalize(query)

    if any(x in t for x in ["what does", "what do", "eat", "food", "diet"]):
        for animal in ["lion", "tiger", "elephant", "eagle", "shark"]:
            if animal in t:
                return f"{animal} diet habitat description"

    if any(x in t for x in ["where", "live", "habitat", "found"]):
        for animal in ["lion", "tiger", "elephant", "eagle", "shark"]:
            if animal in t:
                return f"{animal} habitat continent description"

    if any(x in t for x in ["endangered", "conservation", "protected", "extinction"]):
        return "animal conservation status endangered vulnerable least concern"

    if any(x in t for x in ["mammal", "bird", "fish"]):
        return f"{query} animal type"

    if any(x in t for x in ["africa", "asia", "worldwide"]):
        return f"{query} continent habitat animal"

    return query


# ======================================================
# QUERY EXPANSION
# ======================================================

QUERY_SYNONYMS: Dict[str, List[str]] = {
    "eat": ["diet", "food", "feeds on"],
    "food": ["diet", "eat"],
    "diet": ["eat", "food"],
    "live": ["habitat", "found in"],
    "habitat": ["live", "found in", "environment"],
    "africa": ["savanna", "grasslands"],
    "asia": ["forests", "grasslands"],
    "worldwide": ["global", "oceans", "many regions"],
    "carnivore": ["predator", "meat eater"],
    "herbivore": ["plant eater"],
    "lion": ["big cat", "predator", "savanna"],
    "tiger": ["big cat", "striped predator", "forest"],
    "elephant": ["large mammal", "herbivore"],
    "eagle": ["bird of prey", "flying predator"],
    "shark": ["marine predator", "ocean fish"],
    "endangered": ["conservation status", "protected"],
}


def expand_query(query: str) -> str:
    q = normalize(query)
    additions: List[str] = []

    for key, synonyms in QUERY_SYNONYMS.items():
        if key in q:
            additions.extend(synonyms)

    additions = list(dict.fromkeys(additions))

    if additions:
        return query + " " + " ".join(additions)

    return query


# ======================================================
# DETECTION / ROUTING
# ======================================================

ANIMAL_NAMES = ["lion", "tiger", "elephant", "eagle", "shark"]


def detect_animal_name(text: str) -> Optional[str]:
    t = normalize(text)
    for animal in ANIMAL_NAMES:
        if animal in t:
            return animal
    return None


def detect_continent(text: str) -> Optional[str]:
    t = normalize(text)
    for continent in ["africa", "asia", "worldwide"]:
        if continent in t:
            return continent
    return None


def detect_animal_type(text: str) -> Optional[str]:
    t = normalize(text)
    for animal_type in ["mammal", "bird", "fish"]:
        if animal_type in t:
            return animal_type
    return None


def detect_diet(text: str) -> Optional[str]:
    t = normalize(text)
    if "carnivore" in t or "carnivores" in t:
        return "carnivore"
    if "herbivore" in t or "herbivores" in t:
        return "herbivore"
    return None


def detect_conservation_status(text: str) -> Optional[str]:
    t = normalize(text)
    if "endangered" in t:
        return "endangered"
    if "vulnerable" in t:
        return "vulnerable"
    if "least concern" in t:
        return "least concern"
    return None


def detect_query_mode(text: str) -> str:
    t = normalize(text)

    if any(x in t for x in ["which animals", "what animals", "list animals", "show animals"]):
        return "list"

    if any(x in t for x in ["what does", "what do", "diet", "eat", "food"]):
        return "fact"

    if any(x in t for x in ["where", "habitat", "live", "found"]):
        return "fact"

    if any(x in t for x in ["endangered", "conservation", "protected"]):
        return "list"

    return "semantic"


# ======================================================
# EMBEDDINGS
# ======================================================

async def embed(text: str) -> List[float]:
    result = await openai.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return result.data[0].embedding


# ======================================================
# SEARCH HELPERS
# ======================================================

def dedupe_matches(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_by_id: Dict[str, Dict[str, Any]] = {}

    for m in matches:
        match_id = m.get("id")
        score = m.get("score", 0) or 0

        if not match_id:
            continue

        if match_id not in best_by_id or score > (best_by_id[match_id].get("score", 0) or 0):
            best_by_id[match_id] = m

    return sorted(best_by_id.values(), key=lambda x: -(x.get("score", 0) or 0))


async def pinecone_query(
    vector: List[float],
    filters: Optional[Dict[str, Any]],
    top_k: int
) -> List[Dict[str, Any]]:
    response = await asyncio.to_thread(
        index.query,
        vector=vector,
        filter=filters,
        top_k=top_k,
        include_metadata=True
    )

    if not response or not getattr(response, "matches", None):
        return []

    matches = []

    for m in response.matches:
        matches.append({
            "id": m.id,
            "score": m.score,
            "metadata": m.metadata
        })

    return matches


async def metadata_search(filters: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
    # Uses tiny/core records for metadata filtering
    dummy_vector = [0.0] * 3072
    matches = await pinecone_query(dummy_vector, filters, top_k)
    return dedupe_matches(matches)[:top_k]


async def semantic_search(
    query: str,
    top_k: int,
    filters: Optional[Dict[str, Any]] = None,
    min_score: Optional[float] = MIN_SCORE
) -> List[Dict[str, Any]]:
    rewritten = rewrite_query(query)
    expanded = expand_query(rewritten)
    vector = await embed(expanded)

    candidate_top_k = max(top_k, RERANK_CANDIDATES)
    matches = await pinecone_query(vector, filters, candidate_top_k)
    deduped = dedupe_matches(matches)

    if min_score is None:
        return deduped[:top_k]

    filtered = [
        m for m in deduped
        if (m.get("score", 0.0) or 0.0) >= min_score
    ]

    return filtered[:top_k]


# ======================================================
# RENDERING
# ======================================================

def get_record_text(meta: Dict[str, Any]) -> str:
    return (meta.get("text") or "").strip()


def render_matches(matches: List[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    seen: set[str] = set()

    for match in matches:
        meta = match.get("metadata", {})
        text = get_record_text(meta)
        name = (meta.get("name") or "unknown").title()
        record_id = meta.get("animal_id") or match.get("id") or "UNKNOWN"

        if not text:
            continue

        dedupe_key = f"{record_id}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        title = f"### ANIMAL {name} ###"
        blocks.append(f"{title}\n{text}")

    return "\n\n---\n\n".join(blocks)


def render_list(matches: List[Dict[str, Any]]) -> str:
    names = []
    seen = set()

    for match in matches:
        meta = match.get("metadata", {})
        name = meta.get("name")
        if name and name not in seen:
            seen.add(name)
            names.append(name.title())

    return "\n".join(f"• {n}" for n in names)


def infer_confidence(matches: List[Dict[str, Any]], exact_filter_match: bool = False) -> ConfidenceLevel:
    if exact_filter_match and matches:
        return "high"

    if not matches:
        return "unknown"

    best = max((m.get("score", 0.0) or 0.0) for m in matches)

    if best >= 0.80:
        return "high"
    if best >= 0.55:
        return "medium"
    if best >= 0.25:
        return "inferred"
    return "unknown"


# ======================================================
# FILTER BUILDERS
# ======================================================

def build_filter_from_query(query: str) -> Optional[Dict[str, Any]]:
    animal_name = detect_animal_name(query)
    continent = detect_continent(query)
    animal_type = detect_animal_type(query)
    diet = detect_diet(query)
    conservation_status = detect_conservation_status(query)

    clauses: List[Dict[str, Any]] = []

    if animal_name:
        clauses.append({"name": {"$eq": animal_name}})
    if continent:
        clauses.append({"continent": {"$eq": continent}})
    if animal_type:
        clauses.append({"animal_type": {"$eq": animal_type}})
    if diet:
        clauses.append({"diet": {"$eq": diet}})
    if conservation_status:
        clauses.append({"conservation_status": {"$eq": conservation_status}})

    if not clauses:
        return None

    if len(clauses) == 1:
        return clauses[0]

    return {"$and": clauses}


# ======================================================
# MAIN ENDPOINT
# ======================================================

@app.post("/query")
async def query_animals(data: QueryRequest):
    query = (data.query or "").strip()
    top_k = max(1, min(data.top_k, 25))

    if not query:
        return {
            "kbContext": "",
            "kbAnswer": "No information is available because the query is empty.",
            "api_answer": 0,
            "confidence": "unknown",
            "match_type": "no_query",
            "debug": None
        }

    debug: Dict[str, Any] = {}

    query_mode = detect_query_mode(query)
    detected_animal = detect_animal_name(query)
    detected_continent = detect_continent(query)
    detected_type = detect_animal_type(query)
    detected_diet = detect_diet(query)
    detected_status = detect_conservation_status(query)

    pinecone_filter = build_filter_from_query(query)

    rewritten = rewrite_query(query)
    expanded = expand_query(rewritten)

    debug["query_mode"] = query_mode
    debug["detected_animal"] = detected_animal
    debug["detected_continent"] = detected_continent
    debug["detected_type"] = detected_type
    debug["detected_diet"] = detected_diet
    debug["detected_status"] = detected_status
    debug["filter"] = pinecone_filter
    debug["rewritten_query"] = rewritten
    debug["expanded_query"] = expanded

    # --------------------------------------------------
    # 1. LIST MODE WITH METADATA FILTERS
    # --------------------------------------------------

    if query_mode == "list" and pinecone_filter:
        matches = await metadata_search(pinecone_filter, top_k=top_k)

        if matches:
            return {
                "kbContext": render_list(matches),
                "api_answer": 1,
                "confidence": "high",
                "match_type": "metadata_list",
                "debug": debug if data.debug else None
            }

    # --------------------------------------------------
    # 2. FACT MODE WITH ANIMAL FILTER
    # --------------------------------------------------

    if query_mode == "fact" and detected_animal:
        matches = await semantic_search(
            query=query,
            top_k=top_k,
            filters={"name": {"$eq": detected_animal}},
            min_score=MIN_SCORE
        )

        if matches:
            return {
                "kbContext": render_matches(matches),
                "api_answer": 1,
                "confidence": infer_confidence(matches, exact_filter_match=True),
                "match_type": "animal_fact",
                "debug": debug if data.debug else None
            }

    # --------------------------------------------------
    # 3. FILTERED SEMANTIC SEARCH
    # --------------------------------------------------

    if pinecone_filter:
        matches = await semantic_search(
            query=query,
            top_k=top_k,
            filters=pinecone_filter,
            min_score=MIN_SCORE
        )

        if matches:
            if query_mode == "list":
                context = render_list(matches)
            else:
                context = render_matches(matches)

            return {
                "kbContext": context,
                "api_answer": 1,
                "confidence": infer_confidence(matches),
                "match_type": "filtered_semantic",
                "debug": debug if data.debug else None
            }

    # --------------------------------------------------
    # 4. GENERAL SEMANTIC SEARCH
    # --------------------------------------------------

    matches = await semantic_search(
        query=query,
        top_k=top_k,
        filters=None,
        min_score=MIN_SCORE
    )

    if matches:
        return {
            "kbContext": render_matches(matches),
            "api_answer": 1,
            "confidence": infer_confidence(matches),
            "match_type": "semantic",
            "debug": debug if data.debug else None
        }

    # --------------------------------------------------
    # 5. WIDE FALLBACK
    # --------------------------------------------------

    fallback_matches = await semantic_search(
        query=query,
        top_k=3,
        filters=None,
        min_score=None
    )

    if fallback_matches:
        return {
            "kbContext": render_matches(fallback_matches),
            "api_answer": 1,
            "confidence": "inferred",
            "match_type": "fallback",
            "debug": debug if data.debug else None
        }

    return {
        "kbContext": "",
        "kbAnswer": "No relevant animal information was found in the knowledge base.",
        "api_answer": 0,
        "confidence": "unknown",
        "match_type": "no_match",
        "debug": debug if data.debug else None
    }
