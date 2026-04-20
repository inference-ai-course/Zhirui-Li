"""
RAG Evaluation -- Week 4

Reference-free RAG evaluation, RAGAS-style. We implement four core metrics
locally with an LLM judge so students don't need an OpenAI key:

  * faithfulness        - Are all answer claims supported by retrieved context?
  * answer_relevancy    - Does the answer actually address the question?
  * context_precision   - Of retrieved chunks, how many are actually relevant?
  * context_recall      - Of the ground-truth answer, how much was retrievable?
                          (requires a reference answer)

For full production eval, plug in:
  - RAGAS         (pip install ragas)         — LLM-judge metrics + dataset gen
  - DeepEval      (pip install deepeval)       — pytest integration / CI gates
  - TruLens       (pip install trulens-eval)   — dashboards + tracing
"""

import json
import re
from typing import Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------

CLAIM_EXTRACT_PROMPT = """Decompose the following answer into a list of atomic
factual claims. Output ONLY a JSON array of strings, no commentary.

Answer:
{answer}

JSON array of claims:"""


CLAIM_VERIFY_PROMPT = """Given the context below, decide whether the claim
can be inferred from it. Answer ONLY "YES" or "NO" — nothing else.

Context:
{context}

Claim: {claim}

Answer:"""


def faithfulness(
    llm_client,
    question: str,
    answer: str,
    contexts: List[str],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Faithfulness = supported_claims / total_claims.
    A score of 1.0 means every claim in the answer is grounded in context.
    """
    # 1. Extract claims
    resp = llm_client.generate(
        prompt=CLAIM_EXTRACT_PROMPT.format(answer=answer),
        model=model, max_tokens=600, temperature=0.0,
    )
    claims = _safe_json_list(resp.get("content", ""))
    if not claims:
        return {"score": 0.0, "claims": [], "supported": []}

    # 2. Verify each claim against the joint context
    joint = "\n\n".join(contexts)
    supported = []
    for claim in claims:
        v = llm_client.generate(
            prompt=CLAIM_VERIFY_PROMPT.format(context=joint, claim=claim),
            model=model, max_tokens=8, temperature=0.0,
        )
        verdict = (v.get("content", "") or "").strip().upper().startswith("YES")
        supported.append(verdict)

    score = sum(supported) / len(supported) if supported else 0.0
    return {
        "score": round(score, 3),
        "claims": claims,
        "supported": supported,
    }


# ---------------------------------------------------------------------------
# Answer relevancy (via reverse-question generation)
# ---------------------------------------------------------------------------

REVERSE_Q_PROMPT = """Generate a question that the following answer is
designed to answer. Output ONLY the question, no preamble.

Answer:
{answer}

Question:"""


def answer_relevancy(
    llm_client,
    embedding_model,
    question: str,
    answer: str,
    n_questions: int = 3,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate N reverse-questions from the answer, embed them all, average
    cosine sim with the original question. Higher = more on-topic.
    """
    import numpy as np
    generated = []
    for _ in range(n_questions):
        r = llm_client.generate(
            prompt=REVERSE_Q_PROMPT.format(answer=answer),
            model=model, max_tokens=80, temperature=0.7,
        )
        if "error" not in r:
            q = r["content"].strip().splitlines()[0]
            generated.append(q)
    if not generated:
        return {"score": 0.0, "generated_questions": []}

    q_vec = embedding_model.encode(question)
    g_vecs = embedding_model.encode(generated)
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-12)
    g_vecs = g_vecs / (np.linalg.norm(g_vecs, axis=1, keepdims=True) + 1e-12)
    sims = g_vecs @ q_vec
    return {
        "score": round(float(np.mean(sims)), 3),
        "generated_questions": generated,
    }


# ---------------------------------------------------------------------------
# Context precision
# ---------------------------------------------------------------------------

CONTEXT_RELEVANT_PROMPT = """You are evaluating retrieval quality. Decide if
the following context is RELEVANT to answering the question. Answer ONLY
"YES" or "NO".

Question: {question}

Context: {context}

Answer:"""


def context_precision(
    llm_client,
    question: str,
    contexts: List[str],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Of the retrieved chunks, what fraction are actually relevant?
    Weighted by rank (earlier hits matter more, like Precision@k mean).
    """
    if not contexts:
        return {"score": 0.0, "verdicts": []}

    verdicts = []
    for ctx in contexts:
        r = llm_client.generate(
            prompt=CONTEXT_RELEVANT_PROMPT.format(question=question, context=ctx),
            model=model, max_tokens=8, temperature=0.0,
        )
        verdicts.append((r.get("content", "") or "").strip().upper().startswith("YES"))

    # Mean of Precision@k for each k where verdict[k-1] is True
    precisions = []
    relevant_so_far = 0
    for k, v in enumerate(verdicts, start=1):
        if v:
            relevant_so_far += 1
            precisions.append(relevant_so_far / k)
    score = sum(precisions) / sum(verdicts) if any(verdicts) else 0.0
    return {
        "score": round(score, 3),
        "verdicts": verdicts,
        "fraction_relevant": round(sum(verdicts) / len(verdicts), 3),
    }


# ---------------------------------------------------------------------------
# Context recall (needs a reference answer)
# ---------------------------------------------------------------------------

GROUND_CLAIM_ATTRIB_PROMPT = """Given the reference answer, decompose it into
atomic claims, and for each claim decide whether it can be inferred from the
retrieved context. Output a JSON array of objects:
[{{"claim": "...", "supported": true/false}}, ...]
Do not output anything else.

Reference answer:
{reference}

Retrieved context:
{context}
"""


def context_recall(
    llm_client,
    question: str,
    reference_answer: str,
    contexts: List[str],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Context recall = fraction of reference-answer claims supported by context.
    Requires a ground-truth reference (synthetic or human).
    """
    joint = "\n\n".join(contexts)
    r = llm_client.generate(
        prompt=GROUND_CLAIM_ATTRIB_PROMPT.format(
            reference=reference_answer, context=joint,
        ),
        model=model, max_tokens=800, temperature=0.0,
    )
    items = _safe_json_objects(r.get("content", ""))
    if not items:
        return {"score": 0.0, "items": []}
    supported = sum(1 for it in items if it.get("supported"))
    return {
        "score": round(supported / len(items), 3),
        "items": items,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json_list(text: str) -> List[str]:
    text = text.strip()
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
        return [str(x).strip() for x in data if x]
    except Exception:
        return []


def _safe_json_objects(text: str) -> List[Dict[str, Any]]:
    text = text.strip()
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
        return [d for d in data if isinstance(d, dict)]
    except Exception:
        return []


def evaluate_rag(
    llm_client,
    embedding_model,
    question: str,
    answer: str,
    contexts: List[str],
    reference_answer: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full suite. Skips context_recall when no reference provided."""
    print(f"  Evaluating: '{question[:60]}...'")
    f = faithfulness(llm_client, question, answer, contexts, model=model)
    r = answer_relevancy(llm_client, embedding_model, question, answer, model=model)
    p = context_precision(llm_client, question, contexts, model=model)
    out = {
        "question": question,
        "faithfulness":      f["score"],
        "answer_relevancy":  r["score"],
        "context_precision": p["score"],
        "details": {"faithfulness": f, "answer_relevancy": r, "context_precision": p},
    }
    if reference_answer:
        cr = context_recall(llm_client, question, reference_answer, contexts, model=model)
        out["context_recall"] = cr["score"]
        out["details"]["context_recall"] = cr
    return out
