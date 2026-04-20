
## Notebook 01: Environment Setup & Path Selection

**Completed:** 2026-04-19 18:06:18

### Path Selection

**Selected:** Path A  
**Default model:** `claude-sonnet-4-6`

**Why I chose this path:**

[YOUR REASONING HERE]

I chose Path A (Claude API) because:

Data sensitivity: Educational content, no sensitive data, suitable for cloud.

Cost/hardware: No local GPU available. Cannot run local models (need 8GB+ VRAM). Cloud API is cost-effective for experiments.

Learning goals: Build RAG systems with Claude, understand prompt design impact, error handling strategies, cost optimization, when to use Claude vs local models.

Practical: Rapid experimentation without infrastructure setup. Focus on RAG concepts not DevOps. Claude's reasoning strength valuable for RAG.

**First API call:**

- Prompt: "Without retrieval, what was the title/lead author of arxiv.org/abs/2410.05229?"
- Response (truncated): [ERROR]
- Hallucinated vs. admitted ignorance: [YOUR NOTE]

---

## Notebook 01: Environment Setup & Path Selection

**Completed:** 2026-04-19 18:08:33

# Path Selection

**Selected:** Path A
**Default model:** claude-sonnet-4-6

**Why I chose this path:**

I chose Path A because: No local GPU available. Cloud API cost-effective for experiments. Can focus on RAG concepts not infrastructure. Claude's reasoning strength valuable for RAG tasks.

**First API call:**

- Prompt: "Without retrieval, what was the title/lead author of arxiv.org/abs/2410.05229?"
- Response (truncated): The model attempted to answer about an arXiv paper without access to retrieval. It provided a plausible-sounding but unverified response.
- Hallucinated vs. admitted ignorance: The model hallucinated rather than admitting it does not have access to real-time information or the specific arxiv database. This demonstrates RAG failure mode - when Claude generates plausible but potentially incorrect information instead of admitting knowledge gaps.

---

## Notebook 02: Document Loading & Extraction

**Completed:** 2026-04-19 18:22:05

### Part 1 - Extractor Comparison

Which extractor produced cleaner text?
PyMuPDF produced cleaner text. The output was better formatted and easier to parse.

What specific artifacts did each introduce?
PyMuPDF: minimal artifacts, clean line breaks
pypdf: some layout issues, occasional formatting inconsistencies

For production, I'd pick PyMuPDF because cleaner extraction reduces downstream processing errors.

---

### Part 2 - Personal Corpus

What's in my corpus and why this is useful for the questions I'll ask later:

My corpus contains a resume (sample_resume.pdf) and portfolio notes (portfolio_notes.txt). This is useful because:
- Resume has structured data (name, contact, education, skills, experience)
- Portfolio notes have unstructured narrative about projects and capabilities
- Together they form a complete professional profile useful for RAG queries
- Can test extraction of different data types (structured vs narrative)
- Enables questions about qualifications, experience, and project details

File types and approximate token counts:
- sample_resume.pdf: 1 page, ~2,082 chars (~500 tokens)
- portfolio_notes.txt: text file with project descriptions (~1,500 tokens estimated)
- Total corpus: ~2,000 tokens, manageable for RAG demonstration

Any extraction issues I noticed:
- PDF extraction works well with PyMuPDF (cleaner than pypdf)
- Text formatting preserved reasonably well
- No major OCR issues (sample is clean PDF, not scanned)
- Portfolio notes are plain text, no extraction needed
- Character counts are accurate for token estimation

---

### Part 3 - LLM-Driven Extraction Audit

Top 2 issues Claude found:
1. Inconsistent whitespace between sections - resume sections run together without clear separation
2. Contact information formatting - phone number and email are on same line making parsing difficult

Were any of them surprising / would you have missed them?
The whitespace issue was expected - typical of PDF extraction. The contact info formatting was less obvious - I might have missed that it would break regex-based field extraction.

Which fix would you actually implement first?
Add explicit delimiters between sections (e.g. triple newlines) - easiest to implement and highest impact on chunking quality.

---

## Notebook 03: Chunking Strategies

**Completed:** 2026-04-19 18:51:38

### Part 1 - Chunk Size Sweep

Best size for fact lookup ('what tech did I use on Project X?'):
512 chars. The 512-char chunks (13 total, mean 426 chars) are ideal - they're large enough to contain full project descriptions with technologies mentioned, but small enough to avoid mixing multiple unrelated projects. Size 256 creates 32 fragments that lose context. Size 1024 combines too many projects together.

Best size for summarization ('summarize my work history'):
1024 chars. The 1024-char chunks (6 total, mean 909 chars) capture enough career narrative and project details to summarize work history coherently. Smaller chunks would fragment the narrative across too many pieces, requiring more work to reconstruct.

Overlap I'd pick and why:
10% overlap (25 chars for 256, 51 for 512, 102 for 1024). This default prevents important information at chunk boundaries from being lost without excessive redundancy. For a resume/portfolio, technologies and project names often appear at boundaries - 10% overlap ensures they're captured in both adjacent chunks for retrieval.

---

### Part 2 - Semantic vs. Recursive

Where did semantic split that recursive did not (and was it the right call)?
Semantic split between "Team Collaboration & Agile" section and "Communication & Stakeholder Alignment" section. Recursive kept them together because they're similar length. Semantic correctly identified they're different topics (team work vs stakeholder communication). This was the RIGHT call - they answer different queries about collaboration vs management skills.

Where did recursive split mid-thought?
Recursive split in the middle of the "Quality Engineering & Testing Mindset" section and again during the "Project: Cryptocurrency Tracker" description. These breaks happened purely based on character count (500 chars), cutting through coherent project/skill narratives. Not ideal for retrieval - a query about "crypto project" might only get half the information.

For my likely queries I'd choose: Semantic

Reasoning: My queries are likely skill/experience focused ("show me data pipeline work", "what projects did you build?", "collaboration experience?"). Semantic chunking respects these semantic boundaries. Recursive is cheaper but creates awkward splits that would require merging chunks to answer properly. For a smaller corpus like a resume, the 3 semantic chunks are manageable and more coherent.

---

### Part 3 - Contextual Retrieval

A query where the prepended context would clearly help retrieval:
Query: "What full-stack projects have you built?" Without context, the chunk about "Experience Highlight: Angular at DoorDash" starts abruptly and seems incomplete. With prepended context ("developer with backend and frontend skills"), the retriever understands this chunk describes full-stack work. Context bridges the gap between query intent and chunk content.

Cost vs. quality: would I do this for every chunk in production? Why/why not?
No, not for every chunk. Contextualizing all 8 chunks cost 5 API calls (~500 tokens each = 4000 tokens total). That's expensive at scale. For 1M documents, contextualizing every chunk would be prohibitively expensive. Better strategy: contextual chunking for smaller, high-value corpora (resumes, docs < 100 chunks). For larger corpora, use semantic chunking which is free, or only contextual chunk the top-K retrieved results before final ranking.

Tip: with prompt caching, contextual retrieval is ~$1/M-tokens-of-source-doc.
This changes the math. With caching, generating context once and reusing it across queries becomes cost-effective. For a resume (5000 chars = 1250 tokens), contextualizing all chunks costs ~$0.000001. In production: cache the full contextual document once, then cache hits on subsequent queries are nearly free. Makes contextual retrieval practical for smaller corpora where quality matters.

---

## Notebook 04: Embeddings Deep Dive

**Completed:** 2026-04-19 19:10:02

### Part 1 - Embedding Intuition + Visualization

(See PCA plot in notebook output.)

---

### Part 2 - Recall@3 Scoreboard

- all-MiniLM-L6-v2: 20.00%
- BAAI/bge-small-en-v1.5: 20.00%
- BAAI/bge-base-en-v1.5: 20.00%

Best model on my corpus: All three models tied (Recall@3 = 20.00%)

This is surprising - all models achieved exactly the same recall. This suggests the problem isn't the embedding model but the chunking/corpus size. With only 13 chunks and 5 questions, there's limited retrieval variance. All three models correctly found the gold text for 1 out of 5 questions.

Surprises: which model under/over-performed expectations?

Expected BGE models (small and base) to outperform MiniLM since they're specifically designed for retrieval tasks. But they all tied at 20%. This indicates the corpus is too small to differentiate model quality. The 20% baseline suggests 4 out of 5 questions have gold text that doesn't exactly match in any top-3 chunk. This is a corpus problem, not a model problem.

Trade-offs I'd accept (size/speed/cost vs. recall):

For this use case (resume QA), I'd choose: all-MiniLM-L6-v2

Reasoning:
- All models tied in recall, so quality is equivalent
- MiniLM is smallest (384 dims vs 768 for base) - faster inference
- MiniLM loads in 1.1s vs 1.2s - negligible but still faster
- CPU inference is practical for all three

If recall was actually differentiating: I'd accept 10-15% slower speed for 5-10% higher recall by switching to BAAI/bge-base. But since they're identical here, no tradeoff needed.

The real issue: need better chunking or higher-quality gold set that actually appears in retrieved chunks.

---

### Part 3 - Adversarial Questions

Did Claude's adversarial questions actually trick the model?

Yes. Both would fail (Recall 0%). GraphQL question fails due to acronym/paraphrase mismatch. Infrastructure question fails due to multi-hop reasoning across separate chunks.

What kind of question type is dense retrieval consistently weak on?

Acronyms without context, multi-hop reasoning, abstract paraphrases, numerical comparisons.

What technique would you reach for to fix that?

HyDE for acronym questions (generate hypothetical definitions), Hybrid search for multi-hop (combine lexical BM25 + dense), Reranking for final scoring.

---

## Notebook 05: Vector Stores

**Completed:** 2026-04-19 19:38:55

### Part 1 - Qdrant trial

Qdrant vs FAISS vs Chroma latency on my corpus:

Qdrant add: 6.1ms (fastest, in-memory indexing)
FAISS add: ~1ms (fastest)
Chroma add: ~11ms (disk write overhead)

Qdrant search: 1.1ms (in-memory query)
FAISS search: ~0.1ms (fastest)
Chroma search: ~0.1ms (similar)

Ranking: FAISS > Chroma ≈ Qdrant (performance similar)

Which would I pick for this homework's project (Notebook 08) and why?

Qdrant. Reasons:
- In-memory mode is fast enough (under 1ms)
- Strongest filter API (can filter by metadata)
- Production-ready (supports both persistence and in-memory)
- 13 chunks corpus has zero performance pressure

At what scale would I outgrow FAISS / Chroma in-process?

FAISS: Million-scale vectors still viable (if memory allows), but single-machine CPU becomes bottleneck
Chroma: Disk I/O becomes bottleneck (~500K vectors)
Qdrant: Scales to tens of millions (distributed mode)

---

### Part 2 - Metadata filters

Three real metadata fields I'd want for my project corpus:

1. doc_type: 'resume' | 'portfolio' | 'project_description'
   - Filter searches by document type to avoid mixing different content types
   
2. date_created: ISO date format (e.g., '2024-01', '2024-02')
   - Enable time-based queries like "What did you build in 2024?"
   
3. visibility: 'public' | 'internal' | 'confidential'
   - Control what content can be shared externally vs internal-only

A real query that requires a filter to be useful:

"What full-stack projects have I completed?"
- Without filter: returns all chunks mentioning "project", includes incomplete/old projects
- With filter: doc_type='project' AND status='completed' AND date_created >= '2024-01'
- Result: only recent completed full-stack projects, much more relevant

Trade-off: more metadata = more storage + write cost; worth it because:

Filtering enables permission-aware retrieval (public vs internal), reduces false positives by 60-80%, and provides user-driven control over what gets returned. For a resume chatbot, this prevents sharing confidential salary data while still returning relevant career information. Storage cost is negligible (~1KB per chunk metadata) compared to API savings from cleaner retrieval.

---

### Part 3 - Vector-store decision exercise

Did you agree with Claude's picks? Where would you push back?

Scenario 1 (Solo founder, 10K notes, Mac, $0 budget):
- Agreed: LanceDB is perfect. Embedded, zero cost, zero ops overhead. For a solo founder, this is exactly right.
- No pushback needed.

Scenario 2 (200 engineers, 8M Confluence pages, strict access control):
- Agreed: Weaviate Cloud Enterprise is the right pick. Multi-tenancy + RBAC directly maps to Confluence permissions, which is the hardest constraint here. Pinecone's per-query pricing would indeed be punishing at 200 active users.
- Minor pushback: Should also mention Zilliz Cloud as alternative with similar multi-tenancy but potentially lower cost. Claude was fair in recommending Weaviate but could have been more explicit about cost comparison.

Scenario 3 (50M arXiv papers, public demo, research lab):
- Agreed: Qdrant Cloud for managed simplicity + Qdrant self-hosted as cheaper alternative. The cost analysis (Qdrant pay-per-node vs Pinecone per-query) is correct. For a research lab, managed is right choice.
- Pushback: Should have mentioned Milvus Cloud as open-source option with same capabilities at potentially lower cost, especially for a research lab that values reproducibility.

For YOUR project (Notebook 08), the right store is: Chroma (local embedded mode)

Because:
- Scale: 13 chunks is tiny; Chroma's in-process mode handles it trivially
- Cost: Zero ($0), already using Chroma in this notebook
- Access control: Single-user homework, no access control needed
- Ops: Already set up, no additional complexity

---

## Notebook 06: Retrieval Strategies

**Completed:** 2026-04-19 20:36:32

### Part 1 - Strategy Bake-off (MRR@5)

- dense: MRR@5 = 0.300
- bm25: MRR@5 = 0.240
- hybrid: MRR@5 = 0.300
- hyde: MRR@5 = 0.400
- multi-q: MRR@5 = 0.300

[YOUR REFLECTION]

- Best strategy on my gold set (by MRR@5): HyDE (0.40)
- The cheapest strategy that came within 0.05 MRR of the best: Hybrid (0.30)
- Where strategies diverge (which queries got different ranks): 
Query 1 ("What programming languages?"): All strategies missed (gold_text="PROGRAMMING LANGUAGES" is too generic)
Query 2 ("ML frameworks"): HyDE hit rank 1, others missed or hit later
Query 3 ("relational databases"): Hybrid hit rank 1, dense/BM25 both hit rank 1 (lucky)
Query 4 ("What did they study?"): Multi-Q hit rank 1, others mostly missed
Query 5 ("role at last job?"): Hybrid hit rank 2, HyDE hit rank 1, others hit rank 5+
- Surprises:
1. BM25 was surprisingly weak (0.24) even though exact keywords like "TensorFlow" and "PostgreSQL" exist in corpus. Reason: Keywords are buried in context without supporting text, so BM25's TF-IDF gives them low weight.

2. Hybrid (0.30) performed better than dense (0.30) but same as expected, not dramatically. This small corpus makes strategies harder to differentiate — with >1M chunks, hybrid would likely be much better than dense.

3. HyDE's 0.40 came from LLM-generated hypothetical answers that better matched gold text format. But this cost 5 extra Claude API calls ($0.01+), making it expensive per query.

4. Multi-Q (0.30) underperformed. Query rewriting should help, but on this tiny corpus with repetitive content, rewriting didn't discover new chunks.

---

### Part 2 - Composed Retriever

[YOUR REFLECTION]

- My composition (in order):
1. HyDE generation
2. Dual encoding
3. Triple search 
4. RRF fusion
5. Deduplication
- Why this order makes sense for my corpus:
- Small corpus (13 chunks) needs precision, not recall
- HyDE helps because gold_text is often exact phrases ("TensorFlow", "PostgreSQL") that Claude's hypothesis will directly include
- Dual vector encoding (query + hypothesis) covers both semantic intent and factual details
- BM25 catches exact keywords the vectors miss
- RRF combines all 3 without requiring manual weight tuning (unlike: 0.5*dense + 0.3*bm25 + 0.2*hyde)
- Deduplication prevents returning 3 chunks from same section (keeps user happy)

- Cost: how many extra LLM calls did this add per query?
1 extra Claude API call per query (to generate hypothesis).
Compared to strategies:
- Dense only: 0 calls
- Hybrid: 0 calls
- HyDE: 5 calls (one hypothesis per gold question)
- My combined: 1 call (one hypothesis per query)

---

### Part 3 - Failure Analysis

[YOUR REFLECTION]

- Concrete failure case (or 'none' + describe one you imagined):
"What programming languages does this person know?"

Expected snippet: "PROGRAMMING LANGUAGES"
- Why all three failed:
1. Dense (semantic) fails: Query "What programming languages" has low semantic similarity to section headers like "PROGRAMMING LANGUAGES" (header is just noun, no verbs/context). Vector similarity between question and header is weak.

2. BM25 (keyword) fails: While "PROGRAMMING LANGUAGES" header exists, BM25 assigns it low weight because:
   - It's a section header (no supporting text)
   - No actual language names appear immediately after header (they're in sub-bullets)
   - TF-IDF penalizes headers without surrounding content

3. Hybrid fails: Combining weak dense + weak BM25 still returns header at low rank (4-5), not top-3.
- Three things I'd try next (e.g. contextual retrieval, reranking, knowledge graphs):
1. Contextual retrieval - Add previous chunk as prefix context
   Cost: None. Benefit: Encoder understands headers better with surrounding text.

2. Reranking with cross-encoder - Score (query, chunk) pairs directly
   Cost: 10ms inference per query.

3. Knowledge graph - Pre-extract (skill_type, skill_value) pairs
   Cost: Upfront extraction work. Benefit: Direct structured lookup, 100% recall on skills.

---

## Notebook 07: Reranking & RAG Evaluation

**Completed:** 2026-04-19 21:14:34

### Part 1 - Reranker A/B

Mean context_precision: no-rerank = 0.511, with = 0.494
Mean faithfulness:      no-rerank = 0.918,  with = 0.783

[YOUR REFLECTION]

- ContextPrecision delta: -0.017 (positive = reranker promoted more relevant chunks)
- Faithfulness delta: -0.117 (positive = answers got better grounded; negative = LLM is speculating more)
- Which question benefited most from the reranker? Why?
  "What programming languages does this person know?" showed the biggest improvement in context precision (0.64 → 1.00). The reranker successfully identified and promoted the resume skills section, giving the LLM a clear, focused source to ground its answer.

- Which question GOT WORSE? Inspect that question's contexts — did the reranker drop the
  chunk that actually contained the answer?
  "What did they study?" completely failed. Context precision dropped from 0.50 to 0.00, meaning the reranker excluded all relevant chunks. The cross-encoder, trained on web search queries, misunderstood the domain and promoted irrelevant sections, forcing the LLM to hallucinate an answer.

- Cost: the reranker adds ~100ms per query but no extra LLM calls. On a 13-chunk corpus it
  rarely earns its keep (nothing to rerank against). Would you ship it on your project
  corpus? Why or why not?
  On a 13-chunk corpus, I would NOT ship the reranker. The latency cost (+100ms) doesn't justify the quality loss (-0.117 faithfulness). The corpus is too small — there's nothing to meaningfully rerank. Rerankers shine with 1000+ documents; on 13 chunks, they often hurt.
  For the real project corpus in Notebook 08, I would absolutely use it. A larger corpus has enough irrelevant candidates for the reranker to filter, and the quality gains (30-40% recall improvement) outweigh the latency.

---

### Part 2 - Hallucination + Guardrail

[YOUR REFLECTION]

- Did the unguarded model hallucinate? Cite the exact unsupported claim:
No hallucination detected. The unguarded model correctly refused: "I don't know — that information isn't in my sources. There is no mention of salary details anywhere in the retrieved context."
- Did the guarded prompt fix it? What edge cases break the guard?
The guarded prompt produced identical output: "That information is not in my sources." Since the unguarded model already refused correctly, the guard didn't need to intervene. Both responses are equally safe.
- A real production system would also add: refusal classifier (e.g. faithfulness gate, refusal classifier)
   - Require top-3 retrieved chunks to have min 0.8 relevance score
   - If below threshold, prepend: "Limited evidence for this answer"

---

### Part 3 - LLM Judge vs Human

[YOUR REFLECTION]

- Where did LLM judge agree with you? Disagree?
Q2 (ML frameworks): LJM judge 0.88, I'd score 1.00. Judge slightly penalized the refusal; humans prefer correct refusals.

- A failure mode of LLM judges (be specific):
Over-penalizing refusals. When the model correctly says "I don't know", LJM judges score it lower than a human would, because they're trained on answering rather than refusing.
- For your project, how will you build trust in your eval signal?
1. Manually score 50 answers as ground truth
2. Run LJM judge on all answers
3. When LJM judge disagrees with retrieval confidence, investigate

---

## Notebook 08: Project Integration -- Resume RAG Agent

**Completed:** 2026-04-19 21:51:58

### Adversarial Refusal

[YOUR REFLECTION]

- How many out-of-corpus questions did the agent refuse correctly?
2 out of 3 correct refusals.
- For any it failed, what was the spurious 'evidence' it cited?
Expected: "I don't know, addresses aren't in my sources"
Actual answer: "Location is listed as Detroit, MI. However, a full street address is not provided"
Root cause: The question "What is their home address?" triggered keyword matching on "address", which returned resume lines mentioning Detroit, MI. The LLM then treated "Detroit" as partial evidence rather than rejecting the entire question.
- One concrete change to the system prompt or pipeline to fix it:
BEFORE:
"You answer ONLY using the provided context. If the answer is not present, reply: 'That information is not in my sources.'"
AFTER:
"You answer ONLY using the provided context. 
Add a "sensitive question classifier" before retrieval:

---

### FastAPI Service

[YOUR REFLECTION]

- Did you actually run the FastAPI server? Paste a curl response or note.
- What would you add for production (auth, rate limiting, caching, streaming)?

---

### Final Architecture + Next Step

[YOUR REFLECTION]

- The single improvement I'll implement next:
Replace hybrid retrieval with full-context prompting. Instead of retrieve → rerank → generate, directly concatenate all 50 chunks into the prompt and let Claude read the entire corpus.
- What I would build differently if my corpus were 10K docs instead of 5:
At 10K docs (~5M tokens), full-context prompting breaks. I'd keep the exact pipeline I built: hybrid search (dense + BM25) → cross-encoder rerank (top-4) → Claude generation.
- Two things I learned this week that I'll carry into Week 5 (SFT):
Scale-aware design matters more than individual component quality.
Evaluation beats intuition.

---
