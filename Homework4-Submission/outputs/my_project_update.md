# Week 4 -- Project Update: My Resume RAG Assistant

**Path:** A  
**Default model:** `claude-sonnet-4-6`

## 1. Architecture

- **corpus**: [describe: source files, total docs, total tokens]
- **extractor**: [PyMuPDF / pypdf / other]
- **chunker**: [recursive 500/50 / semantic / contextual / other]
- **embedding**: [all-MiniLM-L6-v2 / BGE / OpenAI / other]
- **vector_store**: [FAISS / Chroma / Qdrant / pgvector]
- **retriever**: [dense / hybrid + RRF / HyDE]
- **reranker**: [none / MS-MARCO MiniLM / BGE / FlashRank / Cohere]
- **generator**: [claude-sonnet-4-6 / qwen3.5:27b / other]
- **eval_metrics**: [faithfulness, context_precision, context_recall]

## 2. Demo Q&A

**Q1: What Python projects has this person contributed to?**

A: Based on the retrieved context, here are the Python-related projects and contributions this person has been involved in:

1. **RESTful APIs at Knewton** Worked alongside another developer to implement RESTful APIs in **Django**, which helped the internal analytics team increase reporting speed by 24%. [sample_resume.pdf]

2. **Data Pipelines at DoorDash** Built and maintained data pipelines using ingested data, which led to discovering an opportunity for a new site feature, boosting revenue by 6%. [sample_resume.pdf]

3. **Cryptocurrency Tracker (Solo Project)** Built a full-stack application featuring API integrations for real-time crypto price data, **PostgreSQL** for storage, and **D3.js** for dynamic chart visualizations. [portfolio_notes.txt]

4. **Open-Source Python Utilities** Maintains a personal GitHub where they contribute to open-source Python utilities and experiment with automation tools. [portfolio_notes.txt]

5. **Django Tutorials**  Occasionally writes tutorials on using **Django** with cloud-native databases. [portfolio_notes.txt]

- Sources: ['portfolio_notes.txt', 'portfolio_notes.txt', 'sample_resume.pdf', 'sample_resume.pdf']
- Faithfulness: 1.00, ContextPrecision: 0.58

---
**Q2: List the main technical skills with approximate years of experience.**

A: Based on the available context, here are the main technical skills listed. However, I should note that **exact years of experience per skill are not explicitly stated** in the sources only employment dates are provided, so I can offer rough estimates based on work history:

---

### Technical Skills (with approximate experience estimates):

| Skill | Notes on Experience |
|---|---|
| **Python (Django)** | Current role at DoorDash started Sept 2017 ~7+ years |
| **SQL (PostgreSQL, Oracle)** | Used at DoorDash and in personal projects |
| **AWS (Redshift, S3)** | Used in data pipeline work at DoorDash |
| **JavaScript (Angular)** | Used at DoorDash for frontend components |
| **HTML/CSS** | Listed as a core skill |
| **REST APIs (GraphQL)** | Listed as a core skill |

---

**Important caveat:** The sources list these skills and confirm they were used professionally, but do **not** break down specific years of experience for each individual skill. The estimates above are inferred from employment timelines, not explicitly stated.

[sample_resume.pdf] [portfolio_notes.txt]

- Sources: ['sample_resume.pdf', 'portfolio_notes.txt', 'portfolio_notes.txt', 'portfolio_notes.txt']
- Faithfulness: 0.88, ContextPrecision: 1.00

---
**Q3: Which projects involved NLP or AI?**

A: Based on the retrieved context, I don't know that information isn't in my sources. The only project specifically described in the available context is the **Cryptocurrency Price Tracker**, which involved API integrations, PostgreSQL, and D3.js visualizations. No projects involving NLP or AI are mentioned. [sample_resume.pdf] [portfolio_notes.txt]

- Sources: ['sample_resume.pdf', 'portfolio_notes.txt', 'sample_resume.pdf', 'portfolio_notes.txt']
- Faithfulness: 0.75, ContextPrecision: 0.00

---
**Q4: What was their most recent job title and company?**

A: Based on the retrieved context, their most recent job title was **Python Developer** at **DoorDash**, where they have been working from September 2017 to current, based in Detroit, MI. [sample_resume.pdf]

- Sources: ['portfolio_notes.txt', 'sample_resume.pdf', 'portfolio_notes.txt', 'sample_resume.pdf']
- Faithfulness: 0.83, ContextPrecision: 1.00

---
**Q5: What relational databases do they have experience with?**

A: Based on the retrieved context, they have experience with the following relational databases:

- **PostgreSQL** listed under skills and also used in the Cryptocurrency Tracker project for storage
- **Oracle** listed under skills

Both are mentioned as part of their SQL expertise. [sample_resume.pdf] [portfolio_notes.txt]

- Sources: ['sample_resume.pdf', 'portfolio_notes.txt', 'portfolio_notes.txt', 'sample_resume.pdf']
- Faithfulness: 0.75, ContextPrecision: 1.00

---

## 3. Adversarial Test (TODO 1)

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

## 4. FastAPI Service (TODO 2)

Generated `outputs/main.py`. To run:
```bash
cd outputs && uvicorn main:app --reload
curl "http://localhost:8000/search?q=What+ML+frameworks+do+they+know"
```

[YOUR REFLECTION]

- Did you actually run the FastAPI server? Paste a curl response or note.
- What would you add for production (auth, rate limiting, caching, streaming)?

## 5. Architecture Critique + Next Step (TODO 3)

[YOUR REFLECTION]

- The single improvement I'll implement next:
Replace hybrid retrieval with full-context prompting. Instead of retrieve rerank generate, directly concatenate all 50 chunks into the prompt and let Claude read the entire corpus.
- What I would build differently if my corpus were 10K docs instead of 5:
At 10K docs (~5M tokens), full-context prompting breaks. I'd keep the exact pipeline I built: hybrid search (dense + BM25) cross-encoder rerank (top-4) Claude generation.
- Two things I learned this week that I'll carry into Week 5 (SFT):
Scale-aware design matters more than individual component quality.
Evaluation beats intuition.

## 6. Cost

Total tokens this week: 8,342  
Total cost: $0.0450
