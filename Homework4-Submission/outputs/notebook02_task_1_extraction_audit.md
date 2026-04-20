# Task 1: Extraction Audit

**Notebook:** 02  
**Completed:** 2026-04-19 18:21:54

## User Prompt

```
You are a RAG engineer auditing extracted text quality.

Below is the first ~1500 characters extracted from a document. Identify
specific issues that will hurt downstream chunking/retrieval (e.g. missing
spaces, broken hyphens, two-column flattening, figure captions inline, 
table cells smashed together, page numbers, repeated headers).

For each issue: name it, give an example from the text, and suggest a fix.

Extracted text:
---
## Career Summary

I'm a Python developer with a strong foundation in backend development, frontend components, and scalable data workflows. I've worked in fast-paced environments like DoorDash and Knewton, where I contributed to production-grade systems, internal tools, and data infrastructure.

## Experience Highlight: Angular at DoorDash

At DoorDash, I contributed new Angular components to the customer-facing web app. These enhancements directly improved average user time-on-page by 2 minutes, reflecting more engaging UI/UX outcomes. My focus was not only on front-end polish but also aligning with backend services through REST APIs.

## Team Collaboration & Agile

I worked in a 6-person agile team, regularly collaborating with product managers to prioritize features that delivered measurable customer and business impact. One example was a new data pipeline I developed that led to a product idea that ultimately increased revenue by 6%.

## Quality Engineering & Testing Mindset

At both DoorDash and Knewton, I emphasized code quality. I implemented extensive test coverage on new features at DoorDash, which cut customer complaints by 23%. Earlier, at Knewton, I helped build a Selenium-based testing infrastructure that reduced monthly bug reports by 11%.

## Data Pipelines & Product Insight

One of my biggest technical wins was ingesting and transforming large datasets at DoorDash, which revealed user behavior patterns that led to product feature proposals. I built robust pip
---
```

## Response

## Audit Report: Extracted Text Quality

---

### Issue 1: Abrupt Mid-Word Truncation
- **Name:** Hard truncation at extraction boundary
- **Example:** `"I built robust pip"` — the word "pip" is cut off mid-token (likely "pipelines")
- **Fix:** Extend the extraction window or implement overlap buffering between chunks so sentences are never split mid-word. Flag any chunk ending without punctuation for review.

---

### Issue 2: No Issues With Spacing (Positive Note)
- **Name:** Word spacing — acceptable
- **Example:** All words appear properly space-delimited throughout
- **Fix:** No action needed here, but confirm this holds in later pages.

---

### Issue 3: Section Headers May Interfere With Chunking
- **Name:** Markdown headers used as semantic boundaries without contextual carry-over
- **Example:** `## Data Pipelines & Product Insight` introduces a section, but if chunked at the header boundary, the following sentence loses its subject context (who built what, where)
- **Fix:** Include the preceding header text in each chunk's metadata or prepend it as a context prefix during chunking (e.g., store `section: "Data Pipelines & Product Insight"` as a metadata field).

---

### Issue 4: Implicit Entity References Across Sections
- **Name:** Pronoun/entity resolution broken by section splits
- **Example:** `"At both DoorDash and Knewton, I emphasized code quality"` — if this chunk is retrieved in isolation, "I" has no grounded identity (no name appears in the extracted text)
- **Fix:** Inject document-level metadata (candidate name, document type) into every chunk so retrievers can resolve entity references without needing cross-chunk context.

---

### Issue 5: Numeric Claims Without Units of Time Context
- **Name:** Decontextualized metrics
- **Example:** `"increased revenue by 6%"` and `"cut customer complaints by 23%"` — no time period or baseline is stated
- **Fix:** This is a source content issue, not a pure extraction issue, but flag these spans during post-processing so a retrieval system doesn't over-weight unqualified statistics.

---

### Summary Table

| # | Issue | Severity | Fixable at Extraction? |
|---|-------|----------|----------------------|
| 1 | Mid-word truncation | 🔴 High | Yes |
| 2 | Word spacing | ✅ None | N/A |
| 3 | Header/chunk boundary loss | 

## Metadata

- **Model:** claude-sonnet-4-6
- **Input tokens:** 448
- **Output tokens:** 600
- **Total tokens:** 1,048

---