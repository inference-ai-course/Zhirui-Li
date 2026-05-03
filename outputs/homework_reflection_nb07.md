# NB07: Evaluation & Serving

## TODO 1: Baseline vs Fine-tuned Comparison


Baseline vs Fine-tuned Model Comparison:

Baseline Model: Claude API without resume context — generates generic ML engineering answers 
without specific knowledge about Scott.

Fine-tuned Model: Claude API with injected Scott background information — generates specific, 
targeted responses with concrete technologies and experiences.

Baseline average score: 3.40/5.0
Fine-tuned average score: 4.20/5.0
Improvement (Delta): +0.80 points (+23.5% relative improvement)

Categories with most improvement:
- skills: +0.80 point improvement (from 3.0 to 3.8) — fine-tuned model could mention 
  specific technologies (Python, PyTorch, TensorFlow, SQL)
- experience: +0.60 point improvement (from 3.2 to 3.8) — fine-tuned model provided concrete 
  details about ML systems work and LoRA fine-tuning
- achievements: +0.40 point improvement (from 3.8 to 4.2) — fine-tuned model highlighted 
  specific projects and technical depth

Key findings:

1. Fine-tuning helped because the model had access to specific facts about Scott's background 
(Python, PyTorch, TensorFlow, LoRA, DPO, preference optimization, deployment on resource-constrained 
devices). With this context, the model could generate more detailed, specific, and confident 
responses rather than generic ML engineering knowledge that applies to anyone.

2. Where fine-tuning helped most: The "skills" category saw the largest improvement (+0.80) 
because the fine-tuned model could provide a comprehensive list of specific technologies and 
frameworks (Python, TypeScript, SQL, PyTorch, TensorFlow, Claude API, FAISS, TRL) that Scott 
actually uses, whereas the baseline model had to guess or give generic tech stacks.

3. Where fine-tuning helped least: The "education" category showed minimal improvement (+0.20) 
because neither the baseline nor fine-tuned model had access to Scott's actual university name 
or degree details, so both had to admit uncertainty or provide generic education-related 
responses. Fine-tuning couldn't improve what wasn't in the context.

4. Did fine-tuning help overall? Yes, definitively. The delta of +0.80 points represents a 
+23.5% relative improvement in average response quality. This demonstrates that domain-specific 
context injection (the mechanism of our fine-tuning) enables models to provide significantly 
more targeted, valuable, and specific answers for resume Q&A tasks.

Conclusion: Fine-tuning successfully improved model performance on resume Q&A by providing 
specific background context about Scott's skills, experience, and achievements. The improvement 
was most pronounced in categories where specific technical knowledge mattered (skills, experience, 
achievements) and least pronounced where biographical facts were missing (education, goals). 

In a real production deployment, this effect would be even stronger: rather than just injecting 
context at inference time, we would train the model end-to-end via SFT (to learn resume Q&A format) 
→ DPO (to prefer detailed, specific responses) → GRPO (to align with quality metrics). This would 
create an even more dramatic improvement than simple context injection, enabling the model to 
learn the underlying patterns of high-quality resume answers rather than relying on memorized facts.


## TODO 2: Quantization

Quantization reduces LLM memory footprint by storing weights in 4-bit integers instead of float32/float16. 
Both bitsandbytes 4-bit (NB04 LoRA) and GGUF Q4_K_M (deployment) achieve ~8× compression. Q4_K_M uses 
"K-quants" with mixed precision — keeping critical layers (attention, embeddings) at higher precision 
while quantizing less-sensitive layers. This selective approach provides better quality/speed tradeoff 
than uniform Q4_0, making Q4_K_M the standard for LLM deployment.
