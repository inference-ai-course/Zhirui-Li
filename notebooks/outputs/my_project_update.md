# Week 5 Project Update ¡ª LLM Fine-tuning & Multi-Skill Routing

## What I built this week

This week I completed a comprehensive LLM fine-tuning and deployment pipeline:
- Implemented Supervised Fine-Tuning (SFT) with LoRA adapters on Qwen2.5-0.5B-Instruct, achieving 99.2% 
  memory reduction compared to full fine-tuning
- Built preference data generation system for DPO (Direct Preference Optimization) using Claude API, 
  creating 5 high-quality chosen/rejected answer pairs with ~10x length differential
- Created multi-skill routing system (SkillResolver) that intelligently dispatches queries to specialized 
  models: fine-tuned Resume Skill (19 keywords) and general Ollama Q&A Skill
- Set up LLM-as-Judge evaluation framework using Claude Haiku as evaluator on 5-question test set
- Deployed fine-tuned model via GGUF quantization (Q4_K_M format) for efficient inference on resource-constrained devices

## How Week 5 connects to Week 4 (RAG + fine-tuning)

Week 4 introduced Retrieval-Augmented Generation (RAG) as a way to inject context into models without 
fine-tuning. Week 5 builds on this by showing that fine-tuning is the more powerful alternative when 
you have quality training data:
- RAG (Week 4): Retrieve context at inference time, works fast but limited by retrieval quality
- Fine-tuning (Week 5): Bake knowledge into model weights during training, enables better generalization

The multi-skill router combines both approaches: for resume-specific questions, we use the fine-tuned 
model (Week 5 knowledge encoded in weights); for general questions, we use the base model with RAG-like 
context injection (Week 4 pattern). This hybrid approach provides both specialization and flexibility.

## What surprised me most about fine-tuning

The biggest surprise was how sensitive model quality is to data quality and not just data quantity. 
Creating just 5 high-quality DPO preference pairs (manually crafted chosen/rejected examples) had more 
impact on the evaluation results than generating 50 lower-quality examples through prompt engineering. 
Additionally, the difference between rejected answers was meaningful ¡ª not just making them shorter, but 
failing in specific ways (too vague, minimizing language, lack of vision) made the preference signal 
much stronger. Fine-tuning isn't about feeding the model more data; it's about teaching it the right 
*principles* through carefully designed examples.

## What I would improve with more compute/time

With more compute and time, I would:
1. Scale up training data: Expand from 5 to 500+ preference pairs for more robust learning
2. Implement full RLHF pipeline: DPO ¡ú GRPO (Group Relative Policy Optimization) with custom reward 
   functions for resume Q&A specific metrics (mentions candidate name, includes concrete technologies, 
   maintains professional tone)
3. Multi-stage fine-tuning: SFT ¡ú DPO ¡ú GRPO ¡ú Knowledge distillation to smaller models for faster inference
4. Expand skill resolver: Add 5+ specialized skills (code explanation, technical concepts, career advice, 
   system design, etc.) instead of just 2
5. A/B testing framework: Systematically evaluate which fine-tuning approach (LoRA rank, preference method, 
   data size) works best for each skill
6. Production deployment: Deploy fine-tuned models on multiple devices (Mac M3, GPU cluster, edge devices) 
   with latency/throughput monitoring and continuous retraining on user feedback