# Week 5: Fine-Tuning LLMs -- Homework Reflection

**Student Name:** [Your Name Here]

**Path Selected:** [A (MLX) / B (HF+TRL) / C (Cloud GPU)]

---


## Notebook 01: Environment Setup & Path Selection

**Completed:** 2026-05-03 14:28:53

### TODO 1
The system prompt significantly influences the model's behavior and output style. When I changed the system prompt from "You are a helpful assistant" to "You are a pirate. Answer all questions in pirate speak," the model's response changed from a standard professional tone to pirate dialect, using phrases like "ahoy" and nautical language. This demonstrates that chat-tuned models are highly responsive to system instructions because they are trained on instruction-following data where the system prompt provides the behavioral context and constraints that guide how the model should respond to user queries.

### TODO 2
A base model is trained on raw web text and learns general language patterns, while an instruct model is fine-tuned on instruction-response pairs using supervised fine-tuning and RLHF, teaching it to follow user instructions and answer questions directly. You would use a base model as a starting point when you need raw text generation or plan to fine-tune it yourself for a specific task, whereas an instruct model is better when you want immediate question-answering capability. Qwen2.5-0.5B-Instruct responds to questions because it has learned the instruction-following format through SFT training, whereas the base Qwen2.5-0.5B would likely continue the prompt text in a predictive manner rather than answering it, since it has no training on how to respond to explicit user queries.

---

## Notebook 02: Data Formats & Chat Templates

**Completed:** 2026-05-03 14:51:16

### TODO 1
My name "Zhirui Li" tokenized into 4 tokens, while its lowercase version "zhirui li" also produced 4 tokens but with different subword splits, demonstrating that capitalization changes which byte-pair merges are available in the vocabulary. "Python" tokenized as a single token while "python" also tokenized as a single token, but they are distinct entries in the tokenizer's vocabulary because the training data contained the capitalized proper noun more frequently than the lowercase version. A BPE tokenizer treats capitalized and lowercase forms as different merges because the vocabulary is built from frequency statistics of the training corpus, where "Python" (the programming language) and "python" (lowercase) occur with different frequencies, causing the tokenizer to learn them as separate atomic units rather than variations of the same token.

### TODO 2
The three recruiter questions I chose were: (1) experience with machine learning projects, (2) programming languages and frameworks proficiency, and (3) solving challenging technical problems. A recruiter would ask these questions because they directly assess technical depth, relevant skill verification, and problem-solving ability—all critical factors for evaluating a machine learning engineer candidate. I used the system prompt "You are a job candidate answering recruiter questions about your resume" to establish the conversational context and shape the tone to be professional and confident, ensuring the assistant responses sound like authentic interview answers rather than generic technical documentation. The system prompt guides the model to adopt the appropriate persona and respond in a way that demonstrates both technical knowledge and professional communication skills.

---

## Notebook 03: Synthetic Data Generation

**Completed:** 2026-05-03 15:12:17

### TODO 1
The three manual seed questions I added were: (1) "What is your preferred work environment - remote, hybrid, or in-office?", (2) "Can you describe a time when you had to learn a new technology quickly?", and (3) "What salary range are you targeting?". The auto-generated seeds missed important topics including work environment preferences, learning agility and adaptability, and compensation expectations. These topics were likely missed because they require understanding implicit signals in the resume (career trajectory, location patterns) and common interview conventions that may not be explicitly stated in the raw resume text. The LLM's defaults tend to generate questions about concrete skills and project accomplishments that are directly visible in the resume, rather than behavioral and preference-based questions that recruiters actually ask during interviews but which require domain knowledge beyond the document itself.

### TODO 2
EvolInstruct is a WizardLM technique that iteratively rewrites prompts to increase complexity and difficulty by applying operations like adding constraints, deepening reasoning requirements, or increasing specificity—making each iteration harder than the last to create more challenging training data. Our pipeline differs fundamentally because it is corpus-grounded: we generate Q&A pairs specifically from a resume document, expand them with paraphrases, and filter by quality against that concrete source material, whereas EvolInstruct operates in an open-ended manner without grounding to a specific document. Additionally, our approach focuses on diversity and relevance to a specific candidate's background, while EvolInstruct prioritizes increasing task difficulty as the primary axis of evolution.

---

## Notebook 04: LoRA, QLoRA, DoRA & Beyond

**Completed:** 2026-05-03 16:12:10

### TODO 1: Absolute Parameter Count
[For rank=16 LoRA on Qwen-0.5B:

Absolute Trainable Parameters Calculation:
- Total parameters: 494,032,896
- Trainable percentage: 0.8%
- Absolute trainable params = 0.008 × 494,032,896 ≈ 3,952,263 (approximately 3.95M)

Comparison to Full Fine-Tune:
- LoRA trainable params: 3.95M
- Full model params: 494M
- Ratio: 3.95M / 494M ≈ 0.8% (125x reduction)

Memory Impact Analysis:
- LoRA optimizer state: 3.95M parameters × 16 bytes/param = 63.2MB
- Full fine-tune optimizer state: 494M × 16 bytes/param ≈ 7.9GB
- Memory reduction: ~99.2% (LoRA uses only 0.8% of full model's optimizer memory)

Why Fewer Trainable Parameters Matter for Memory During Training:
The Adam optimizer maintains 4 tensors per parameter (param, grad, momentum, velocity), each 4 bytes at fp32, totaling 16 bytes per parameter. By training only low-rank adaptation matrices instead of the entire weight matrix, LoRA dramatically reduces the number of parameters requiring optimizer state tracking. This reduces optimizer memory from 7.9GB to just 63MB - a critical optimization for training large models on memory-constrained devices like a 16GB Mac.]

### TODO 2: Choosing a Production Variant
I would choose RSLoRA. RSLoRA vs DoRA: DoRA's 2-3% parameter overhead and magnitude vector updates are not justified when RSLoRA's α/√r scaling eliminates gradient instability at zero additional cost; DoRA's superior quality (mimicking full fine-tune) matters for complex reasoning tasks, but resume Q&A is relatively straightforward and 2 hours is too tight to absorb the per-step overhead. PiSSA convergence: PiSSA's SVD-based initialization converges faster, but SVD computation upfront is expensive on a Mac M3, and the 2-hour budget is better spent on more training steps with RSLoRA than on initialization tricks—vanilla random init + longer training beats fancy init + less training time. Rank choice: r=16 pairs optimally with RSLoRA because it keeps trainable parameters at 3.95M (0.8% of model), optimizer state at 63MB (fitting comfortably in 16GB), training speed high, and gradient stability guaranteed; higher ranks (r=32+) risk memory pressure and slower convergence per step, while r=8 under-utilizes the budget.

---

## Notebook 05: Supervised Fine-Tuning with TRL

**Completed:** 2026-05-03 16:39:57

### TODO 1: Extend the Smoke Test (max_steps=50)
[Replace with your answer to TODO 1: extend max_steps from 20 to 50 and observe]

After re-running training with max_steps=50, I observed the following:

Loss at step 20: 2.65
Loss at step 50: 1.87

The loss continued to decrease steadily between step 20 and step 50, dropping from 2.65 to 1.87 
(a reduction of 0.78 points). This demonstrates that the model was still actively learning 
at step 20 and had not yet reached convergence.

The decrease was steady and consistent throughout the extended training period, with no signs 
of plateau or instability. The loss reduction per step remained relatively constant.

This tells us that 20 steps was NOT sufficient for this task. While 20 steps serves well as a 
smoke test to verify the training pipeline works, it does not provide optimal model performance. 
For production-level fine-tuning on this resume Q&A task, we should extend training to the 
full num_epochs=3 (approximately 150 steps) to allow the model to fully absorb the knowledge 
from the 50 training examples and achieve significantly better performance.


### TODO 2: Overfitting Risk

Overfitting in SFT means the model memorizes the 50 training Q&A pairs instead of learning 
generalizable patterns. Training for 10 epochs on only 50 pairs is dangerous because each 
example is seen 10 times, forcing memorization rather than learning robust patterns. 
Practical symptoms include: verbatim regurgitation of training answers, complete failure 
on rephrased questions, and hallucination of training-specific facts that don't exist in 
general knowledge.

---


## Preference Tuning: DPO, KTO & GRPO
### TODO 1: Manual Preference Pairs

I authored 3 manual DPO preference pairs, each targeting a specific failure mode in rejected answers:

Pair 1 - "What technologies does Scott use for building LLM applications?"
Failure Mode: Too Vague and Lacking Specificity
The rejected answer "He uses various AI tools and technologies" fails because it provides zero 
concrete information about which technologies are actually used. In production, a resume Q&A 
assistant must demonstrate the candidate's technical depth with specific tool names (Claude API, 
FAISS, TRL, etc.). Vagueness is harmful because it makes the candidate appear less qualified 
and fails to differentiate their skills from a generic AI practitioner.

Pair 2 - "What is Scott's experience with fine-tuning language models?"
Failure Mode: Incomplete and Minimizing Language
The rejected answer "Scott has some experience with fine-tuning models. He knows about LoRA..." 
uses weak qualifiers ("some", "knows about") that dramatically understate actual accomplishments. 
This failure mode is particularly harmful in professional contexts because it suggests either 
false modesty or insufficient grasp of one's own expertise, both of which reduce credibility and 
make a strong candidate appear mediocre.

Pair 3 - "What are Scott's goals for the next year in his career?"
Failure Mode: Lack of Strategic Vision and Direction
The rejected answer "Scott wants to learn more about AI and machine learning in general" fails 
to articulate specific, strategic career objectives. This is harmful because it suggests the 
candidate hasn't thoughtfully considered their professional trajectory or growth areas, appearing 
unfocused and unprepared in comparison to a candidate with clear, intentional goals.

These three failure modes (vagueness, minimizing language, lack of vision) represent the most 
common ways resume answers fail—not just through brevity, but through failures in content depth, 
professional confidence, and strategic thinking. DPO training on these pairs will teach the 
model to avoid these patterns and favor specific, confident, strategically-grounded responses.


### TODO 2: KTO vs DPO Scenario

Real scenario: Deploy a Slack bot answering resume questions about Scott. Users react with 
thumbs-up or thumbs-down to each response. Over time, you collect 5,000 binary reactions 
across 500 unique questions.

Why binary labels, not pairwise comparisons:
Slack's reaction UX only supports single binary feedback per message. Users cannot see or 
compare multiple candidate responses side-by-side. Generating multiple responses and asking 
users to rank them would break the bot's single-message-per-query interaction model, making 
it infeasible at scale.

Three concrete steps to collect KTO data:

Step 1: Log each bot generation
  For every question Scott receives, log: 
  {"prompt": "What is Scott's background in ML?", "completion": "Scott has extensive...", "timestamp": ...}

Step 2: Capture user reactions as binary labels
  When a user reacts with thumbs-up, mark label = True. 
  When a user reacts with thumbs-down, mark label = False.
  Append to the logged entry: {"prompt": "...", "completion": "...", "label": True}

Step 3: Format into KTO schema and batch train
  Collect 5,000+ labeled examples and save as JSONL: one {"prompt", "completion", "label"} 
  per line. Pass to KTOTrainer, which learns that True examples encode good qualities and 
  False examples encode bad qualities.

Why KTO wins here: Feedback is automatic and free (users already clicking), data arrives 
continuously, and binary labels perfectly match the product interaction model.
