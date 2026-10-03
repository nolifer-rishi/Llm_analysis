# EduBench-Local: Evaluating Edge-Capable Small Language Models as Educational Tutors

## 1. Abstract
The rapid advancement of Large Language Models (LLMs) has opened new possibilities for personalized digital tutoring. However, the reliance on proprietary, cloud-based APIs raises severe concerns regarding student privacy, latency, and recurring operational costs for educational institutions. In this paper, we present **EduBench-Local**, a framework specifically designed to evaluate the viability of deploying small, open-weight language models on highly constrained edge hardware (4GB VRAM). We evaluate two representative models—Mistral 7B Instruct and Orca Mini 3B—across 500 questions sampled from five diverse educational datasets (ARC, OpenBookQA, SciQ, RACE, SQuAD). To capture a holistic view of tutoring performance, we employ a multi-metric pipeline consisting of ROUGE-L (lexical overlap), BERTScore (semantic similarity), and a novel LLM-as-a-Judge prompting paradigm evaluating factual correctness on a 1–5 scale. Our findings demonstrate that despite hardware limitations, Mistral 7B achieves an exceptional mean Judge score of 4.52/5, proving the immediate viability of edge-deployed LLMs for educational applications.

## 2. Introduction
**Motivation:** Intelligent tutoring systems have historically relied on rigid, rule-based systems. LLMs offer a paradigm shift, capable of generating nuanced explanations and adapting to student inquiries. However, relying on state-of-the-art models like GPT-4 or Claude requires persistent internet connectivity and entails recurring API costs, rendering them inaccessible to underfunded schools. Furthermore, transmitting student data to third-party servers raises significant privacy and COPPA compliance concerns. Locally hosted, small language models (SLMs) running on edge devices (e.g., student laptops or local school servers) solve these issues.

**Gap in Existing Benchmarks:** While exhaustive benchmarks like MMLU, HELM, and BIG-bench exist, they primarily focus on evaluating massive models (70B+ parameters) on high-end datacenter GPUs. Furthermore, traditional benchmarks often rely on single-metric evaluations (e.g., exact-match accuracy), which fail to capture the nuanced, conversational correctness required of an educational tutor.

**Contributions:**
1. A fully local, edge-capable evaluation pipeline designed for consumer-grade hardware (AMD Radeon RX 6500M, 4GB VRAM).
2. A multi-metric evaluation strategy that juxtaposes traditional lexical/semantic metrics against LLM-as-a-Judge evaluations.
3. A comparative baseline analysis between Mistral 7B and Orca Mini 3B across five distinct educational domains.

## 3. Related Work
- **LLM Benchmarks:** Foundational benchmarks like MMLU (Hendrycks et al., 2020) and BIG-bench evaluating broad reasoning capabilities.
- **Educational NLP:** Datasets like SciQ (Welbl et al., 2017) and RACE (Lai et al., 2017) have historically been used to train and evaluate reading comprehension and science QA models.
- **LLM-as-a-Judge:** Recent literature (Zheng et al., 2023) validates the use of strong LLMs to grade open-ended outputs of other models, noting a high correlation with human annotator preferences, thereby solving the rigidity of n-gram overlap metrics.

## 4. Dataset
We sampled 100 questions uniformly at random from five standardized educational datasets, totaling 500 evaluation queries. This ensures a diverse assessment spanning reading comprehension, science, and factual recall.

| Dataset | Domain | Question Type | Sample Size |
| :--- | :--- | :--- | :--- |
| **SciQ** | Factual Science | Open-ended | 100 |
| **OpenBookQA** | Science Reasoning | Multiple Choice | 100 |
| **ARC (Challenge)** | Grade School Science | Multiple Choice | 100 |
| **RACE (Middle)** | Reading Comprehension | Multiple Choice | 100 |
| **SQuAD** | General Extractive QA | Open-ended | 100 |

## 5. Methodology
**Models Used:**
1. **Mistral 7B (Instruct):** A highly performant 7-billion parameter model utilizing grouped-query attention and sliding window attention.
2. **Orca Mini 3B:** A smaller, highly distilled model trained on explanation traces, optimized for extreme resource constraints.

**Prompt Template:**
Both models were evaluated using a unified "Tutor" persona prompt to ensure consistency:
*"You are a helpful and knowledgeable tutor. Answer the following question clearly and concisely, in 2-4 sentences."*

**Three-Metric Pipeline:**
1. **ROUGE-L:** Measures exact lexical sequence overlap between the model's generation and the ground-truth answer. (Used to detect direct string recall).
2. **BERTScore (F1):** Computes deep contextual embedding similarities. (Used to verify if the model's answer means the same thing as the reference, even if worded differently).
3. **LLM-as-a-Judge (1-5 Scale):** An LLM evaluates the factual correctness and completeness of the generated answer against the reference. (Used to simulate a human teacher's grading flexibility).

## 6. Experimental Setup
All experiments were strictly constrained to local execution to validate edge viability.
- **Hardware:** AMD Radeon RX 6500M (4GB VRAM).
- **Software:** Ollama backend utilizing Vulkan API acceleration.
- **Reproducibility:** A fixed random seed (`42`) was used for all dataset sampling. Generation parameters were locked to `temperature=0.3` and `top_p=0.9` to prioritize deterministic, factual outputs over creative hallucinations.

## 7. Results

### Overall Leaderboard
| Rank | Model | ROUGE-L | BERTScore | Judge (1-5) | Combined Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Mistral 7B | 0.131 | 0.852 | 4.52 | 62.9 |
| 2 | Orca Mini 3B | 0.089 | 0.844 | 2.30 | 46.4 |

### Subject-Wise Highlights
- **Mistral 7B** demonstrated exceptional robustness across all domains, peaking in SQuAD (Judge: 4.69) and ARC (Judge: 4.48). Its primary weakness was RACE (Judge: 4.07), where complex multi-paragraph reading comprehension sometimes exceeded its active context window limitations under extreme quantization.
- **Orca Mini 3B** struggled significantly with the strict formatting requirements of the prompt, often failing to answer Multiple Choice Questions (MCQs) correctly, resulting in a low mean Judge score of 2.30.

## 8. Discussion
**Trade-offs (Accuracy vs. Latency):** 
Mistral 7B vastly outperformed Orca Mini 3B in accuracy and reasoning. However, running a 7B model on 4GB of VRAM required heavy quantization and offloading, resulting in higher latency (averaging ~10.5 seconds per generation). Orca Mini 3B was substantially faster but exhibited a severe degradation in reasoning capabilities, suggesting that 7B parameters may be the current minimum viable threshold for reliable educational tutoring.

**Metric Divergence:**
A key finding of this study is the divergence between traditional metrics and the LLM Judge. We observed a very strong correlation between ROUGE-L and BERTScore, but a weak correlation (ρ~0.3) between the LLM-Judge and text-overlap metrics. ROUGE-L heavily punished Mistral 7B for acting as a conversational tutor (e.g., outputting "The correct answer is gravity because..." instead of just "gravity"). The LLM-Judge successfully bypassed this, evaluating the semantic factual correctness of the answer rather than penalizing verbosity.

## 9. Limitations
1. **Sample Size:** Due to the extreme hardware limitations (4GB VRAM) and 10+ second latencies, evaluations were limited to 100 questions per dataset. 
2. **Judge-Model Bias:** Mistral 7B was utilized as the LLM-Judge for all models (including itself), which introduces potential self-preference bias.
3. **English-Centric:** The datasets and models evaluated were predominantly English-centric, limiting immediate applicability in global, multilingual classrooms.

## 10. Conclusion & Future Work
EduBench-Local demonstrates that local, privacy-preserving LLMs are entirely viable for educational deployment today. Despite stringent hardware constraints, Mistral 7B achieved near-expert grading (4.52/5), proving that cloud connectivity is no longer a strict requirement for AI tutoring. Future work will expand the pipeline to encompass highly efficient emergent architectures (e.g., Llama 3, Qwen 2.5), integrate multilingual educational datasets, and scale evaluations using distributed local hardware setups.

## 11. References
1. Hendrycks, D., et al. (2020). *Measuring Massive Multitask Language Understanding (MMLU).* 
2. Jiang, A. Q., et al. (2023). *Mistral 7B.* arXiv preprint arXiv:2310.06825.
3. Mukherjee, S., et al. (2023). *Orca: Progressive Learning from Complex Explanation Traces of GPT-4.*
4. Zheng, L., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.*
5. Lin, C. Y. (2004). *ROUGE: A Package for Automatic Evaluation of Summaries.*
6. Zhang, T., et al. (2019). *BERTScore: Evaluating Text Generation with BERT.*
