# Multi-Agent LLM Research Evaluation Prompts

This README contains 4 distinct prompts/tasks for students or LLM systems. Each prompt is enclosed in a copy-pasteable code block so that you can easily copy and paste it into an LLM interface or use it in command-line scripts.

---

## Prompt 1: ResearchAgentBench Benchmark Design

```text
You are part of a university lab designing a benchmark to evaluate AI systems that assist students with research tasks.

Your task is to design a benchmark called ResearchAgentBench.

The benchmark should evaluate whether an AI system can actually help a graduate student complete research work, not just produce fluent text.

You must produce a full benchmark design document that includes:
1. Benchmark goal
2. Core assumptions
3. Task categories
4. At least 12 example tasks across different categories
5. A scoring rubric
6. Baselines to compare against
7. Likely failure modes and gaming risks
8. Human evaluation protocol
9. Limitations of the benchmark
10. Recommendations for version 2 of the benchmark

Constraints:
- The benchmark must be realistic for a small academic lab
- It must not depend on expensive annotation
- It must distinguish between usefulness, correctness, and research judgment
- It must include at least one adversarial or stress-test component
- It must explicitly discuss how systems might game the evaluation

Output format:
Write this as a mini design document suitable for discussion in a research lab meeting.
```

---

## Prompt 2: Research Briefing on Multi-Agent LLMs

```text
You are helping a PhD student prepare a research briefing on the topic:

"Do multi-agent LLM systems actually outperform single-agent systems on complex tasks?"

Your task is not just to summarize the topic. You must produce a structured research briefing that:
1. Defines the main claim precisely
2. Breaks the literature into major positions or schools of thought
3. Identifies arguments supporting the claim
4. Identifies arguments challenging the claim
5. Explains where empirical evidence is weak, incomplete, or confounded
6. Distinguishes between true multi-agent gains and gains caused by other factors such as more tokens, more prompt engineering, or repeated self-reflection
7. Proposes 3 concrete experiments that could better resolve the debate
8. Ends with a balanced final judgment

Constraints:
- Do not write a generic overview
- Explicitly discuss what kinds of evidence would count as convincing
- The final judgment must include uncertainty and unresolved issues
- Organize the answer so it could be used as speaking notes for a research group meeting

Output format:
- Core question
- Main positions
- Evidence for
- Evidence against
- Methodological concerns
- Proposed experiments
- Final judgment
```

---

## Prompt 3: Experimental Design - Single-Call vs. Multi-Agent

```text
You are designing a research experiment to compare a single-call LLM system with a multi-agent LLM system on complex research tasks.

Your deliverable must be a complete experimental plan.

You must include:
1. The research question
2. Hypotheses
3. Task design
4. Datasets or source materials
5. Fair comparison setup
6. Metrics
7. Human evaluation criteria
8. Statistical or methodological considerations
9. Expected results and alternative interpretations
10. A red-team section explaining how the experiment could be misleading, unfair, or easy to game
11. A revised experiment design after taking the red-team critique seriously

Constraints:
- You must explicitly address token budget fairness
- You must explain how to avoid giving the multi-agent system an unfair advantage
- You must discuss how to separate gains from decomposition vs gains from more inference time
- The red-team critique must be concrete, not superficial

Output format:
Write the answer as a structured internal lab proposal.
```

---

## Prompt 4: Survey Paper Blueprint

```text
You are helping prepare a survey paper titled:

"AI Agents for Research Assistance: Capabilities, Evaluation, and Open Problems"

Your task is to create a detailed survey blueprint that a graduate student could actually use to write the paper.

You must produce:
1. A proposed paper title
2. A draft abstract
3. A section-by-section outline with 6 to 8 main sections
4. For each section:
   - purpose of the section
   - key themes
   - questions the section should answer
   - likely pitfalls
   - open problems
5. A final section identifying the most important evaluation gaps in current research
6. A section explaining what a strong future benchmark should measure

Constraints:
- The sections should not overlap too much
- The outline should feel like a real survey, not a blog post
- Open problems must be specific
- The final product should be useful as a writing plan, not just a topic list
```
