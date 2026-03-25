# Claude Code — Prompt Library

Reusable, copy-paste prompts for common workflows on this project.
Each prompt is self-contained: paste it as your first message in a new session.

---

## 1. Session Initialization (New Conversation)

Use this every time you start a new Claude Code session on this project.
It gives Claude everything it needs to understand the current state without asking questions.

```
New Claude Code session — Master's thesis project.

Orient yourself in this order:
1. Read README.md for the project overview
2. Read reasearch_notes/private/private - Implementation Roadmap.md for the current plan
3. Read code/EVALUATION_GUIDE.md for the experiment pipeline
4. Read code/RESULTS_TRACKER.md for the current job/results status
5. Run: git log --oneline -10
6. Run: git status

Then summarise in 5 bullet points:
- What the project is trying to do
- Where we are in the experiment pipeline
- What results are available or pending
- What was the last significant code change
- What the logical next step appears to be

Do not start any task yet — just orient yourself and report.
```

---

## 2. Check Cluster Job Status

Use when returning after leaving jobs queued on Alan.

```
I'm coming back after leaving SLURM jobs running on the Alan cluster.
Help me assess the current state.

1. Read code/RESULTS_TRACKER.md
2. Read the results directory structure: list code/results/ recursively (2 levels deep)
3. For each result JSON found, read and report: model, dataset, F1, accuracy, errors, inference speed

Then tell me:
- Which (model × dataset) pairs have completed results
- Which are missing (expected from EVALUATION_GUIDE.md)
- Whether the results look plausible (no obvious bugs, reasonable F1 range)
- What I should run next

Do not rerun anything yet.
```

---

## 3. Analyse Experiment Results

Use after results are available to get a structured interpretation.

```
Analyse the baseline evaluation results in code/results/.

For each result file found:
1. Load the JSON and extract: model, dataset, accuracy, precision, recall, F1, TPR, FPR, avg_inference_ms, gpu_memory_mb, energy_kwh, errors
2. Build a model × dataset F1 matrix
3. Identify the top 3 models by average F1 across all datasets
4. Identify the top 3 models by inference speed (lowest avg_inference_ms)
5. Flag any anomalies: F1 < 0.3, errors > 5%, or implausibly fast/slow inference

Then give me a short interpretive paragraph suitable for a thesis discussion section:
- Which model family performs best and why (speculatively, based on architecture)
- Where there are clear trade-offs between accuracy and deployability
- Which models should be considered for fine-tuning in Phase 2

Focus on deployability (VRAM, speed, energy) as an explicit evaluation axis alongside accuracy.
This is a key argument of the thesis: a model that requires 24GB VRAM is not deployable on
a small NGO platform like Shareish, regardless of its accuracy.
```

---

## 4. Analyse HateCheck Results

Use after run_hatecheck_analysis.py has produced output.

```
Analyse the HateCheck functionality results in code/results/hatecheck_analysis/.

For each model's result file:
1. Extract the per-functionality breakdown (hateful functionalities → TPR, non-hateful → TNR)
2. Identify the 3 worst-detected hateful functionality types per model (lowest TPR)
3. Identify the 3 most over-flagged non-hateful types per model (lowest TNR)
4. Compare EN vs FR results for multilingual models

Then build two interpretive summaries:
- STRENGTHS: what types of harmful content are reliably caught across all models
- BLIND SPOTS: what types consistently evade detection (worst average TPR)

Format the blind spots summary as thesis-ready prose.
These blind spots are the primary justification for why off-the-shelf models are insufficient
and why fine-tuning or a two-tier system is needed.
```

---

## 5. Debug a Failing SLURM Job

Use when a job produced an error or unexpected output.

```
A SLURM job failed or produced unexpected results. Help me diagnose it.

Job details:
- Script: [paste sbatch filename]
- Job ID: [paste job ID]
- Partition: [a5000 / all / 1080ti]

1. Read the script file
2. Read the .err log: code/logs/[relevant log file]
3. Read the .out log: code/logs/[relevant log file]

Then:
- Identify the root cause of the failure (be specific: line number, exception type)
- Check if it is a VRAM issue (model too large for GPU assigned)
- Check if it is a missing file/directory issue
- Check if it is a Python environment / import issue
- Propose the minimal fix — do not refactor, only fix the bug

If it is a VRAM issue: confirm the VRAM guard in the script would prevent it next time,
or explain why it didn't trigger.
```

---

## 6. Prepare a New Cluster Submission

Use before submitting a new sbatch job to do a pre-flight check.

```
I'm about to submit this SLURM job. Do a pre-flight check before I submit.

Script to review: [paste sbatch filename or content]

Check:
1. Does the log output directory exist? (SLURM can't create it)
   If not, print the mkdir command I need to run first.
2. Are the Python script paths correct relative to $HOME?
3. Does the script handle checkpointing — will it safely resume if cancelled?
4. Is the partition appropriate for the models being tested?
   (a5000 = 24GB, 1080ti = 10.9GB — flag any model that won't fit on 1080ti)
5. Is the time limit realistic? (check EVALUATION_GUIDE.md for time estimates)
6. Is N_RUNS or any loop variable set to a sensible value?

Output: a checklist of PASS / WARN / FAIL for each point,
plus a one-line "safe to submit" or "fix these issues first" verdict.
```

---

## 7. Write or Improve a Thesis Section

Use when transitioning from code work to writing.

```
I need to write (or improve) a thesis section.

Section: [e.g. "Results — Baseline Evaluation", "Discussion — Deployability Trade-offs"]
Target length: [e.g. ~500 words / ~2 pages]

Context for this section:
- What comes before: [brief description]
- What comes after: [brief description]
- Key data or results to include: [paste relevant metrics or findings]
- Papers to cite (already in my literature review): [list titles or just say "none yet"]

Task:
1. Propose a paragraph-level outline for this section first
2. Wait for my approval before drafting prose
3. When drafting, use formal academic English, active voice, past tense for experiments
4. Do not fabricate citations or data — if I haven't provided a number, write [INSERT X]
5. Flag any claim that needs a citation with [CITE]

Important thesis argument to weave in where relevant:
Deployability — VRAM requirement, inference speed, and energy cost — must be an explicit
evaluation axis alongside accuracy. A model that cannot run on a small NGO's infrastructure
is not a viable solution regardless of benchmark performance.
```

---

## 8. Design the Next Experiment

Use when deciding what to test or implement next.

```
Help me decide what experiment to run next.

Current state:
- Completed: [list what's done, e.g. "full baseline on 8 datasets, HateCheck EN/FR analysis"]
- Available GPU time: [e.g. "a5000 partition, roughly 2 days"]
- Thesis deadline: June 2026

Read:
1. reasearch_notes/private/private - Implementation Roadmap.md
2. code/EVALUATION_GUIDE.md
3. Any result summaries in code/results/

Then recommend the single highest-value next experiment, considering:
- What gaps remain in the baseline evaluation
- Whether statistical reliability runs (multi-run) are needed before moving to fine-tuning
- Whether HateCheck analysis has surfaced specific weaknesses worth targeted testing
- GPU budget vs. expected information gain

Give me: a one-paragraph justification, the exact command to run, and the sbatch file to use.
```

---

## Tips for Better Prompts

- **Be specific about files**: name the exact file/path rather than "the results"
- **State the desired output format**: "as a table", "as thesis-ready prose", "as a checklist"
- **Separate research from writing**: Claude Code works best when given one clear task at a time
- **Provide the data inline** when possible: paste a JSON snippet rather than asking Claude to find it
- **Always mention the deployability argument** in any results or discussion task — it's the thesis's core claim
