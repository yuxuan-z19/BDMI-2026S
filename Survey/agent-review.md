# Agentic Review

## Gemini 3.5 Thinking

**Familiarity**: Expert (I have extensive knowledge of pedagogical frameworks for project-based learning and the integration of generative AI into computing education).

### 1. Contributions

This paper makes three primary, valuable contributions to the ERT track:

- **Problem Characterization:** It clearly articulates the pedagogical shift in GenAI-era education, identifying the new instructional bottleneck as "high-level design reasoning" rather than "code production". 
- **Pedagogical Framework:** It introduces Speculative Scaffolding, an adaptive, pattern-driven framework designed to manage project-based LLM-agent instruction. 
- **Empirical Deployment:** It provides a comparative classroom study across two course iterations, demonstrating the framework's effectiveness in shifting instructional workload and fostering systematic student design reasoning. 

### 2. Strengths

- **Practicality and Adoption:** The paper is highly actionable. It defines specific instruments (e.g., the Scoping Blueprint, TA Synthesis Guide, and RTAM Failure Attribution Protocol) that are explicitly designed to be adopted by other instructors, which is a core requirement of the ERT track .  
- **Strong Pedagogical Grounding:** The transition from curriculum-centered models (too rigid) to student-centered models (unscalable) is well-motivated by current literature, making the "middle ground" of Speculative Scaffolding compelling .  
- **Clear Reflection:** The authors provide a "rich reflection" on both what worked (e.g., earlier design convergence, temporal redistribution of workload) and the limitations of their approach (e.g., small cohort size, requirement for instructor expertise) .  3. 

### 3. Weaknesses

- **Generalizability Limitations:** As the authors acknowledge, the study is based on a small, single-course deployment ($N=18$ vs $N=17$). While acceptable for an ERT paper, readers must be cautious about the scalability of the framework to significantly larger or more diverse cohorts.  
- **Implicit Assumptions:** The framework relies heavily on TA/instructor expertise in agent-based systems. The paper could be strengthened by detailing how this scaffolding might be implemented in environments where teaching staff are less familiar with these specific, emerging architectures.  

### 4. Novelty

The novelty is strong. While project-based learning (PBL) is well-studied, this work is among the first to systematically address the stochastic nature of LLM-agent development in a classroom setting. The RTAM (Replicate-Transfer-Analyze-Modify) cycle  effectively operationalizes "debugging" for non-deterministic AI systems, distinguishing it from traditional software engineering debugging.  

### 5. Final Verdict

**Recommendation:** Accept

**Justification:** This paper is an excellent fit for the SIGCSE ERT track. It presents a well-documented intervention that addresses a pressing, contemporary challenge in computing education. The framework is not merely theoretical; it provides concrete, reproducible tools that practitioners can adopt to improve their own courses. The evaluation is honest about its limitations and provides a clear reflection on the outcomes, fully meeting the criteria for an Experience Report and Tool contribution.

## Stanford Agentic Reviewer

### 1. Summary

This paper proposes Speculative Scaffolding, a pedagogical framework for LLM-agent–focused project courses that shifts mentorship from ad hoc, implementation help to structured support for architectural design reasoning. The approach aggregates heterogeneous student projects into a small number of shared “technical routes,” organizes instruction around adaptive architectural anchors, and operationalizes iteration through a Replicate-Transfer-Analyze-Modify (RTAM) cycle supported by three instruments (scoping blueprint, TA synthesis guide, failure attribution protocol). A semester-long, mixed-methods deployment (N=18) is contrasted with a prior offering (N=17), and the authors report earlier architectural convergence, more systematic design reasoning, and a redistribution of TA workload toward earlier weeks.

### 2. Strengths

**Technical novelty and innovation**
- Presents a clear, actionable reframing of support in LLM-era courses from code-centric help to design reasoning under uncertainty.
- Introduces a concrete operationalization via the RTAM cycle and three well-scoped instruments that instructors could plausibly adopt.
- Pattern-based cohorting and just-in-time curriculum adaptation offer a pragmatic middle ground between prescriptive curricula and unconstrained projects.

**Experimental rigor and validation**
- Mixed-methods deployment includes surveys, interviews, observational logs, and a between-iteration comparison, enabling some triangulation.
- The workload shift analysis (earlier TA demand followed by stability) is consistent with the proposed mechanism (earlier structuring reduces later thrash).

**Clarity of presentation**
- The problem motivation is timely and well-articulated; figures and examples clarify the two-phase workflow and RTAM loop.
- Student trajectory vignettes (research-oriented vs engineering-oriented) and quotes concretize how scaffolding shaped practices.

**Significance of contributions**
- Addresses an urgent need as many programs integrate LLMs and agentic systems yet lack scalable, design-centered mentoring models.
- Offers portable artifacts (instruments, protocols) that could immediately benefit instructors experimenting with open-ended, AI-assisted projects.

### 3. Weaknesses

**Technical limitations or concerns**
- The evaluation is limited to a single institution with small cohorts and no randomization; cohort differences and instructor learning effects may confound results.
- Key constructs (e.g., “earlier convergence,” “systematic design reasoning,” “reduced instructional burden”) are not operationalized with robust, reproducible metrics.
- Several references include anomalous dates or incomplete details, suggesting bibliographic inconsistencies that should be corrected.

**Experimental gaps or methodological issues**
- No direct measures of learning gains or artifact quality; the study focuses on perceptions and process timing without assessing outcomes against rubrics.
- TA workload is measured in “active support days/week,” which is coarse; time-on-task or ticket counts would improve precision.
- Lack of statistical analysis, inter-rater reliability for any qualitative coding, or control for differing student backgrounds, tools, or semester contexts.
- Unclear details on survey instruments (items, validation), interview protocols, and data analysis methods.

**Clarity or presentation issues**
- Some figures are placeholders and several metadata artifacts (e.g., dates, venue text, “7 pages”) detract from polish and raise questions about completeness.
- The “AI-assisted synthesis workflow” for proposal aggregation is under-specified (what models, prompts, TA calibration, turnaround times).
Reported “0.0%” for some supports in Figure 6a seems implausible and merits explanation or correction.

**Missing related work or comparisons**
- Limited engagement with established literatures on studio-based learning, design thinking in CS education, cognitive apprenticeship, and orchestration of project-based courses at scale.
- While agent memory/governance papers are cited, connections to pedagogical frameworks that teach design reasoning under uncertainty could be deepened, and comparisons to similar scaffolding structures (e.g., milestone-driven PBL, crit-based studio pedagogy) are not made.

### 4. Detailed Comments

**Technical soundness evaluation**
- The RTAM cycle is a sensible adaptation of iterative enhancement/design–build–test to non-deterministic agents; the failure attribution checklist is a valuable addition that nudges students from prompt tinkering to systematized hypothesis testing.
- Architectural pattern aggregation and cohorting are technically plausible as a means to reduce feedback variance and TA load; however, the stability of these “routes” across semesters and institutions is not established.
- The claim that coding is no longer the bottleneck is well motivated and reflected in student perceptions; still, the framework should emphasize guardrails to ensure students retain core implementation literacy and do not over-depend on tools.

**Experimental evaluation assessment**
- The study leverages multiple data sources but needs stronger methodological transparency (survey items, response rates, coding schemas, inter-rater reliability, statistical tests).
- “Earlier convergence” could be operationalized as time-to-stable-architecture, count of architectural pivots, or time from baseline replication to first domain-specific feature; the current evidence is descriptive.
- TA workload would be more credible with per-TA time logs (hours), issue counts, and categorization; “days with support” is not granular and may mask intensity.
- No outcome metrics (e.g., external judge ratings of artifacts, design rationale quality rubrics, replication fidelity scores) are reported; adding these would strengthen claims that reasoning improved.
- Between-iteration comparisons should disclose cohort equivalence (prior experience, demographics), unchanged assessments, and tooling; otherwise, maturation/instructor effects could explain differences.

**Comparison with related work (using the summaries provided)**
- Recent agent memory/governance/skills works (e.g., MEMORYARENA, AMA-Bench, MemGym, SkillsBench, runtime governance) underscore that agentic systems are non-deterministic, skill- and memory-dependent, and evaluation-sensitive—supporting the paper’s core premise that students need structured design reasoning.
- The paper could leverage these findings pedagogically: e.g., framing failure attribution around memory vs retrieval vs reasoning, or building cohort exemplars that surface known failure modes from AMA-Bench or MemGym-like scenarios.
- To contextualize the educational contribution, the paper should also connect more fully to cognitive apprenticeship, studio pedagogy, and scalable mentoring/orchestration tools (e.g., structured crits, design journals, analytic dashboards) beyond the brief nods to scaffolding and apprenticeship.

**Discussion of broader impact and significance**
- The proposed artifacts (scoping blueprint, TA synthesis guide, failure attribution checklist) are promising for adoption; releasing them with example filled-in templates and TA training materials would accelerate community uptake.
- The framework appears adaptable to other open-ended, tool-rich domains (e.g., data science capstones, HCI prototyping) where uncertainty is structural, not incidental.
- Risks and constraints include dependence on TA expertise in LLM-agent systems, institutional resources (API costs, infrastructure), and equity (ensuring cohorts receive comparable quality of support despite differing “routes”).
- Positioning as an experience report or design case with richer artifacts and replication guidance may best serve SIGCSE’s practitioner audience while the authors build a larger-evidence base.

### 5. Questions for Authors

1. How did you operationalize “earlier convergence” and “more systematic design reasoning”? Could you report concrete measures (e.g., time-to-stable-architecture, number of architectural pivots, rubric-based ratings of design rationales) and any statistical comparisons?
2. What were the exact survey items and scales used (Figures 4–6), response rates, and any reliability/validation steps taken? Can you provide the instruments in an appendix or repository?
3. How was TA workload recorded (e.g., hours, tickets, meetings), and could you share more granular data (counts, durations, categories) rather than “support days/week”?
4. Were there notable differences between cohorts (experience, major mix, available tools, grading policies) that might influence the between-iteration comparison? How did you control for instructor learning effects?
5. Could you describe the “AI-assisted” proposal synthesis in more detail (models, prompts, TA calibration, time per batch, inter-rater agreement)? How reproducible and safe is this process?
6. Did you measure artifact quality or learning outcomes with external or blinded raters? If not, what plans do you have to incorporate outcome-based measures beyond perception and process timing?
7. How were students and TAs oriented or trained to use the RTAM cycle and the checklists? Do you have evidence that students adhered to the cycle (e.g., trace logs, commit histories)?
8. Several references have unusual dates or incomplete metadata. Could you review and correct the bibliography, and add missing education-focused related work on studio/design pedagogy and scalable PBL orchestration?

### 6. Overall Assessment

This paper tackles a timely and substantive challenge: as LLMs reduce implementation friction, open-ended courses must help students reason about architecture, failure modes, and trade-offs under non-determinism. Speculative Scaffolding is a coherent, well-motivated framework with practical instruments that many instructors will find useful, and the preliminary deployment provides encouraging signals (earlier structuring, shifted TA load, self-reported emphasis on diagnosis and attribution). However, as a research contribution, the empirical evidence is currently under-specified: constructs are not rigorously operationalized, measurements are coarse, and the between-iteration comparison lacks controls and statistical analysis. The paper would be strengthened by clearer methodology, validated instruments, outcome metrics (artifact quality and reasoning quality), richer workload analytics, and tighter engagement with education literature on design/studio pedagogy and scalable mentoring. In its current form, this is a valuable experience/design report with promising ideas and artifacts for the SIGCSE community; with stronger, well-instrumented evaluations, it could become an impactful research contribution.

## Seed 2.0 Pro

### Summary

This paper addresses a pressing pedagogical challenge in the generative AI era: as LLMs lower barriers to code implementation, the core bottleneck in project-based computing education has shifted from code production to high-level architectural design reasoning. This challenge is amplified in fast-evolving domains like LLM-agent development, where both rigid curricula (prone to rapid obsolescence) and fully open-ended student projects (creating unsustainable mentorship load) fail to scale effectively.

To resolve this tension, the authors propose **Speculative Scaffolding**: a two-phase pedagogical framework that aggregates heterogeneous student projects into shared technical routes, supports structured design iteration via the Replicate-Transfer-Analyze-Modify (RTAM) cycle, and balances student autonomy with instructional scalability. The framework was deployed in a 16-week cross-disciplinary undergraduate AI course and evaluated via mixed methods including surveys, TA workload analysis, project trajectory observation, and student interviews. Findings indicate the framework is associated with earlier architectural stabilization, more systematic student design reasoning, and a shift of instructional workload from late-stage debugging to early-stage design framing.

Overall, this is a well-conceived, practice-oriented contribution highly aligned with the Experience Reports and Tools track mission.

### Familiarity

**High.** I have extensive research and reviewing experience in computing education, with a specific focus on generative AI’s impact on CS pedagogy, scaffolding design for project-based learning, and instructional practices for AI/agent development courses. I am deeply familiar with the core challenges of teaching open-ended LLM-agent projects and the SIGCSE ERT track review criteria.

### Criterion-by-Criterion Evaluation

#### 1. Motivation (ERT)

**Strength:** 
The motivation is exceptionally well-grounded and timely. The authors clearly articulate the paradigm shift in CS education: as LLMs commoditize code implementation, student learning challenges have moved from syntax and debugging to architectural design, failure attribution, and design tradeoff reasoning under uncertainty.

The paper further sharpens this problem by focusing on LLM-agent development, where non-deterministic system behavior and rapid technological evolution break both traditional curriculum-centered models and open student-directed models. This is a widely felt pain point for CS/AI educators today, and the contextual framing is clear, specific, and highly relevant to the SIGCSE community.

**Minor gap:** The paper could more explicitly situate the target student population’s needs in the introduction, though this is addressed later in the deployment section.

#### 2. Prior and Related Work

**Strength:** 
The related work is well-structured across three relevant dimensions: the shift from syntax to system design in GenAI CS education, architectural uncertainty in LLM-agent development, and the tension between autonomy and scalability in project-based learning.

The authors effectively map the existing research landscape, clearly establishing that while prior work has documented the implementation bottleneck shift, there is a scarcity of actionable pedagogical frameworks for supporting architectural design reasoning in high-uncertainty, fast-evolving technical domains. This properly motivates the novelty of the proposed framework.

**Minor gap:** 
The paper could deepen its connection to broader scaffolding theory in education (e.g., adaptive vs. fixed scaffolding) to better ground the "speculative scaffolding" concept theoretically, though this is not a critical requirement for the ERT track.

#### 3. Approach

**Strength:**
The Speculative Scaffolding framework is described in clear, actionable detail, making it highly adoptable by other practitioners — a core strength for the ERT track.

The two-phase operational workflow (Phase 1: Architecture Discovery & Curriculum Alignment; Phase 2: Pattern-Based Mentorship & Design Iteration) is logically structured. The three accompanying instruments (Student Project Scoping Blueprint, TA Architecture Synthesis Guide, RTAM Failure Attribution Protocol) provide concrete, reusable tools for educators. The deployment context (16-week cross-disciplinary undergraduate AI course, comparison to a prior un-scaffolded cohort) is clearly described, and the course timeline figure effectively illustrates how the framework integrates into the term.

The RTAM cycle is a particularly strong pedagogical design: it replaces unstructured prompt tinkering with a systematic, hypothesis-driven iteration method tailored to the non-determinism of agent systems.

**Gaps:**
- Key implementation details are under-described: there is limited information about how teaching assistants were trained and calibrated to apply the framework, the specific workflow for AI-assisted architectural pattern aggregation, and the standards for developing the shared technical exemplars. These details would be critical for other educators seeking to replicate the practice.

- The paper does not address how the framework accommodates highly idiosyncratic student projects that do not fit into the 3-4 dominant technical routes, a common edge case in open project courses.

#### 4. Evidence

**Strength:** 
The paper draws on multiple sources of evidence to evaluate the framework, including student perception surveys, TA workload tracking, observational analysis of student project trajectories, and qualitative student interview quotes.

The before-and-after comparison of instructional workload distribution (peak shifting from late-stage debugging to early-stage design framing, with zero reported late-stage architectural redefinition) is a particularly compelling and practice-relevant finding. The paper correctly scopes its claims to the context of the deployment, avoiding overgeneralization.

**Gaps:**
- Reflection on failures and challenges is insufficient. ERT papers thrive on rich reflection on what did and did not work, but this paper focuses almost exclusively on positive outcomes. There is no discussion of unexpected barriers, student resistance, edge cases where the scaffold failed, or tradeoffs of the approach (e.g., whether architectural grouping limits student creativity, or whether exemplars encourage over-reliance). This limits the utility of the experience for other practitioners.
- The sample size is small (N=18 intervention, N=17 control), and there is no objective, pre-post measure of student design reasoning ability or project design quality — all learning outcome data is self-reported perception. While statistical rigor is not required for ERT, more objective evidence would strengthen the claims.
- There is no analysis of differential outcomes across student subgroups (e.g., CS vs. non-CS majors, low vs. high prior programming experience), which would add nuance to the findings.

#### 5. Contribution & Impact

**Strength:** 

This paper makes a valuable, practice-focused contribution to the SIGCSE community:

1. It delivers a complete, adoptable pedagogical framework with reusable tools for teaching open-ended LLM-agent projects, addressing a widespread and growing need as AI agent courses expand.
2. It articulates a clear paradigm for teaching in fast-evolving technical domains: instead of teaching fixed technical content, educators can scaffold structured design reasoning and hypothesis-testing practices that transfer across technological changes.
3. It provides a proven model for scaling mentorship in open project courses, by aggregating individual projects into shared architectural cohorts — a solution applicable beyond agent development to other fast-changing CS subfields.

The contribution is highly aligned with the ERT track’s mission of enabling adoption by other practitioners.

#### 6. Presentation

**Strength:** 
The paper is well-structured, clearly written, and logically organized. Figures are clear and support comprehension of the framework and findings. The use of formal instruments to document scaffolding practices improves readability and reusability. The paper adheres to double-anonymity requirements.

**Minor issues:**
Some framing language leans toward computing education research (CER) phrasing (e.g., "mixed-methods deployment study") rather than the observational, practice-focused tone typical of ERT. Minor rewording would better align with the track.
A few implementation details (e.g., timing of scoping blueprint submissions, cohort formation timeline) are buried in the text and would benefit from more explicit labeling.

#### 7. Reference Integrity
All 29 references appear to be genuine, appropriately cited, and relevant to the claims they support. The bibliography includes foundational education theory, key CS education research, and recent work on LLM agents, with accurate venue and author information. No hallucinated references are apparent.

### Core Evaluation Dimensions

#### Novelty
This work makes meaningful novel contributions to computing education practice:

1. **Conceptual novelty:** The idea of "speculative scaffolding" — dynamic, cohort-generated architectural anchors that evolve alongside student projects, rather than pre-defined fixed scaffolds — addresses a key gap in project-based learning for fast-changing technical domains. This differs from traditional scaffolding models, which are designed for stable, well-defined knowledge and skills.
2. **Methodological novelty:** The RTAM iteration cycle is a targeted, innovative pedagogical method for teaching design reasoning under non-determinism. It transforms the common, unproductive practice of ad-hoc prompt tinkering into a structured, hypothesis-driven engineering process, specifically tailored to the unique challenges of LLM-agent development.
3. **Practical novelty:** The two-phase aggregation model for scaling open-project mentorship is a novel operational solution to the widely recognized problem of unsustainable TA workload in student-directed project courses.

While related ideas exist in broader project-based learning literature, their application to LLM-agent education and the integrated framework presented here are new to the CS education community.

#### Strengths

1. **Timely, well-motivated problem:** The paper accurately identifies and articulates a core, widely experienced pedagogical shift brought on by generative AI in CS education, with a sharp focus on the under-addressed domain of LLM-agent projects.
2. **Highly adoptable framework:** The Speculative Scaffolding framework is complete, well-structured, and accompanied by concrete, reusable instruments, making it directly usable by other educators — a major strength for the ERT track.
3. **Innovative RTAM cycle:** The RTAM method provides a much-needed structured approach for teaching iterative design and debugging in non-deterministic AI systems, filling a critical gap in existing pedagogical resources.
4. **Multi-faceted evaluation:** The use of multiple data sources (surveys, workload metrics, observations, interviews) and a prior-cohort comparison provides credible, practice-relevant evidence for the framework’s effects.
5. **Scalability solution:** The cohort-based aggregation model offers a practical path to scaling mentorship for open-ended projects, without sacrificing student autonomy — a balance that has been difficult to achieve in prior models.

#### Weaknesses

1. **Limited reflection on challenges and failures:** The paper lacks robust discussion of what did not work, implementation barriers, or unintended consequences of the framework. This is a notable gap for an experience report, as practitioners need to understand both successes and pitfalls to adopt the approach.
2. **Small, single-context deployment:** The evaluation is limited to one small cohort in a single course, which constrains the generalizability of the findings. The paper would benefit from clearer discussion of how the framework might (or might not) translate to larger classes, different student populations, or other technical domains.
3. **Under-specified implementation details:** Key operational details (TA training, AI-assisted aggregation workflows, exemplar development standards, handling of outlier projects) are not sufficiently described, which could hinder replication by other educators.
4. **Lack of objective learning outcomes:** The evaluation relies heavily on self-reported student perceptions and instructional workload metrics, with no objective measurement of improvements in students’ design reasoning abilities or project quality.

### Overall Recommendation

**Accept**

This paper is an excellent fit for the SIGCSE TS 2027 Experience Reports and Tools track. It addresses a highly relevant, timely problem in CS education, presents a complete and adoptable pedagogical framework with novel elements, and provides credible practice-based evidence of its effects. While the paper has areas for improvement — particularly around deeper reflection on challenges and more detailed implementation guidance — these are readily addressable in revision and do not undermine the paper’s significant value to the CS education practitioner community.

#### Specific Revision Suggestions
1. Add a dedicated subsection reflecting on implementation challenges, limitations, and unexpected outcomes. For example: Did any students struggle with the scoping blueprint or RTAM cycle? Were there projects that did not fit the aggregated technical routes, and how were those handled? What tradeoffs came with shifting workload earlier in the term?
2. Expand implementation details: Briefly describe how TAs were trained to conduct architectural synthesis and RTAM-guided mentorship; outline the basic workflow for AI-assisted proposal summarization; and explain how technical exemplars were developed and maintained.
3. Clarify the framework’s boundary conditions: Explicitly discuss what types of courses and student populations this framework is (and is not) well-suited for, and how it might be adapted for larger class sizes.
4. Soften research-focused language (e.g., rephrase "mixed-methods deployment study" to "mixed-methods evaluation of our classroom deployment") to better align with the ERT track’s observational, practice-focused framing.

