# assets.py
from __future__ import annotations

from typing import Any, List, Literal, Optional, TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# Schemas
# =========================================================

class InputField(BaseModel):
    key: str
    name: str
    desc: str
    benchmark_stats: Optional[str]


class ComputedField(BaseModel):
    key: str
    name: str
    desc: str

    op: str            # e.g., "ratio_mean_product" | "ratio_mean_diff"
    fields: List[str]  # input field keys
    benchmark_stats: Optional[str] = None  # always None for computed


# =========================================================
# Insights Package Schemas (for LLM output)
# =========================================================

HealthLabel = Literal["green", "yellow", "red"]
ConfidenceLabel = Literal["low", "med", "high"]
SeverityLabel = Literal["low", "med", "high"]
ImpactLabel = Literal["low", "med", "high"]
Visibility = Literal["internal", "both"]
ActionOwner = Literal["CSM", "Customer", "Both"]


# EvidenceItem is just a string in practice, but we validate it via the evidence lists
# No separate model needed - evidence is List[str] in the models above


class AccountSummary(BaseModel):
    one_liner: str = Field(..., min_length=1)
    health_label: HealthLabel
    confidence: ConfidenceLabel


class Win(BaseModel):
    title: str = Field(..., min_length=1)
    why_it_matters: str = Field(..., min_length=1)
    evidence: List[str] = Field(..., min_length=1)


class Risk(BaseModel):
    title: str = Field(..., min_length=1)
    severity: SeverityLabel
    why_it_matters: str = Field(..., min_length=1)
    visibility: Visibility
    evidence: List[str] = Field(..., min_length=1)


class Opportunity(BaseModel):
    title: str = Field(..., min_length=1)
    impact: ImpactLabel
    why_it_matters: str = Field(..., min_length=1)
    visibility: Visibility
    evidence: List[str] = Field(..., min_length=1)


class RecommendedAction(BaseModel):
    owner: ActionOwner
    next_step: str = Field(..., min_length=1)
    expected_outcome: str = Field(..., min_length=1)
    visibility: Visibility
    evidence: List[str] = Field(..., min_length=1)


class ResearchSignal(BaseModel):
    title: str = Field(..., min_length=1)
    short_summary: str = Field(..., min_length=1)
    source: Literal["research"] = "research"


class InsightsPackage(BaseModel):
    """
    Top-level schema for the "insights package" step.
    """
    timeframe: str = Field(..., min_length=1)
    account_identifier: str = Field(..., min_length=1)

    account_summary: AccountSummary

    wins: List[Win] = Field(default_factory=list)
    risks: List[Risk] = Field(default_factory=list)
    opportunities: List[Opportunity] = Field(default_factory=list)

    recommended_actions: List[RecommendedAction] = Field(default_factory=list)

    # Only populate when research input is provided; otherwise output [].
    research_signals: List[ResearchSignal] = Field(default_factory=list)

    # Optional: capture explicit assumptions when the model had to infer,
    # but keep them separated to avoid leaking into client-safe content.
    assumptions: List[str] = Field(default_factory=list)


# =========================================================
# Formatting Output Schemas (for LLM output)
# =========================================================

class EvidenceMapItem(BaseModel):
    id: str = Field(..., description="Short ID like 'E1', 'E2', ...")
    evidence: str = Field(..., description="Original evidence string from extracted_insights (raw:/computed:/text:/research:)")


class InternalBullet(BaseModel):
    title: str
    detail: str = Field(..., description="1–2 sentences. Candid, operational.")
    evidence_ids: List[str] = Field(default_factory=list, description="List of EvidenceMapItem.id")


class InternalAction(BaseModel):
    priority: int = Field(..., ge=1, description="1 = highest priority")
    title: str
    owner: Literal["CSM", "Customer", "Both"]
    next_step: str
    expected_outcome: str
    evidence_ids: List[str] = Field(default_factory=list)


class InternalBrief(BaseModel):
    executive_snapshot: str = Field(..., description="Short paragraph summarizing state of the account for the quarter.")
    health_label: Literal["green", "yellow", "red"]
    confidence: Literal["low", "med", "high"]

    wins: List[InternalBullet] = Field(default_factory=list)
    risks: List[InternalBullet] = Field(default_factory=list, description="Include mitigation framing in detail where appropriate.")
    opportunities: List[InternalBullet] = Field(default_factory=list)

    action_plan: List[InternalAction] = Field(default_factory=list)

    company_news_snapshot: str = Field(
        ...,
        description="1–3 bullets as single short paragraph OR 'No meaningful news detected in the last few months.'"
    )
    company_news_confidence: Optional[Literal["low", "med", "high"]] = Field(
        default=None,
        description="Confidence in relevance/quality of research findings"
    )

    evidence_map: List[EvidenceMapItem] = Field(
        default_factory=list,
        description="Deduplicated mapping of evidence strings to short IDs."
    )


class ExternalSection(BaseModel):
    title: str
    bullets: List[str] = Field(default_factory=list)


class ExternalQBR(BaseModel):
    opening_narrative: str = Field(..., description="1 short paragraph, client-safe. Must include 1–2 concrete numeric proof points when available.")
    highlights: List[str] = Field(default_factory=list, description="1–2 consolidated theme bullets MAX. Merge related metrics into single highlights.")
    risks_and_watchouts: List[str] = Field(default_factory=list, description="Only client-safe. Constructive framing. Soften risk language by default.")
    recommendations_next_steps: List[str] = Field(default_factory=list, description="3–5 bullets, actionable.")
    closing: str = Field(default="", description="1–2 sentence forward-looking closing that aligns on next-quarter focus and shared ownership. Tone: collaborative, optimistic, concrete.")


class FormattedOutput(BaseModel):
    timeframe: str
    account_identifier: str

    internal_csm_brief: InternalBrief
    external_qbr_summary: ExternalQBR


# =========================================================
# Research Summarization Schemas
# =========================================================

class ResearchDigest(BaseModel):
    status: Literal["relevant", "none"]
    bullets: List[str] = Field(
        default_factory=list,
        description="0–3 concise bullets, company-specific, last few months only"
    )
    timeframe_months: int = Field(3, description="Window used for recency phrasing")







PIPELINE_INPUT_FIELDS: List[InputField] = [
    {
        "key": "account_name",
        "name": "Account name",
        "desc": "Customer/account identifier (human-readable name).",
        "benchmark_stats": None,
    },
    {
        "key": "plan_type",
        "name": "Plan type",
        "desc": "Subscription tier / plan category (e.g., Basic, Pro, Enterprise).",
        "benchmark_stats": None,
    },
    {
        "key": "active_users",
        "name": "Active users",
        "desc": "Number of active users/seats in the account during the measured period.",
        "benchmark_stats": "Mean: 188.6, SD: 170.0",
    },
    {
        "key": "usage_growth_qoq",
        "name": "Usage growth (QoQ)",
        "desc": "Quarter-over-quarter usage growth rate as a fraction (e.g., 0.10 = +10%, -0.12 = -12%).",
        "benchmark_stats": "Mean: 0.066, SD: 0.131",
    },
    {
        "key": "automation_adoption_pct",
        "name": "Automation adoption",
        "desc": "Share of the account's usage/workflows that leverage automations (0–1 fraction; higher = more adoption).",
        "benchmark_stats": "Mean: 0.442, SD: 0.283",
    },
    {
        "key": "tickets_last_quarter",
        "name": "Support tickets (last quarter)",
        "desc": "Number of support tickets opened by this account in the last quarter.",
        "benchmark_stats": "Mean: 52.2, SD: 32.8",
    },
    {
        "key": "avg_response_time",
        "name": "Avg response time",
        "desc": "Average support response time for the account (in hours).",
        "benchmark_stats": "Mean: 4.82, SD: 1.92",
    },
    {
        "key": "nps_score",
        "name": "NPS score",
        "desc": "Customer satisfaction / loyalty proxy provided for the account (numeric score as given in the dataset).",
        "benchmark_stats": "Mean: 7.76, SD: 1.37",
    },
    {
        "key": "preferred_channel",
        "name": "Preferred channel",
        "desc": "Primary communication channel preference (e.g., Email, Phone, In-app).",
        "benchmark_stats": None,
    },
    {
        "key": "scat_score",
        "name": "SCAT score",
        "desc": "Internal account health/sentiment score (numeric; higher typically indicates healthier sentiment).",
        "benchmark_stats": "Mean: 75.2, SD: 17.7",
    },
    {
        "key": "risk_engine_score",
        "name": "Risk engine score",
        "desc": "Internal churn/renewal risk score (0–1; higher typically indicates higher risk).",
        "benchmark_stats": "Mean: 0.308, SD: 0.276",
    },
    {
        "key": "crm_notes",
        "name": "CRM notes",
        "desc": "Qualitative CSM/CRM notes (context, recent events, stakeholder updates, narrative).",
        "benchmark_stats": None,
    },
    {
        "key": "feedback_summary",
        "name": "Feedback summary",
        "desc": "Summarized customer feedback and requests captured for the period.",
        "benchmark_stats": None,
    },
]



PIPELINE_COMPUTED_FIELDS: List[ComputedField] = [
    {
        "key": "support_burden_index",
        "name": "Support burden index",
        "desc": "Composite support load from tickets and response time (each normalized by mean).",
        "op": "ratio_mean_product",
        "fields": ["tickets_last_quarter", "avg_response_time"],
    },
    {
        "key": "adoption_risk_gap",
        "name": "Adoption–risk gap",
        "desc": "Difference between normalized automation adoption and normalized risk.",
        "op": "ratio_mean_diff",
        "fields": ["automation_adoption_pct", "risk_engine_score"],
    },
    {
        "key": "growth_vs_satisfaction_gap",
        "name": "Growth vs satisfaction gap",
        "desc": "Difference between normalized usage growth and normalized NPS.",
        "op": "ratio_mean_diff",
        "fields": ["usage_growth_qoq", "nps_score"],
    },
]



# =========================================================
# Prompts
# =========================================================

REASONING_PROMPT_TEMPLATE_OLD = """SYSTEM (role):
You are an expert Customer Success Strategy Analyst.
You generate data-grounded account insights that will later be formatted into:
(A) an internal CSM summary (candid, diagnostic), and
(B) a client-facing QBR-ready narrative (polished, safe).

You MUST stay grounded in the provided data.
Do NOT invent facts.

OBJECTIVE:
Given (1) a list of features (raw + computed), (2) optional research bullets, and (3) a timeframe label,
produce a complete, structured "insights package" that contains every insight needed later to write
BOTH internal and external outputs.

This step produces the full insight substrate.
Do NOT decide section placement or wording for Part A vs Part B.
Only control visibility via explicit visibility fields.

GROUNDING RULES (CRITICAL):
- Use ONLY the provided features and provided research bullets.
- If a claim cannot be supported, either:
  - omit it, or
  - mark it explicitly as an assumption with low confidence.
- Every win, risk, opportunity, and action MUST include evidence.
- Evidence items MUST be strings prefixed with exactly one of:
  - "raw:<key>=<value>"          (a raw feature value)
  - "computed:<key>=<value>"     (a computed feature value)
  - "text:<quote or paraphrase>" (from qualitative text fields in the features)
  - "research:<bullet>"          (only if research is provided)
- Do not exceed what the data supports.
- Prefer specific, falsifiable statements over general claims.
- Quantitative signals (usage, adoption, volume, scores) should be extracted once and reused across
  insights with different interpretations. Do NOT treat metrics as external-only.
- Separate internal-only vs client-safe insights strictly via the visibility field.
- Benchmark stats (if present) may be used to say "high/low relative to peers" ONLY if the benchmark
  text explicitly supports that claim.

SPARSITY RULES (CRITICAL):
- Wins, Risks, Opportunities, and Actions are ALL optional (0–N items).
- It is valid to return zero risks in healthy quarters.
- It is valid to return zero opportunities if no clear expansion signal exists.
- Prefer omission over weak or generic insights.
- Generic statements like "monitor as usage scales" must NOT be generated.
- Only include items that are materially significant and directly supported by the data.

INPUTS YOU WILL RECEIVE (INJECTED AT RUNTIME):

- timeframe: {timeframe}
  (e.g., "Q3 2025")

- account_identifier: {account_identifier}
  (e.g., account name or account ID)

- features: a list of objects, each EXACTLY:
  {{
    "key": "<string>",
    "name": "<string>",
    "desc": "<string>",
    "benchmark_stats": "<string or null>",
    "value": <any scalar or short text>
  }}

  Notes:
  - All features share the same structure.
  - benchmark_stats may be null or empty.
  - Some feature values may be null or missing.

{research_section}

WHAT TO PRODUCE:

1) Account Summary
   - one_liner: a concise description of the quarter
   - health_label: green | yellow | red
     - Must be justified by at least TWO independent signals
     - If signals conflict, default to yellow
   - confidence: low | med | high
     - Based on data completeness
     - Based on consistency across signals

2) Wins
   - What went well this quarter (usage, adoption, sentiment, operations)
   - A win must represent at least one of:
     - positive change,
     - strong absolute level (if benchmark-supported), or
     - resilience/stability under known constraints
   - Each win must include:
     - title
     - why_it_matters
     - evidence (list of evidence strings)

3) Risks
   - Churn, renewal, product, operational, or sentiment risks grounded in the data
   - Each risk must include:
     - title
     - severity: low | med | high
     - why_it_matters (include mitigation angle when relevant)
     - visibility: internal | both
     - evidence
   - MATERIALITY: Treat severity as a confidence signal.
     - Low-severity risks should generally be omitted.
     - Only include low-severity risks if driven by:
       - an explicit customer request, or
       - a concrete dependency clearly supported by data.

4) Opportunities
   - Expansion, enablement, automation/workflow improvement, value realization,
     executive alignment, or operational optimization
   - Each opportunity must include:
     - title
     - impact: low | med | high
     - why_it_matters
     - visibility: internal | both
   - Visibility should default to "both" unless it depends on internal scoring
     or fragile assumptions.
   - MATERIALITY: Treat impact as a confidence signal.
     - Low-impact opportunities should generally be omitted.
     - Only include low-impact opportunities if driven by:
       - an explicit customer request, or
       - a concrete dependency clearly supported by data.

5) Recommended Actions (MOST IMPORTANT OUTPUT)
   - Provide 1–5 actions MAX, ordered by priority
   - Each action must include:
     - owner: CSM | Customer | Both
     - next_step: a concrete step that could realistically be done next week
     - expected_outcome: measurable or clearly observable
     - visibility: internal | both
     - evidence
   - CRITICAL: Every action must be directly justified by a stated win, risk, or opportunity.
     - Do not generate actions that are not clearly linked to the insights above.
     - If no actionable insights exist, return fewer actions (or zero if truly none are warranted).

{research_signals_section}

OUTPUT FORMAT (STRICT):
- Return ONLY valid JSON.
- The JSON MUST match the provided Pydantic schema EXACTLY.
- No markdown.
- No commentary.
- No extra keys.

DATA YOU MUST USE:
Below is the injected feature list. This is your ONLY source of truth.

FEATURES (RAW + COMPUTED):
{features}

{research_data_section}

FINAL REMINDERS:
- This step creates ALL underlying insights.
- Later steps will control tone, phrasing, and layout for internal vs external outputs.
- Do NOT leak internal metrics or AI risk scores externally:
  if an insight depends on them, set visibility="internal".
"""

REASONING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert Customer Success Strategy Analyst.
You generate data-grounded account insights that will later be formatted into:
(A) an internal CSM summary (candid, diagnostic), and
(B) a client-facing QBR-ready narrative (polished, safe).

You MUST stay grounded in the provided data.
Do NOT invent facts.

Follow the instructions exactly."""),
    ("human", """OBJECTIVE:
Given (1) a list of features (raw + computed), (2) optional research bullets, and (3) a timeframe label,
produce a structured insights package that contains everything needed later to write BOTH internal and external outputs.

This step produces the full insight substrate.
Do NOT decide phrasing or layout for internal vs external outputs.
Control visibility only via explicit visibility fields.

GROUNDING RULES:
- Use ONLY the provided features and research bullets.
- If a claim cannot be supported, omit it or mark it as a low-confidence assumption.
- Every win, risk, opportunity, and action MUST include evidence.
- Evidence strings must be prefixed with exactly one of:
  - "raw:<key>=<value>"
  - "computed:<key>=<value>"
  - "text:<quote or paraphrase>"
  - "research:<bullet>"
- Do not exceed what the data supports.
- Prefer specific, falsifiable statements.
- Use benchmark comparisons ONLY if benchmark text explicitly supports them.
- Separate internal-only vs client-safe insights strictly via the visibility field.

RESEARCH USAGE RULES (CRITICAL):
- Research is optional and never overrides account data.
- Use research ONLY to:
  (a) validate an insight (increase confidence), or
  (b) challenge an insight/assumption (decrease confidence, add internal-only risk/watchout).
- If research contradicts an assumption, either:
  - add an internal-only risk with evidence research:<bullet>, or
  - add an assumption in assumptions[] explaining uncertainty.
- Do NOT add new opportunities/actions that are based only on research.

SPARSITY & MATERIALITY RULES:
- All sections are optional (0–N items).
- Prefer omission over weak, generic, or redundant insights.
- Do not generate generic monitoring statements.
- Merge multiple signals that describe the same underlying outcome.
- Fewer, higher-signal insights are preferred over completeness.

INPUTS (INJECTED AT RUNTIME):

- timeframe: {timeframe}
- account_identifier: {account_identifier}

- features: a list of objects, each exactly:
  {{
    "key": "<string>",
    "name": "<string>",
    "desc": "<string>",
    "benchmark_stats": "<string or null>",
    "value": <any scalar or short text>
  }}

Notes:
- benchmark_stats may be null or empty.
- Some feature values may be missing.

{research_section}

WHAT TO PRODUCE:

1) Account Summary
   - one_liner
   - health_label: green | yellow | red
     - Justified by at least two independent signals
     - If signals conflict, default to yellow
   - confidence: low | med | high
     - Based on data completeness and signal consistency

2) Wins
   - 0–3 items MAX
   - Each win must represent a distinct, high-signal outcome for the quarter
   - Combine related metrics into a single win
   - Each win must include:
     - title
     - why_it_matters
     - evidence

3) Risks
   - 0–3 items MAX
   - Each risk must include:
     - title
     - severity: low | med | high
     - why_it_matters (include mitigation angle when relevant)
     - visibility: internal | both
     - evidence
   - Low-severity risks should generally be omitted unless driven by:
     - an explicit customer request, or
     - a concrete dependency clearly supported by data

4) Opportunities
   - 0–3 items MAX
   - Each opportunity must include:
     - title
     - impact: low | med | high
     - why_it_matters
     - visibility: internal | both
   - Low-impact opportunities should generally be omitted unless driven by:
     - an explicit customer request, or
     - a concrete dependency clearly supported by data

5) Recommended Actions
   - 1–4 actions MAX, ordered by priority
   - Each action must include:
     - owner: CSM | Customer | Both
     - next_step: something realistically doable next week
     - expected_outcome: measurable or clearly observable
     - visibility: internal | both
     - evidence
   - Every action must be directly justified by a stated win, risk, or opportunity
   - If no strong actions are warranted, return fewer actions (or zero)

{research_signals_section}

OUTPUT FORMAT:
- Return ONLY valid JSON.
- The JSON must match the provided Pydantic schema exactly.
- No markdown, commentary, or extra keys.

DATA SOURCE:
The injected feature list below is your ONLY source of truth.

FEATURES:
{features}

{research_data_section}

FINAL RULE:
Omission is preferable to weak insight.""")
])



# =========================================================
# Output templates
# =========================================================

FORMATTING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a Customer Success QBR Writer + Editor.
You will receive (A) the structured "ExtractedInsights" JSON from the insights step, and (B) the account feature list (raw+computed), and you will produce TWO documents:
1) INTERNAL CSM BRIEF (candid, operational, diagnostic)
2) EXTERNAL QBR SUMMARY (client-safe, polished, ready to paste)

You MUST stay grounded in the provided inputs.
Do NOT invent metrics, events, timelines, product usage, or outcomes.
If something is uncertain, either omit it or label it as an assumption (and keep assumptions INTERNAL only).

Follow the instructions exactly."""),
    ("human", """VISIBILITY RULES (critical):
- Any item with visibility="internal" must never appear in the EXTERNAL QBR.
- Items with visibility="both" may appear in both.
- Do not include internal scoring logic, model/AI/risk scoring details, or speculation in EXTERNAL QBR.
- Research-derived content must never appear in external.

STYLE:
- Be concise. Avoid repetition across sections.
- Prefer specific statements anchored to evidence.
- Use neutral, professional tone. No sarcasm, no hype.
- Use the account's language from feature names where possible (but don't copy raw keys verbatim unless requested).

EVIDENCE POLICY:
- You may quote or paraphrase text evidence, but do not dump raw data.
- Every major claim in INTERNAL BRIEF should reference at least one evidence item ID (see "Evidence Map" below).
- EXTERNAL QBR should reference evidence implicitly (no "raw:key=…" strings), but must remain faithful to it.

INPUTS YOU WILL RECEIVE (injected at runtime):
- timeframe: {timeframe}
- account_identifier: {account_identifier}

- features: a list of objects, each exactly:
  {{
    "key": "<string>",
    "name": "<string>",
    "desc": "<string>",
    "benchmark_stats": "<string or null>",
    "value": <any scalar or short text>
  }}

- extracted_insights: JSON matching the ExtractedInsights schema:
  {{
    "account_summary": {{...}},
    "wins": [...],
    "risks": [...],
    "opportunities": [...],
    "recommended_actions": [...],
    "research_signals": [...]
  }}

{research_digest_section}

YOUR TASK:
Produce a single JSON object matching the output schema below. It must include:

A) INTERNAL CSM BRIEF
- Executive snapshot (1 short paragraph + health + confidence)
- Key wins (bullets)
- Key risks (bullets) + suggested mitigations
- Opportunities (bullets)
- Action plan (prioritized, 1–4 items, or fewer if fewer actions are provided)
- Company news snapshot (internet, last few months):
  - If extracted_insights.research_signals is empty OR research_digest.status="none":
    set company_news_snapshot to: "No meaningful company-specific news detected in the last few months."
  - Else summarize into 1–3 crisp bullets (single short paragraph), no speculation.
  - Set company_news_confidence based on research quality (low/med/high).
- Evidence map: assign short evidence IDs (E1, E2, …) and map them back to original evidence strings
  - Use the evidence list items from extracted_insights as your source.
  - Deduplicate identical evidence strings across items.

B) EXTERNAL QBR SUMMARY (client-facing)
- Opening: 1 paragraph account narrative for the quarter
  - MUST include 1–2 concrete numeric proof points when available (e.g., active_users, usage_growth_qoq, automation_adoption_pct, tickets_last_quarter, avg_response_time, nps_score, scat_score)
  - Select the strongest available metrics per account; do not force metrics if missing
  - Format percentages appropriately (e.g., "+16%" for usage_growth_qoq=0.16, "78%" for automation_adoption_pct=0.78)
- Highlights: 1–2 consolidated theme bullets MAX (not one bullet per metric)
  - Prefer themes: (A) adoption/growth/value, (B) sentiment/support experience (if relevant)
  - Merge related metrics into single highlights; avoid redundant bullets
  - Each highlight should include numeric proof points when available
- Risks & Watch-outs: only client-safe items (visibility="both"), phrased constructively
  - TONE GUARD: Soften risk language by default
  - Avoid words like "risk", "blocker", "avoid delays", "fragmentation" unless data explicitly indicates a risk
  - Prefer phrasing like "to support expansion / to move efficiently / to align"
- Recommendations / Next steps: 3–5 bullets, client-safe
- Closing: 1–2 sentence forward-looking closing (ALWAYS REQUIRED)
  - Align on next-quarter focus and shared ownership
  - Tone: collaborative, optimistic, concrete
- NO internal labels (green/yellow/red), NO "confidence", NO "severity/impact" words unless naturally phrased.
- NO org/team structure assumptions: Do not infer org structure unless explicitly mentioned in crm_notes or feedback_summary text fields.

SELECTION / PRIORITIZATION RULES:
- Prefer items with higher severity/impact in INTERNAL.
- EXTERNAL should focus on value delivered + clear next steps.
- Avoid duplicates: if a win and opportunity are essentially the same, keep it in the best-fitting section only.
- Keep total length reasonable:
  - INTERNAL: ~250–450 words equivalent
  - EXTERNAL: ~150–300 words equivalent

OUTPUT FORMAT RULES:
- Return ONLY valid JSON.
- Do not include markdown.
- Follow the schema exactly (no extra keys).
- Strings should be ready-to-use (final text), not placeholders.

RUNTIME INPUTS:
TIMEFRAME: {timeframe}
ACCOUNT: {account_identifier}

FEATURES:
{features}

EXTRACTED INSIGHTS:
{extracted_insights}""")
])


# =========================================================
# Research Summarization Prompt
# =========================================================

RESEARCH_SUMMARIZATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an assistant helping a Customer Success Manager prepare for a Quarterly Business Review (QBR).

Follow the instructions exactly."""),
    ("human", """TASK:
You are given recent online search results about a company.
Each result includes: title, url, snippet, score, and rank.

Your task is to extract 0–3 concise, company-specific business bullets from the last few months that could be meaningful for a QBR discussion or internal CSM preparation.

RULES (CRITICAL):
- Bullets must be COMPANY-SPECIFIC and RECENT (last 3–6 months only).
- Bullets must be BUSINESS-RELEVANT (e.g., financial performance, restructuring, pricing changes, product roadmap, major incidents, leadership changes, strategic announcements).
- Bullets must be FACTUAL and GROUNDED in the snippet/title (no inference, no speculation).
- Ignore biographies, social profiles, directory pages, generic company descriptions, or "about us" pages.
- Do NOT include generic "market trends" unless explicitly tied to a specific company event or announcement.
- If results are generic, irrelevant, or contain no meaningful company-specific news: set status="none" and bullets=[].
- Bullets should be concise phrases or short sentences (not full paragraphs).

OUTPUT:
Return structured output matching the ResearchDigest schema:
- status: "relevant" if meaningful company-specific bullets exist, "none" otherwise
- bullets: List[str] with 0–3 concise bullets (empty if status="none")
- timeframe_months: integer (typically 3, representing the recency window)

SEARCH RESULTS:
{search_results}""")
])
