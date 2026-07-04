"""
agents/company_agents.py

Defines all 5 agents for the company analysis pipeline and wires them into a
LangGraph StateGraph.

Pipeline:
  DataCollectorAgent → ResearchAgent → AnalystAgent → DecisionAgent → PredictorAgent

Each agent is a plain Python function that:
  - receives the current CompanyAnalysisState
  - calls Groq (via langchain-groq) with a purpose-built prompt
  - returns a dict of state updates
"""

import json
import os
from typing import Dict, Any

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from agents.state import CompanyAnalysisState

# ── Groq LLM (shared across all agents) ─────────────────────────────────────
# llama3-70b gives the best balance of speed and reasoning on Groq's free tier.
def _get_llm(temperature: float = 0.3) -> ChatGroq:
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=temperature,
    )


def _parse_json_safe(text: str) -> Dict:
    """Try to extract JSON from an LLM response, fallback to empty dict."""
    try:
        # Sometimes the model wraps JSON in markdown fences
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())
    except Exception:
        return {"raw_text": text}


# ─────────────────────────────────────────────────────────────────────────────
#  AGENT 1 — Data Collector
#  Role: Review the data the user has shared, identify gaps, and ask for more.
# ─────────────────────────────────────────────────────────────────────────────
def data_collector_agent(state: CompanyAnalysisState) -> Dict:
    """
    Checks if the user has shared enough data to perform a meaningful analysis.
    If not, it asks one targeted follow-up question.
    """
    llm = _get_llm(temperature=0.2)

    system_prompt = """You are a Financial Data Collection Specialist.
Your job is to review data the user has shared about their company and decide if it is
sufficient for a deep financial analysis.

Respond ONLY with valid JSON in this exact structure:
{
  "has_sufficient_data": true/false,
  "data_request_message": "Question to ask the user if more data is needed, else empty string",
  "collected_data_summary": "Brief summary of what data you have",
  "data_gaps": ["list", "of", "missing", "data", "points"],
  "collected_data": { "key": "value structured version of data" }
}

Be specific. If the user shared revenue data, ask for marketing or employee data next.
Only ask for ONE type of additional data at a time so you don't overwhelm the user."""

    user_content = f"""
Company: {state.get('company_name', 'Unknown')}
User Question: {state.get('user_question', '')}
Data Provided:
{json.dumps(state.get('company_data', {}), indent=2)}

Conversation so far:
{json.dumps(state.get('conversation_history', []), indent=2)}
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content)
    ])

    parsed = _parse_json_safe(response.content)

    return {
        "collected_data": parsed.get("collected_data", state.get("company_data", {})),
        "needs_more_data": not parsed.get("has_sufficient_data", True),
        "data_request_message": parsed.get("data_request_message", ""),
        "current_step": "data_collection_complete",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  AGENT 2 — Research Agent
#  Role: Simulate web research about the company's industry, competitors, trends.
# ─────────────────────────────────────────────────────────────────────────────
def research_agent(state: CompanyAnalysisState) -> Dict:
    """
    Researches the company's market environment, sector trends, and competitors.
    In production this would call Tavily/SerpAPI; here Groq simulates it.
    """
    llm = _get_llm(temperature=0.4)

    system_prompt = """You are a Market Research Analyst with expertise in business intelligence.
Based on the company data provided, generate research findings about:
1. Industry trends affecting this company
2. Common competitor strategies
3. Market conditions relevant to the company's problems
4. Benchmarks the company should be measured against

Respond ONLY with valid JSON:
{
  "industry_trends": ["trend1", "trend2"],
  "competitor_insights": ["insight1", "insight2"],
  "market_conditions": "description of current market",
  "benchmarks": {
    "industry_avg_growth": "X%",
    "avg_profit_margin": "X%",
    "key_success_factors": ["factor1", "factor2"]
  },
  "external_risks": ["risk1", "risk2"],
  "external_opportunities": ["opp1", "opp2"]
}"""

    user_content = f"""
Company: {state.get('company_name')}
Question: {state.get('user_question')}
Company Data: {json.dumps(state.get('collected_data', {}), indent=2)}
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content)
    ])

    return {
        "research_findings": _parse_json_safe(response.content),
        "current_step": "research_complete",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  AGENT 3 — Data Analyst
#  Role: Combine internal data + research findings into a structured report.
# ─────────────────────────────────────────────────────────────────────────────
def analyst_agent(state: CompanyAnalysisState) -> Dict:
    """
    Produces a structured analysis report: identifies problems in product,
    marketing, operations, and finances.
    """
    llm = _get_llm(temperature=0.3)

    system_prompt = """You are a Senior Financial & Business Data Analyst.
Combine the company's internal data with market research to produce a comprehensive
diagnostic report. Identify root causes of the problems the user asked about.

Respond ONLY with valid JSON:
{
  "executive_summary": "2-3 sentence overview",
  "problem_areas": {
    "product": { "severity": "high/medium/low", "issues": ["issue1"], "impact": "description" },
    "marketing": { "severity": "high/medium/low", "issues": ["issue1"], "impact": "description" },
    "operations": { "severity": "high/medium/low", "issues": ["issue1"], "impact": "description" },
    "finance": { "severity": "high/medium/low", "issues": ["issue1"], "impact": "description" },
    "human_resources": { "severity": "high/medium/low", "issues": ["issue1"], "impact": "description" }
  },
  "key_metrics": {
    "metric_name": { "current": "value", "benchmark": "value", "status": "good/warning/critical" }
  },
  "root_causes": ["cause1", "cause2"],
  "strengths": ["strength1", "strength2"],
  "weaknesses": ["weakness1", "weakness2"]
}"""

    user_content = f"""
Company: {state.get('company_name')}
User Question: {state.get('user_question')}
Internal Data: {json.dumps(state.get('collected_data', {}), indent=2)}
Research Findings: {json.dumps(state.get('research_findings', {}), indent=2)}
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content)
    ])

    return {
        "analysis_report": _parse_json_safe(response.content),
        "current_step": "analysis_complete",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  AGENT 4 — Decision Agent
#  Role: Turn the analysis into specific, actionable business decisions.
# ─────────────────────────────────────────────────────────────────────────────
def decision_agent(state: CompanyAnalysisState) -> Dict:
    """
    Converts the analysis report into a prioritised list of decisions
    the business owner should make RIGHT NOW.
    """
    llm = _get_llm(temperature=0.3)

    system_prompt = """You are a Chief Business Strategy Officer and Decision-Making Expert.
Based on the analysis report, generate concrete, prioritised business decisions.
Each decision must be specific, measurable, and actionable.

Respond ONLY with valid JSON — an array of decision objects:
[
  {
    "priority": 1,
    "decision": "Specific action to take",
    "rationale": "Why this decision is critical",
    "expected_impact": "What improvement this will create",
    "timeline": "Immediate / 30 days / 90 days / 6 months",
    "resources_needed": "Budget / team / tools required",
    "kpi_to_track": "How to measure success",
    "area": "Product / Marketing / Finance / Operations / HR"
  }
]
Generate 5-8 decisions ordered by priority."""

    user_content = f"""
Company: {state.get('company_name')}
Question: {state.get('user_question')}
Analysis Report: {json.dumps(state.get('analysis_report', {}), indent=2)}
Research Context: {json.dumps(state.get('research_findings', {}), indent=2)}
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content)
    ])

    raw = _parse_json_safe(response.content)
    decisions = raw if isinstance(raw, list) else raw.get("decisions", [raw])

    return {
        "decisions": decisions,
        "current_step": "decisions_complete",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  AGENT 5 — Predictor / Data Scientist
#  Role: Forecast future performance and recommend long-term growth strategies.
# ─────────────────────────────────────────────────────────────────────────────
def predictor_agent(state: CompanyAnalysisState) -> Dict:
    """
    Forecasts the company's future (3–12 months) based on current trajectory
    and recommended decisions. Also writes the final human-readable narrative.
    """
    llm = _get_llm(temperature=0.4)

    system_prompt = """You are a Data Scientist and Financial Forecasting Expert.
Using the company data, analysis, and recommended decisions:
1. Predict future performance under two scenarios: "do nothing" vs "implement decisions"
2. Generate specific growth strategies for the next 12 months
3. Write a final human-readable narrative summary

Respond ONLY with valid JSON:
{
  "predictions": {
    "without_action": {
      "3_months": "description",
      "6_months": "description",
      "12_months": "description",
      "projected_revenue_change": "-X%",
      "risk_level": "high/medium/low"
    },
    "with_action": {
      "3_months": "description",
      "6_months": "description",
      "12_months": "description",
      "projected_revenue_change": "+X%",
      "risk_level": "high/medium/low"
    }
  },
  "growth_strategies": [
    {
      "strategy": "Strategy name",
      "description": "Details",
      "expected_roi": "X%",
      "timeframe": "X months",
      "difficulty": "easy/medium/hard"
    }
  ],
  "future_actions": ["action1", "action2", "action3"],
  "final_narrative": "Full markdown narrative with ## headings covering: what's wrong, what to do, what will happen"
}"""

    user_content = f"""
Company: {state.get('company_name')}
Question: {state.get('user_question')}
Analysis: {json.dumps(state.get('analysis_report', {}), indent=2)}
Decisions: {json.dumps(state.get('decisions', []), indent=2)}
Research: {json.dumps(state.get('research_findings', {}), indent=2)}
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content)
    ])

    parsed = _parse_json_safe(response.content)

    return {
        "predictions": parsed.get("predictions", {}),
        "future_actions": parsed.get("future_actions", []),
        "final_response": parsed.get("final_narrative", response.content),
        "current_step": "prediction_complete",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Routing logic — decides whether to pause and ask the user for more data
# ─────────────────────────────────────────────────────────────────────────────
def route_after_collection(state: CompanyAnalysisState) -> str:
    """
    After the DataCollectorAgent runs:
      - if more data is needed  → return "needs_data" (pipeline pauses)
      - if data is sufficient   → return "research"   (pipeline continues)
    """
    if state.get("needs_more_data", False):
        return "needs_data"
    return "research"


# ─────────────────────────────────────────────────────────────────────────────
#  Build the LangGraph pipeline
# ─────────────────────────────────────────────────────────────────────────────
def build_company_analysis_graph() -> StateGraph:
    """
    Constructs and compiles the 5-agent LangGraph workflow.

    Graph structure:
      [START]
         ↓
      data_collector
         ↓  (conditional)
      ┌──────────────────┐
      │ needs_data → END │  (frontend shows data_request_message)
      └──────────────────┘
         ↓ has data
      research
         ↓
      analyst
         ↓
      decision
         ↓
      predictor
         ↓
      [END]
    """
    graph = StateGraph(CompanyAnalysisState)

    # Register agent nodes
    graph.add_node("data_collector", data_collector_agent)
    graph.add_node("research",       research_agent)
    graph.add_node("analyst",        analyst_agent)
    graph.add_node("decision",       decision_agent)
    graph.add_node("predictor",      predictor_agent)

    # Entry point
    graph.set_entry_point("data_collector")

    # Conditional edge after data collection
    graph.add_conditional_edges(
        "data_collector",
        route_after_collection,
        {
            "needs_data": END,   # pause — ask user for more info
            "research":   "research",
        }
    )

    # Linear edges for the rest of the pipeline
    graph.add_edge("research",  "analyst")
    graph.add_edge("analyst",   "decision")
    graph.add_edge("decision",  "predictor")
    graph.add_edge("predictor", END)

    return graph.compile()


# Singleton — compile once at import time
company_analysis_graph = build_company_analysis_graph()
