"""
agents/state.py - Shared state definitions for all agents in Finance AI.

LangGraph passes this TypedDict between every agent node in the graph.
Think of it as a "shared whiteboard" all agents read from and write to.
"""
from typing import TypedDict, Optional, List, Dict, Any


# ──────────────────────────────────────────────
#  Company Analysis State
# ──────────────────────────────────────────────
class CompanyAnalysisState(TypedDict):
    """
    Flows through the 5-agent company analysis pipeline:
      1. DataCollectorAgent  - gathers user-provided data, asks for more if needed
      2. ResearchAgent       - searches the web for public company/market data
      3. AnalystAgent        - produces a structured report from all gathered data
      4. DecisionAgent       - turns the report into concrete business decisions
      5. PredictorAgent      - forecasts future performance & recommends actions
    """
    # --- Input from the user (front-end POST body) ---
    user_question: str            # e.g. "Why is my company going down?"
    company_name: str             # e.g. "Acme Corp"
    company_data: Dict[str, Any]  # whatever data the user already shared
    conversation_history: List[Dict[str, str]]  # multi-turn chat

    # --- DataCollectorAgent outputs ---
    collected_data: Dict[str, Any]   # merged user + follow-up data
    needs_more_data: bool            # True → ask user for more data
    data_request_message: str        # question sent back to user if True

    # --- ResearchAgent outputs ---
    research_findings: Dict[str, Any]  # publicly available info about company/sector

    # --- AnalystAgent outputs ---
    analysis_report: Dict[str, Any]  # structured analysis (problems, metrics, etc.)

    # --- DecisionAgent outputs ---
    decisions: List[Dict[str, Any]]  # list of recommended decisions with rationale

    # --- PredictorAgent outputs ---
    predictions: Dict[str, Any]   # revenue / growth forecasts
    future_actions: List[str]     # concrete steps the owner should take

    # --- Final assembled response ---
    final_response: str           # markdown narrative sent to the front-end
    current_step: str             # tracks which agent is currently running
    error: Optional[str]          # any error message


# ──────────────────────────────────────────────
#  Employee Financial State
# ──────────────────────────────────────────────
class EmployeeFinancialState(TypedDict):
    """
    Flows through the employee financial advisory pipeline.
    Triggered when a user uploads a salary slip and asks personal finance questions.
    """
    # --- Input ---
    user_question: str
    salary_data: Dict[str, Any]    # parsed from uploaded PDF/image
    conversation_history: List[Dict[str, str]]

    # --- Parsed salary info ---
    monthly_income: float
    monthly_expenses: float
    savings: float
    tax_deductions: float
    other_deductions: float
    net_salary: float

    # --- Advisory agent outputs ---
    emergency_fund_analysis: Dict[str, Any]   # "what if I lose my job?"
    savings_plan: Dict[str, Any]              # personalised savings recommendations
    investment_suggestions: List[Dict]        # where to invest surplus
    budget_breakdown: Dict[str, Any]          # 50/30/20 or similar split

    # --- Final response ---
    final_response: str
    dashboard_data: Dict[str, Any]  # JSON sent to front-end charts
    current_step: str
    error: Optional[str]
