"""
agents/employee_agents.py

Single agent that handles all employee financial advisory questions.
The agent:
  1. Receives parsed salary data (from PDF/image upload)
  2. Answers personal finance questions (job loss, savings, investments)
  3. Returns both a narrative answer AND structured dashboard data
"""

import json
import os
from typing import Dict, Any

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from agents.state import EmployeeFinancialState


def _get_llm(temperature: float = 0.3) -> ChatGroq:
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-70b-8192",
        temperature=temperature,
    )


def _parse_json_safe(text: str) -> Dict:
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())
    except Exception:
        return {"raw_text": text}


# ─────────────────────────────────────────────────────────────────────────────
#  AGENT — Employee Financial Advisor
# ─────────────────────────────────────────────────────────────────────────────
def employee_advisor_agent(state: EmployeeFinancialState) -> Dict:
    """
    Comprehensive personal finance advisor for employees.
    Answers: job loss scenario, savings strategies, investment planning,
    budget breakdown, and emergency fund recommendations.
    """
    llm = _get_llm(temperature=0.3)

    system_prompt = """You are a Personal Finance Expert and Certified Financial Planner.
You help employees make smart financial decisions based on their salary and expenses.

Given the salary data and the user's question, provide:
1. Direct answer to their specific question
2. Emergency fund analysis (how many months they can survive without income)
3. Personalised savings plan
4. Budget breakdown (income allocation)
5. Investment suggestions based on their surplus

Respond ONLY with valid JSON:
{
  "direct_answer": "Specific answer to the user's question",
  "emergency_fund_analysis": {
    "current_savings": 0,
    "monthly_expenses": 0,
    "months_can_survive": 0,
    "recommendation": "text",
    "target_emergency_fund": 0,
    "months_to_build": 0
  },
  "savings_plan": {
    "monthly_savings_target": 0,
    "savings_percentage": 0,
    "strategies": ["strategy1", "strategy2"],
    "6_month_projection": 0,
    "12_month_projection": 0
  },
  "budget_breakdown": {
    "needs": { "amount": 0, "percentage": 0, "items": ["rent", "food", "utilities"] },
    "wants": { "amount": 0, "percentage": 0, "items": ["entertainment", "dining out"] },
    "savings": { "amount": 0, "percentage": 0 },
    "investments": { "amount": 0, "percentage": 0 }
  },
  "investment_suggestions": [
    {
      "type": "Investment type",
      "suggested_amount": 0,
      "expected_return": "X% per year",
      "risk_level": "low/medium/high",
      "description": "Why this suits them"
    }
  ],
  "final_narrative": "Full markdown response to the user's question with actionable steps",
  "dashboard_data": {
    "income_breakdown": [
      { "name": "Net Salary", "value": 0 },
      { "name": "Tax", "value": 0 },
      { "name": "PF/EPF", "value": 0 },
      { "name": "Other Deductions", "value": 0 }
    ],
    "expense_allocation": [
      { "name": "Needs", "value": 0 },
      { "name": "Wants", "value": 0 },
      { "name": "Savings", "value": 0 },
      { "name": "Investments", "value": 0 }
    ],
    "savings_projection": [
      { "month": "Month 1", "savings": 0 },
      { "month": "Month 3", "savings": 0 },
      { "month": "Month 6", "savings": 0 },
      { "month": "Month 12", "savings": 0 }
    ]
  }
}"""

    salary_info = state.get("salary_data", {})
    user_content = f"""
User Question: {state.get('user_question', '')}

Salary Information:
- Gross Monthly Salary: ₹{salary_info.get('gross_salary', salary_info.get('monthly_income', 0)):,}
- Net Monthly Salary: ₹{salary_info.get('net_salary', 0):,}
- Tax Deducted: ₹{salary_info.get('tax', salary_info.get('tds', 0)):,}
- PF/EPF: ₹{salary_info.get('pf', salary_info.get('epf', 0)):,}
- Other Deductions: ₹{salary_info.get('other_deductions', 0):,}
- Monthly Expenses (self-reported): ₹{salary_info.get('monthly_expenses', 0):,}
- Current Savings: ₹{salary_info.get('current_savings', salary_info.get('savings', 0)):,}
- Additional info: {json.dumps({k: v for k, v in salary_info.items() 
                               if k not in ['gross_salary','net_salary','tax','tds','pf','epf',
                                            'other_deductions','monthly_expenses','current_savings','savings']}, indent=2)}

Previous conversation:
{json.dumps(state.get('conversation_history', []), indent=2)}
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content)
    ])

    parsed = _parse_json_safe(response.content)

    return {
        "emergency_fund_analysis": parsed.get("emergency_fund_analysis", {}),
        "savings_plan": parsed.get("savings_plan", {}),
        "investment_suggestions": parsed.get("investment_suggestions", []),
        "budget_breakdown": parsed.get("budget_breakdown", {}),
        "final_response": parsed.get("final_narrative", response.content),
        "dashboard_data": parsed.get("dashboard_data", {}),
        "current_step": "advisory_complete",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Build the LangGraph pipeline (single-agent for employee analysis)
# ─────────────────────────────────────────────────────────────────────────────
def build_employee_analysis_graph() -> StateGraph:
    graph = StateGraph(EmployeeFinancialState)
    graph.add_node("advisor", employee_advisor_agent)
    graph.set_entry_point("advisor")
    graph.add_edge("advisor", END)
    return graph.compile()


employee_analysis_graph = build_employee_analysis_graph()
