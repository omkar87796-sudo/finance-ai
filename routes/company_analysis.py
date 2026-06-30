"""
routes/company_analysis.py

REST API endpoints for company financial analysis.

POST /api/company/analyze     — Start or continue a company analysis session
POST /api/company/provide-data — User provides additional data after being asked
"""

import json
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agents.company_agents import company_analysis_graph
from agents.state import CompanyAnalysisState

router = APIRouter()


# ── Request / Response models ─────────────────────────────────────────────────
class CompanyAnalysisRequest(BaseModel):
    user_question: str
    company_name: str
    company_data: Dict[str, Any] = {}
    conversation_history: List[Dict[str, str]] = []


class AdditionalDataRequest(BaseModel):
    """Sent when user provides extra data after the DataCollectorAgent asked for it."""
    user_question: str
    company_name: str
    original_data: Dict[str, Any] = {}
    additional_data: Dict[str, Any] = {}
    conversation_history: List[Dict[str, str]] = []


# ── Endpoints ─────────────────────────────────────────────────────────────────
@router.post("/analyze")
async def analyze_company(request: CompanyAnalysisRequest):
    """
    Main endpoint. Runs the 5-agent pipeline.
    If the DataCollectorAgent decides more data is needed, this returns a
    `needs_more_data: true` response with a follow-up question.
    The frontend must then call /provide-data with the extra information.
    """
    initial_state: CompanyAnalysisState = {
        "user_question": request.user_question,
        "company_name": request.company_name,
        "company_data": request.company_data,
        "conversation_history": request.conversation_history,
        # Outputs — will be filled by agents
        "collected_data": {},
        "needs_more_data": False,
        "data_request_message": "",
        "research_findings": {},
        "analysis_report": {},
        "decisions": [],
        "predictions": {},
        "future_actions": [],
        "final_response": "",
        "current_step": "starting",
        "error": None,
    }

    try:
        result = await company_analysis_graph.ainvoke(initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent pipeline error: {str(exc)}")

    # If data collector wants more info — pause and ask user
    if result.get("needs_more_data"):
        return {
            "status": "needs_more_data",
            "message": result["data_request_message"],
            "current_step": result.get("current_step"),
        }

    # Full analysis complete
    return {
        "status": "complete",
        "company_name": request.company_name,
        "current_step": result.get("current_step"),
        "analysis_report": result.get("analysis_report", {}),
        "decisions": result.get("decisions", []),
        "predictions": result.get("predictions", {}),
        "future_actions": result.get("future_actions", []),
        "research_findings": result.get("research_findings", {}),
        "final_response": result.get("final_response", ""),
        # Dashboard-ready structures
        "dashboard_data": _build_company_dashboard_data(result),
    }


@router.post("/provide-data")
async def provide_additional_data(request: AdditionalDataRequest):
    """
    Called when the user provides extra data requested by DataCollectorAgent.
    Merges old + new data and re-runs the full pipeline.
    """
    merged_data = {**request.original_data, **request.additional_data}
    updated_history = request.conversation_history + [
        {"role": "assistant", "content": "Please provide additional data."},
        {"role": "user", "content": f"Additional data: {json.dumps(request.additional_data)}"},
    ]

    # Re-use the main endpoint logic with merged data
    full_request = CompanyAnalysisRequest(
        user_question=request.user_question,
        company_name=request.company_name,
        company_data=merged_data,
        conversation_history=updated_history,
    )
    return await analyze_company(full_request)


# ── Helper: build dashboard-ready data from agent results ────────────────────
def _build_company_dashboard_data(result: Dict) -> Dict:
    """
    Converts raw agent output into structures ready for Recharts on the frontend.
    """
    report = result.get("analysis_report", {})
    decisions = result.get("decisions", [])
    predictions = result.get("predictions", {})

    # Problem severity radar chart data
    problem_areas = report.get("problem_areas", {})
    severity_map = {"low": 30, "medium": 60, "high": 90, "critical": 100}
    radar_data = [
        {
            "area": area.title(),
            "severity": severity_map.get(info.get("severity", "low"), 30),
        }
        for area, info in problem_areas.items()
        if isinstance(info, dict)
    ]

    # Decision timeline bar chart data
    timeline_map = {"Immediate": 0, "30 days": 30, "90 days": 90, "6 months": 180}
    timeline_data = [
        {
            "decision": d.get("decision", "")[:40] + "…" if len(d.get("decision","")) > 40 else d.get("decision",""),
            "days": timeline_map.get(d.get("timeline", "90 days"), 90),
            "area": d.get("area", ""),
        }
        for d in (decisions[:6] if isinstance(decisions, list) else [])
    ]

    # Prediction comparison
    without = predictions.get("without_action", {})
    with_action = predictions.get("with_action", {})
    prediction_data = [
        {
            "period": "3 Months",
            "without_action": _pct_to_num(without.get("projected_revenue_change", "-10%")),
            "with_action": _pct_to_num(with_action.get("projected_revenue_change", "+15%")),
        },
        {
            "period": "6 Months",
            "without_action": _pct_to_num(without.get("projected_revenue_change", "-20%")) * 1.5,
            "with_action": _pct_to_num(with_action.get("projected_revenue_change", "+15%")) * 1.8,
        },
        {
            "period": "12 Months",
            "without_action": _pct_to_num(without.get("projected_revenue_change", "-20%")) * 2.5,
            "with_action": _pct_to_num(with_action.get("projected_revenue_change", "+15%")) * 3.2,
        },
    ]

    return {
        "problem_severity_radar": radar_data,
        "decision_timeline": timeline_data,
        "revenue_prediction": prediction_data,
        "strengths": report.get("strengths", []),
        "weaknesses": report.get("weaknesses", []),
        "root_causes": report.get("root_causes", []),
        "key_metrics": report.get("key_metrics", {}),
    }


def _pct_to_num(pct_str: str) -> float:
    """Convert '+15%' or '-10%' to float 15 or -10."""
    try:
        return float(pct_str.replace("%", "").replace("+", "").strip())
    except Exception:
        return 0.0
