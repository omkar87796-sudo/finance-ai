"""
routes/company_analysis.py
"""

import json
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agents.company_agents import company_analysis_graph, build_skip_collection_graph
from agents.state import CompanyAnalysisState

router = APIRouter()


class CompanyAnalysisRequest(BaseModel):
    user_question: str
    company_name: str
    company_data: Dict[str, Any] = {}
    conversation_history: List[Dict[str, str]] = []


class AdditionalDataRequest(BaseModel):
    user_question: str
    company_name: str
    original_data: Dict[str, Any] = {}
    additional_data: Dict[str, Any] = {}
    conversation_history: List[Dict[str, str]] = []
    skip_collection: bool = False


@router.post("/analyze")
async def analyze_company(request: CompanyAnalysisRequest):
    initial_state: CompanyAnalysisState = {
        "user_question": request.user_question,
        "company_name": request.company_name,
        "company_data": request.company_data,
        "conversation_history": request.conversation_history,
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

    if result.get("needs_more_data"):
        return {
            "status": "needs_more_data",
            "message": result["data_request_message"],
            "current_step": result.get("current_step"),
        }

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
        "dashboard_data": _build_company_dashboard_data(result),
    }


@router.post("/provide-data")
async def provide_additional_data(request: AdditionalDataRequest):
    merged_data = {**request.original_data, **request.additional_data}

    # If user skipped — force pipeline to skip data collection entirely
    # by using a graph that starts directly from research
    skip_collection_graph = build_skip_collection_graph()

    initial_state: CompanyAnalysisState = {
        "user_question": request.user_question,
        "company_name": request.company_name,
        "company_data": merged_data,
        "conversation_history": request.conversation_history,
        "collected_data": merged_data,
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
        result = await skip_collection_graph.ainvoke(initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent pipeline error: {str(exc)}")

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
        "dashboard_data": _build_company_dashboard_data(result),
    }


def _build_company_dashboard_data(result: Dict) -> Dict:
    report = result.get("analysis_report", {})
    decisions = result.get("decisions", [])
    predictions = result.get("predictions", {})

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

    timeline_map = {"Immediate": 0, "30 days": 30, "90 days": 90, "6 months": 180}
    timeline_data = [
        {
            "decision": d.get("decision", "")[:40] + "…" if len(d.get("decision","")) > 40 else d.get("decision",""),
            "days": timeline_map.get(d.get("timeline", "90 days"), 90),
            "area": d.get("area", ""),
        }
        for d in (decisions[:6] if isinstance(decisions, list) else [])
    ]

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
    try:
        return float(pct_str.replace("%", "").replace("+", "").strip())
    except Exception:
        return 0.0
