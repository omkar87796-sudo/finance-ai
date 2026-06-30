"""
routes/employee_analysis.py

REST API endpoints for employee financial advisory.

POST /api/employee/upload-salary   — Upload salary slip (PDF/image), get parsed data
POST /api/employee/ask             — Ask financial question using parsed salary data
"""

import os
import shutil
from typing import Dict, Any, List

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from agents.employee_agents import employee_analysis_graph
from agents.state import EmployeeFinancialState
from utils.salary_parser import parse_salary_file

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}


# ── Request models ────────────────────────────────────────────────────────────
class EmployeeQuestionRequest(BaseModel):
    user_question: str
    salary_data: Dict[str, Any]
    conversation_history: List[Dict[str, str]] = []


# ── Endpoints ─────────────────────────────────────────────────────────────────
@router.post("/upload-salary")
async def upload_salary_slip(file: UploadFile = File(...)):
    """
    Upload a salary slip PDF or image.
    Returns the extracted structured salary data, which the frontend should
    store and pass back in subsequent /ask calls.
    """
    # Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Please upload PDF, PNG, or JPG."
        )

    # Save uploaded file temporarily
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Parse salary data
    try:
        salary_data = parse_salary_file(save_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse salary slip: {str(exc)}")
    finally:
        # Always clean up the temporary file
        if os.path.exists(save_path):
            os.remove(save_path)

    if "error" in salary_data:
        raise HTTPException(status_code=422, detail=salary_data["error"])

    return {
        "status": "success",
        "message": "Salary slip parsed successfully",
        "salary_data": salary_data,
        "summary": {
            "employee_name": salary_data.get("employee_name", "N/A"),
            "company": salary_data.get("company_name", "N/A"),
            "month_year": salary_data.get("month_year", "N/A"),
            "gross_salary": salary_data.get("gross_salary", 0),
            "net_salary": salary_data.get("net_salary", 0),
            "total_deductions": salary_data.get("total_deductions", 0),
        }
    }


@router.post("/ask")
async def ask_financial_question(request: EmployeeQuestionRequest):
    """
    Ask a personal finance question using previously parsed salary data.
    The frontend sends salary_data (from /upload-salary) + the question.
    Returns AI advice + dashboard-ready chart data.
    """
    salary = request.salary_data

    initial_state: EmployeeFinancialState = {
        "user_question": request.user_question,
        "salary_data": salary,
        "conversation_history": request.conversation_history,
        # Derived values (pre-compute for the agent prompt)
        "monthly_income": salary.get("gross_salary", salary.get("monthly_income", 0)),
        "monthly_expenses": salary.get("monthly_expenses", 0),
        "savings": salary.get("current_savings", salary.get("savings", 0)),
        "tax_deductions": salary.get("tds", salary.get("tax", 0)),
        "other_deductions": salary.get("other_deductions", 0),
        "net_salary": salary.get("net_salary", 0),
        # Outputs
        "emergency_fund_analysis": {},
        "savings_plan": {},
        "investment_suggestions": [],
        "budget_breakdown": {},
        "final_response": "",
        "dashboard_data": {},
        "current_step": "starting",
        "error": None,
    }

    try:
        result = await employee_analysis_graph.ainvoke(initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Advisory agent error: {str(exc)}")

    return {
        "status": "complete",
        "answer": result.get("final_response", ""),
        "emergency_fund_analysis": result.get("emergency_fund_analysis", {}),
        "savings_plan": result.get("savings_plan", {}),
        "investment_suggestions": result.get("investment_suggestions", []),
        "budget_breakdown": result.get("budget_breakdown", {}),
        "dashboard_data": result.get("dashboard_data", {}),
        "current_step": result.get("current_step"),
    }


@router.post("/manual-salary")
async def manual_salary_entry(
    gross_salary: float = Form(...),
    net_salary: float = Form(...),
    tax: float = Form(0),
    pf: float = Form(0),
    other_deductions: float = Form(0),
    monthly_expenses: float = Form(0),
    current_savings: float = Form(0),
    employee_name: str = Form(""),
):
    """
    Alternative to file upload — user can manually enter salary details.
    Useful when the user doesn't have a salary slip file handy.
    """
    salary_data = {
        "employee_name": employee_name,
        "gross_salary": gross_salary,
        "net_salary": net_salary,
        "tax": tax,
        "tds": tax,
        "pf": pf,
        "epf": pf,
        "other_deductions": other_deductions,
        "total_deductions": tax + pf + other_deductions,
        "monthly_expenses": monthly_expenses,
        "current_savings": current_savings,
        "currency": "INR",
    }

    return {
        "status": "success",
        "message": "Salary data recorded manually",
        "salary_data": salary_data,
    }
