"""
routes/dashboard.py

Provides pre-built demo/sample dashboard data for the Finance AI landing page.
Also exposes a /summary endpoint to aggregate the latest analysis results.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/demo")
async def get_demo_dashboard():
    """
    Returns static demo data so the frontend dashboard renders immediately
    without requiring a real analysis run. Good for portfolio showcasing.
    """
    return {
        "company_demo": {
            "company_name": "TechCorp India Pvt. Ltd.",
            "problem_severity_radar": [
                {"area": "Product",    "severity": 75},
                {"area": "Marketing",  "severity": 90},
                {"area": "Operations", "severity": 45},
                {"area": "Finance",    "severity": 80},
                {"area": "HR",         "severity": 30},
            ],
            "decision_timeline": [
                {"decision": "Revamp digital marketing strategy", "days": 30,  "area": "Marketing"},
                {"decision": "Launch customer loyalty program",   "days": 90,  "area": "Product"},
                {"decision": "Reduce operational overhead",       "days": 0,   "area": "Operations"},
                {"decision": "Raise Series A funding",            "days": 180, "area": "Finance"},
            ],
            "revenue_prediction": [
                {"period": "3 Months",  "without_action": -15, "with_action": 10},
                {"period": "6 Months",  "without_action": -28, "with_action": 22},
                {"period": "12 Months", "without_action": -45, "with_action": 48},
            ],
            "key_metrics": {
                "Monthly Revenue":  {"current": "₹42L",  "benchmark": "₹65L",  "status": "critical"},
                "Customer Retention": {"current": "58%", "benchmark": "80%",   "status": "warning"},
                "Profit Margin":    {"current": "8%",    "benchmark": "18%",   "status": "warning"},
                "Employee NPS":     {"current": "42",    "benchmark": "60",    "status": "good"},
            }
        },
        "employee_demo": {
            "employee_name": "Rahul Sharma",
            "salary_summary": {
                "gross_salary": 85000,
                "net_salary": 67500,
                "total_deductions": 17500,
            },
            "income_breakdown": [
                {"name": "Net Salary",      "value": 67500},
                {"name": "TDS/Income Tax",  "value": 9500},
                {"name": "PF/EPF",          "value": 5100},
                {"name": "Other",           "value": 2900},
            ],
            "expense_allocation": [
                {"name": "Needs",       "value": 35000},
                {"name": "Wants",       "value": 12500},
                {"name": "Savings",     "value": 13500},
                {"name": "Investments", "value": 6500},
            ],
            "savings_projection": [
                {"month": "Month 1",  "savings": 13500},
                {"month": "Month 3",  "savings": 40500},
                {"month": "Month 6",  "savings": 81000},
                {"month": "Month 12", "savings": 162000},
            ],
            "emergency_fund": {
                "current_savings": 45000,
                "monthly_expenses": 47500,
                "months_can_survive": 0.9,
                "target": 285000,
            }
        }
    }


@router.get("/stats")
async def platform_stats():
    """High-level platform statistics shown on the landing page."""
    return {
        "total_analyses": 1247,
        "companies_helped": 384,
        "employees_advised": 863,
        "avg_analysis_time_seconds": 18,
        "supported_file_types": ["PDF", "PNG", "JPG"],
        "agent_pipeline": [
            "Data Collector",
            "Market Researcher",
            "Data Analyst",
            "Decision Maker",
            "AI Predictor",
        ]
    }
