"""
Finance AI - Main Application Entry Point
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from routes.company_analysis import router as company_router
from routes.employee_analysis import router as employee_router
from routes.dashboard import router as dashboard_router

load_dotenv()

app = FastAPI(
    title="Finance AI",
    description="Agentic AI platform for financial analysis and decision making",
    version="1.0.0"
)

# CORS Middleware - allows React frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "http://localhost:3000"), "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all route modules
app.include_router(company_router,  prefix="/api/company",  tags=["Company Analysis"])
app.include_router(employee_router, prefix="/api/employee", tags=["Employee Analysis"])
app.include_router(dashboard_router, prefix="/api/dashboard", tags=["Dashboard"])


@app.get("/")
async def root():
    return {
        "message": "Finance AI Backend is running",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
