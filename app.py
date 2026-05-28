"""Minimal FastAPI app for evaluation pipeline reproduction."""
from fastapi import FastAPI
from api.routers import diagnostics, evaluation, trace

app = FastAPI(title="Cross-Layer Provenance Eval Pipeline")
app.include_router(diagnostics.router)
app.include_router(evaluation.router)
app.include_router(trace.router)
