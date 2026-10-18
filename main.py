"""Entry point:  python -m uvicorn main:app --reload"""
from app.web import create_app

app = create_app()
