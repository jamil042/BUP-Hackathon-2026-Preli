from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

app = FastAPI(title="GridWise LLM")

@app.exception_handler(RequestValidationError)
async def validation_handler(request, exc):
    return JSONResponse(status_code=400, content={"error": "malformed_or_invalid_request", "detail": exc.errors()})

@app.exception_handler(Exception)
async def unhandled_handler(request, exc):
    return JSONResponse(status_code=500, content={"error": "internal_error"})

@app.get("/health")
async def health():
    return {"status": "ok"}
