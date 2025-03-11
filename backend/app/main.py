from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import time

from app.api.routes import router as api_router
from app.core.config import APP_NAME, APP_VERSION, API_PREFIX, CORS_ORIGINS, DEBUG
from app.core.logging_config import configure_logging

# Configure logging using our custom configuration
logger = configure_logging()
logger.info(f"Starting {APP_NAME} v{APP_VERSION}")

# Request ID middleware
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Add request_id to all log records during this request
        old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.request_id = request_id
            return record
            
        logging.setLogRecordFactory(record_factory)
        
        # Measure request duration
        start_time = time.time()
        
        # Process the request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log request details
            logger.info(
                f"Request completed: {request.method} {request.url.path}",
                extra={
                    "duration": round(duration * 1000, 2),  # Convert to ms
                    "status_code": response.status_code,
                    "client_host": request.client.host if request.client else None,
                    "method": request.method,
                    "path": request.url.path
                }
            )
            
            # Add request_id to response headers
            response.headers["X-Request-ID"] = request_id
            return response
            
        except Exception as e:
            # Log exception
            duration = time.time() - start_time
            logger.exception(
                f"Request failed: {request.method} {request.url.path}",
                extra={
                    "duration": round(duration * 1000, 2),
                    "client_host": request.client.host if request.client else None,
                    "method": request.method,
                    "path": request.url.path
                }
            )
            raise

app = FastAPI(
    title=APP_NAME,
    description="A demo application showcasing natural language to SQL conversion using Gemini and BigQuery",
    version=APP_VERSION,
    debug=DEBUG
)

# Add request ID middleware
app.add_middleware(RequestIDMiddleware)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix=API_PREFIX)

# Mount static files (if needed)
# app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    """Root endpoint that redirects to API documentation"""
    return {
        "message": "Welcome to the NL2SQL Agent Demo API",
        "documentation": "/docs",
        "version": APP_VERSION
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=DEBUG) 