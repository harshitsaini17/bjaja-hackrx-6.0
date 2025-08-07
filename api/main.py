"""
FastAPI Application for LLM Document Processing System - Vercel Deployment
Provides REST API endpoints for processing natural language queries against documents
Handles policy documents, contracts, and emails with semantic understanding
"""

import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, List, Optional, Any
import logging
import time
from datetime import datetime

# Add parent directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir))

# Import our modules with error handling
try:
    from general_document_processor import GeneralDocumentProcessor
except ImportError as e:
    # Fallback import
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "general_document_processor", 
            os.path.join(current_dir, "general_document_processor.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        GeneralDocumentProcessor = module.GeneralDocumentProcessor
    except Exception as fallback_error:
        GeneralDocumentProcessor = None
        import_error = f"Failed to import GeneralDocumentProcessor: {str(e)}, Fallback: {str(fallback_error)}"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LLM Document Processing System API",
    description="General-purpose document processing for policies, contracts, and emails",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the document processor (with error handling for Vercel)
document_processor = None
processor_error = None

try:
    if GeneralDocumentProcessor is not None:
        document_processor = GeneralDocumentProcessor()
        logger.info("Document processor initialized successfully")
    else:
        processor_error = import_error if 'import_error' in locals() else "GeneralDocumentProcessor not available"
        logger.error(f"Document processor not available: {processor_error}")
except Exception as e:
    logger.error(f"Failed to initialize document processor: {str(e)}")
    processor_error = str(e)

# Pydantic models for request/response
class DocumentQueryRequest(BaseModel):
    """Request model for document processing"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "46-year-old male, knee surgery in Pune, 3-month-old insurance policy",
                "query_id": "QUERY_2024_001"
            }
        }
    )
    
    query: str = Field(..., description="Natural language query about documents", min_length=5)
    query_id: Optional[str] = Field(None, description="Optional external query ID")

class DocumentQueryResponse(BaseModel):
    """Response model for document processing - matches problem statement"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "decision": "approved",
                "amount": 150000.0,
                "justification": "Knee surgery is covered under surgical procedures section.",
                "clause_references": ["SEC_4_2", "WAIT_2_1", "GEOG_3_1"],
                "confidence": 0.92,
                "processing_time_seconds": 8.5,
                "query_id": "doc_query_1754568526",
                "timestamp": "2025-08-07T10:30:00Z",
                "clause_details": [
                    {
                        "clause_id": "CLAUSE_1",
                        "original_text": "Surgical procedures including orthopedic surgeries are covered...",
                        "source_document": "policy.pdf",
                        "source_page": 15
                    }
                ],
                "metadata": {
                    "chunks_analyzed": 5,
                    "model_used": "gemini-2.5-flash",
                    "query_type": "claim_processing"
                }
            }
        }
    )
    
    decision: str = Field(..., description="Decision: approved, rejected, conditional, or review_required")
    amount: Optional[float] = Field(None, description="Amount if applicable (claims, payouts, limits)")
    justification: str = Field(..., description="Explanation with specific clause references")
    clause_references: List[str] = Field(default=[], description="List of specific clause IDs used")
    confidence: float = Field(..., description="Confidence score 0.0-1.0")
    processing_time_seconds: float = Field(..., description="Time taken to process")
    query_id: str = Field(..., description="Unique query identifier")
    timestamp: str = Field(..., description="ISO timestamp")
    clause_details: List[Dict[str, Any]] = Field(default=[], description="Original text and details of referenced clauses")
    metadata: Dict[str, Any] = Field(default={}, description="Additional processing metadata")

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str
    processor_ready: bool
    version: str
    environment: str
    processor_error: Optional[str] = None

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    message: str
    timestamp: str
    request_id: Optional[str] = None

# In-memory storage for demo purposes
processed_queries: Dict[str, DocumentQueryResponse] = {}

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "LLM Document Processing System API",
        "description": "Process natural language queries against policy documents, contracts, and emails",
        "version": "1.0.0",
        "environment": "Vercel",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if document_processor is not None else "degraded",
        timestamp=datetime.now().isoformat(),
        processor_ready=document_processor is not None,
        version="1.0.0",
        environment="Vercel",
        processor_error=processor_error
    )

@app.post("/process-query", response_model=DocumentQueryResponse)
async def process_document_query(request: DocumentQueryRequest):
    """
    Process a natural language query against documents
    
    - **query**: Natural language description of what you want to find
    - **query_id**: Optional external query identifier
    
    Returns structured decision with justification and clause references
    """
    if document_processor is None:
        raise HTTPException(
            status_code=503,
            detail=f"Document processor not available: {processor_error}"
        )
    
    try:
        start_time = time.time()
        
        # Generate query ID if not provided
        query_id = request.query_id or f"API_QUERY_{int(time.time())}"
        
        # Process the query
        logger.info(f"Processing document query {query_id}: {request.query}")
        decision = document_processor.process_query(request.query)
        
        processing_time = time.time() - start_time
        
        # Create response matching problem statement format with clause details
        clause_details = decision.processing_metadata.get("clause_details", [])
        
        response = DocumentQueryResponse(
            decision=decision.decision,
            amount=decision.amount,
            justification=decision.justification,
            clause_references=decision.clause_references or [],
            confidence=decision.confidence,
            processing_time_seconds=round(processing_time, 2),
            query_id=decision.query_id,
            timestamp=datetime.now().isoformat(),
            clause_details=clause_details,  # Include original text and details
            metadata=decision.processing_metadata or {}
        )
        
        # Store for later retrieval
        processed_queries[query_id] = response
        
        logger.info(f"Query {query_id} processed: {decision.decision} with confidence {decision.confidence}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing query {request.query_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/query/{query_id}", response_model=DocumentQueryResponse)
async def get_query_result(query_id: str):
    """
    Retrieve a previously processed query by ID
    
    - **query_id**: The query identifier
    """
    if query_id not in processed_queries:
        raise HTTPException(
            status_code=404,
            detail=f"Query {query_id} not found"
        )
    
    return processed_queries[query_id]

@app.get("/queries", response_model=List[DocumentQueryResponse])
async def list_queries(limit: int = 50, offset: int = 0):
    """
    List all processed queries with pagination
    
    - **limit**: Maximum number of queries to return (default: 50)
    - **offset**: Number of queries to skip (default: 0)
    """
    queries_list = list(processed_queries.values())
    
    # Sort by timestamp (newest first)
    queries_list.sort(key=lambda x: x.timestamp, reverse=True)
    
    # Apply pagination
    paginated_queries = queries_list[offset:offset + limit]
    
    return paginated_queries

@app.get("/stats", response_model=Dict[str, Any])
async def get_statistics():
    """
    Get processing statistics
    """
    if not processed_queries:
        return {
            "total_queries": 0,
            "approved_queries": 0,
            "rejected_queries": 0,
            "conditional_queries": 0,
            "review_required_queries": 0,
            "approval_rate": 0.0,
            "average_confidence": 0.0,
            "average_processing_time": 0.0,
            "environment": "Vercel"
        }
    
    queries_list = list(processed_queries.values())
    total_queries = len(queries_list)
    
    approved_queries = sum(1 for query in queries_list if query.decision == "approved")
    rejected_queries = sum(1 for query in queries_list if query.decision == "rejected")
    conditional_queries = sum(1 for query in queries_list if query.decision == "conditional")
    review_required_queries = sum(1 for query in queries_list if query.decision == "review_required")
    
    approval_rate = approved_queries / total_queries if total_queries > 0 else 0.0
    average_confidence = sum(query.confidence for query in queries_list) / total_queries
    average_processing_time = sum(query.processing_time_seconds for query in queries_list) / total_queries
    
    return {
        "total_queries": total_queries,
        "approved_queries": approved_queries,
        "rejected_queries": rejected_queries,
        "conditional_queries": conditional_queries,
        "review_required_queries": review_required_queries,
        "approval_rate": round(approval_rate, 3),
        "average_confidence": round(average_confidence, 3),
        "average_processing_time": round(average_processing_time, 2),
        "environment": "Vercel"
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return ErrorResponse(
        error=f"HTTP {exc.status_code}",
        message=exc.detail,
        timestamp=datetime.now().isoformat()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return ErrorResponse(
        error="Internal Server Error",
        message="An unexpected error occurred",
        timestamp=datetime.now().isoformat()
    )

# Vercel serverless function handler
def handler(request, response):
    """Vercel serverless function handler"""
    return app(request, response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
