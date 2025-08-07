"""
FastAPI Application for ICICI Lombard Insurance Claim Processing
Provides REST API endpoints for processing insurance claims with LLM-enhanced decision making
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
import time
import uuid
from datetime import datetime

from llm_enhanced_processor import LLMEnhancedClaimProcessor, DecisionType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ICICI Lombard Claim Processor API",
    description="LLM-Enhanced Insurance Claim Processing System",
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

# Initialize the claim processor
try:
    claim_processor = LLMEnhancedClaimProcessor()
    logger.info("Claim processor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize claim processor: {str(e)}")
    claim_processor = None

# Pydantic models for request/response
class ClaimRequest(BaseModel):
    """Request model for claim processing"""
    query: str = Field(..., description="Natural language claim query", min_length=10)
    claim_id: Optional[str] = Field(None, description="Optional external claim ID")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "45F, breast cancer, chemotherapy treatment, oncology, Delhi, Sum Insured 15 lakhs, 2-year policy",
                "claim_id": "CLAIM_2024_001"
            }
        }

class ClaimResponse(BaseModel):
    """Response model for claim processing"""
    claim_id: str
    decision: str
    confidence: float
    primary_reason: str
    processing_time_seconds: float
    query_id: str
    timestamp: str
    metadata: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "claim_id": "CLAIM_2024_001",
                "decision": "APPROVED",
                "confidence": 0.95,
                "primary_reason": "Chemotherapy treatment for breast cancer is covered as a modern treatment/therapeutic procedure.",
                "processing_time_seconds": 21.92,
                "query_id": "llm_enhanced_claim_1754568526",
                "timestamp": "2025-08-07T10:30:00Z",
                "metadata": {
                    "chunks_analyzed": 6,
                    "model_used": "gemini-2.5-flash",
                    "enhancement_version": "3.0_LLM_Enhanced"
                }
            }
        }

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str
    processor_ready: bool
    version: str

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    message: str
    timestamp: str
    request_id: Optional[str] = None

# In-memory storage for demo purposes (use database in production)
processed_claims: Dict[str, ClaimResponse] = {}

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "ICICI Lombard Claim Processor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        processor_ready=claim_processor is not None,
        version="1.0.0"
    )

@app.post("/process-claim", response_model=ClaimResponse)
async def process_claim(request: ClaimRequest):
    """
    Process an insurance claim using LLM-enhanced analysis
    
    - **query**: Natural language description of the claim
    - **claim_id**: Optional external claim identifier
    
    Returns decision (APPROVED/REJECTED) with reasoning and confidence score
    """
    if claim_processor is None:
        raise HTTPException(
            status_code=503,
            detail="Claim processor not available. Please try again later."
        )
    
    try:
        start_time = time.time()
        
        # Generate claim ID if not provided
        claim_id = request.claim_id or f"API_CLAIM_{int(time.time())}"
        
        # Process the claim
        logger.info(f"Processing claim {claim_id}: {request.query}")
        decision = claim_processor.process_claim(request.query)
        
        processing_time = time.time() - start_time
        
        # Create response
        response = ClaimResponse(
            claim_id=claim_id,
            decision=decision.decision.value,
            confidence=decision.confidence,
            primary_reason=decision.justification.primary_reason,
            processing_time_seconds=round(processing_time, 2),
            query_id=decision.query_id,
            timestamp=datetime.now().isoformat(),
            metadata=decision.processing_metadata
        )
        
        # Store for later retrieval
        processed_claims[claim_id] = response
        
        logger.info(f"Claim {claim_id} processed: {decision.decision.value} with confidence {decision.confidence}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing claim {request.claim_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing claim: {str(e)}"
        )

@app.get("/claim/{claim_id}", response_model=ClaimResponse)
async def get_claim(claim_id: str):
    """
    Retrieve a previously processed claim by ID
    
    - **claim_id**: The claim identifier
    """
    if claim_id not in processed_claims:
        raise HTTPException(
            status_code=404,
            detail=f"Claim {claim_id} not found"
        )
    
    return processed_claims[claim_id]

@app.get("/claims", response_model=List[ClaimResponse])
async def list_claims(limit: int = 50, offset: int = 0):
    """
    List all processed claims with pagination
    
    - **limit**: Maximum number of claims to return (default: 50)
    - **offset**: Number of claims to skip (default: 0)
    """
    claims_list = list(processed_claims.values())
    total_claims = len(claims_list)
    
    # Sort by timestamp (newest first)
    claims_list.sort(key=lambda x: x.timestamp, reverse=True)
    
    # Apply pagination
    paginated_claims = claims_list[offset:offset + limit]
    
    return paginated_claims

@app.get("/stats", response_model=Dict[str, Any])
async def get_statistics():
    """
    Get processing statistics
    """
    if not processed_claims:
        return {
            "total_claims": 0,
            "approved_claims": 0,
            "rejected_claims": 0,
            "approval_rate": 0.0,
            "average_confidence": 0.0,
            "average_processing_time": 0.0
        }
    
    claims_list = list(processed_claims.values())
    total_claims = len(claims_list)
    
    approved_claims = sum(1 for claim in claims_list if claim.decision == "APPROVED")
    rejected_claims = total_claims - approved_claims
    
    approval_rate = approved_claims / total_claims if total_claims > 0 else 0.0
    average_confidence = sum(claim.confidence for claim in claims_list) / total_claims
    average_processing_time = sum(claim.processing_time_seconds for claim in claims_list) / total_claims
    
    return {
        "total_claims": total_claims,
        "approved_claims": approved_claims,
        "rejected_claims": rejected_claims,
        "approval_rate": round(approval_rate, 3),
        "average_confidence": round(average_confidence, 3),
        "average_processing_time": round(average_processing_time, 2)
    }

@app.delete("/claims")
async def clear_claims():
    """
    Clear all processed claims (for demo/testing purposes)
    """
    global processed_claims
    claim_count = len(processed_claims)
    processed_claims.clear()
    
    return {
        "message": f"Cleared {claim_count} claims",
        "timestamp": datetime.now().isoformat()
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
