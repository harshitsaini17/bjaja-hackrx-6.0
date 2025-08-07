"""
Lightweight Document Processor for Vercel Deployment
Simplified version that works with minimal dependencies
"""

import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import time
import uuid
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentDecision:
    """Decision result from document processing"""
    decision: str
    amount: Optional[float] = None
    justification: str = ""
    clause_references: List[str] = field(default_factory=list)
    confidence: float = 0.0
    query_id: str = ""
    processing_metadata: Dict[str, Any] = field(default_factory=dict)

class LightweightDocumentProcessor:
    """
    Lightweight document processor for Vercel deployment
    Uses only Google Gemini without heavy ML dependencies
    """
    
    def __init__(self):
        """Initialize the processor with minimal dependencies"""
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        # Initialize Google Gemini
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.google_api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            logger.info("Google Gemini initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google Gemini: {str(e)}")
            raise
        
        # Sample document context for demo
        self.document_context = self._load_sample_context()
    
    def _load_sample_context(self) -> str:
        """Load sample document context for processing"""
        return """
        SAMPLE INSURANCE POLICY CONTEXT:
        
        Coverage Details:
        - Surgical procedures including orthopedic surgeries are covered up to ₹2,00,000
        - Pre-existing conditions covered after 2-year waiting period
        - Hospitalization covered in network hospitals
        - Geographic coverage: India (excluding certain restricted areas)
        
        Waiting Periods:
        - General waiting period: 30 days
        - Pre-existing conditions: 2 years (24 months)
        - Specific surgeries: 1 year
        
        Exclusions:
        - Cosmetic surgeries
        - Experimental treatments
        - Self-inflicted injuries
        - War and terrorism related injuries
        
        Claim Procedures:
        - Pre-authorization required for planned surgeries
        - Cashless facility available at network hospitals
        - Reimbursement claims to be submitted within 30 days
        """
    
    def process_query(self, query: str) -> DocumentDecision:
        """
        Process a document query using Google Gemini
        
        Args:
            query: Natural language query about documents
            
        Returns:
            DocumentDecision with analysis results
        """
        try:
            start_time = time.time()
            query_id = f"LITE_QUERY_{int(time.time())}"
            
            # Create comprehensive prompt
            prompt = self._create_analysis_prompt(query)
            
            # Get response from Gemini
            response = self.model.generate_content(prompt)
            
            # Parse the response
            decision = self._parse_gemini_response(response.text, query_id)
            
            processing_time = time.time() - start_time
            
            # Add processing metadata
            decision.processing_metadata.update({
                "processing_time_seconds": round(processing_time, 2),
                "model_used": "gemini-2.0-flash-exp",
                "query_type": "lightweight_processing",
                "chunks_analyzed": 1,
                "clause_details": [
                    {
                        "clause_id": "SAMPLE_CLAUSE_1",
                        "original_text": "Surgical procedures including orthopedic surgeries are covered up to ₹2,00,000",
                        "source_document": "sample_policy.pdf",
                        "source_page": 1
                    }
                ]
            })
            
            logger.info(f"Query processed: {decision.decision} with confidence {decision.confidence}")
            return decision
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return DocumentDecision(
                decision="error",
                justification=f"Processing error: {str(e)}",
                confidence=0.0,
                query_id=f"ERROR_{int(time.time())}",
                processing_metadata={"error": str(e)}
            )
    
    def _create_analysis_prompt(self, query: str) -> str:
        """Create analysis prompt for Gemini"""
        return f"""
        You are an expert document analysis AI for insurance and policy processing.
        
        DOCUMENT CONTEXT:
        {self.document_context}
        
        USER QUERY: {query}
        
        TASK: Analyze the query against the document context and provide a JSON response with the following structure:
        {{
            "decision": "approved|rejected|conditional|review_required",
            "amount": <numeric_amount_if_applicable>,
            "justification": "<detailed_explanation_with_specific_references>",
            "clause_references": ["<list_of_relevant_clause_ids>"],
            "confidence": <float_between_0_and_1>
        }}
        
        GUIDELINES:
        1. For medical/insurance queries, consider coverage, waiting periods, and exclusions
        2. For contract queries, focus on terms, conditions, and obligations
        3. For email queries, identify key information and required actions
        4. Always provide specific justification with clause references
        5. Use confidence score based on clarity of information
        6. If information is insufficient, use "review_required" decision
        
        RESPONSE (JSON only):
        """
    
    def _parse_gemini_response(self, response_text: str, query_id: str) -> DocumentDecision:
        """Parse Gemini response into DocumentDecision"""
        try:
            # Extract JSON from response
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            # Parse JSON
            data = json.loads(response_text.strip())
            
            return DocumentDecision(
                decision=data.get("decision", "review_required"),
                amount=data.get("amount"),
                justification=data.get("justification", "No justification provided"),
                clause_references=data.get("clause_references", []),
                confidence=float(data.get("confidence", 0.5)),
                query_id=query_id
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            logger.error(f"Response text: {response_text}")
            
            # Fallback parsing
            return self._fallback_parse(response_text, query_id)
        
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return DocumentDecision(
                decision="review_required",
                justification=f"Response parsing error: {str(e)}",
                confidence=0.0,
                query_id=query_id
            )
    
    def _fallback_parse(self, response_text: str, query_id: str) -> DocumentDecision:
        """Fallback parsing when JSON parsing fails"""
        response_lower = response_text.lower()
        
        # Determine decision
        if "approved" in response_lower or "covered" in response_lower:
            decision = "approved"
            confidence = 0.7
        elif "rejected" in response_lower or "not covered" in response_lower:
            decision = "rejected"
            confidence = 0.7
        elif "conditional" in response_lower:
            decision = "conditional"
            confidence = 0.6
        else:
            decision = "review_required"
            confidence = 0.5
        
        return DocumentDecision(
            decision=decision,
            justification=response_text[:500] + "..." if len(response_text) > 500 else response_text,
            confidence=confidence,
            query_id=query_id,
            clause_references=["FALLBACK_PARSE"]
        )

# For backward compatibility
GeneralDocumentProcessor = LightweightDocumentProcessor
