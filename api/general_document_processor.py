"""
General-Purpose LLM Document Processing System
Handles natural language queries for policy documents, contracts, and emails
"""

import os
import sys
import json
import logging
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import settings

# LLM and vector store imports
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from src.vector_storage import VectorStoreManager

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Types of document processing decisions"""
    APPROVED = "approved"
    REJECTED = "rejected"
    CONDITIONAL = "conditional"
    REVIEW_REQUIRED = "review_required"


@dataclass
class DocumentClause:
    """Represents a relevant document clause with original text"""
    clause_id: str
    original_text: str  # Full original text from document
    summary: str  # Processed/summarized content
    document: str
    page: int
    relevance_score: float
    clause_type: str
    source_reference: str  # Reference identifier


@dataclass
class ProcessingJustification:
    """Detailed justification for a decision"""
    summary: str
    supporting_clauses: List[DocumentClause]
    clause_mapping: Dict[str, str]  # Maps decision factors to specific clauses
    confidence_breakdown: Dict[str, float]
    additional_notes: Optional[List[str]] = None


@dataclass
class DocumentDecision:
    """Complete document processing decision"""
    decision: str  # approved/rejected/conditional/review_required
    justification: str  # Clear explanation
    amount: Optional[float] = None  # Amount if applicable (claims, payouts, etc.)
    clause_references: Optional[List[str]] = None  # Specific clause IDs referenced
    confidence: float = 0.0
    processing_metadata: Optional[Dict[str, Any]] = None
    query_id: str = ""
    timestamp: float = 0.0


@dataclass
class ParsedQuery:
    """Structured representation of any document query"""
    raw_query: str
    query_type: Optional[str] = None  # claim, contract_review, policy_check, email_analysis
    entity_details: Optional[Dict[str, Any]] = None  # Age, gender, location, etc.
    action_requested: Optional[str] = None  # What the user wants to know
    document_context: Optional[str] = None  # Type of document to search
    amount_mentioned: Optional[float] = None
    time_constraints: Optional[Dict[str, Any]] = None
    keywords: Optional[List[str]] = None


class GeneralDocumentProcessor:
    """General-purpose LLM document processor for policies, contracts, and emails"""
    
    def __init__(self):
        """Initialize the general document processor"""
        self.vector_store = VectorStoreManager()
        self.llm = GoogleGenerativeAI(
            model=settings.gemini_model,
            temperature=0.1,
            max_tokens=4096,
            google_api_key=settings.gemini_api_key
        )
        
        logger.info(f"GeneralDocumentProcessor initialized with model: {settings.gemini_model}")
    
    def process_query(self, query: str) -> DocumentDecision:
        """Complete document processing pipeline"""
        start_time = time.time()
        query_id = f"doc_query_{int(start_time)}"
        
        try:
            logger.info(f"Processing document query {query_id}: {query}")
            
            # Step 1: Parse and understand the query
            parsed_query = self._parse_general_query(query)
            logger.info(f"Parsed query - Type: {parsed_query.query_type}, Action: {parsed_query.action_requested}")
            
            # Step 2: Retrieve relevant document content (max 5 chunks)
            relevant_chunks = self._search_relevant_content(parsed_query)
            logger.info(f"Found {len(relevant_chunks)} relevant document chunks")
            
            # Step 3: Make decision based on retrieved content
            decision = self._make_informed_decision(parsed_query, relevant_chunks)
            
            # Step 4: Generate final response with clause details
            clause_details = decision.get("clause_details", [])
            
            final_decision = DocumentDecision(
                decision=decision["decision"],
                amount=decision.get("amount"),
                justification=decision["justification"],
                clause_references=decision.get("clause_references", []),
                confidence=decision.get("confidence", 0.0),
                processing_metadata={
                    "query_id": query_id,
                    "processing_time_seconds": round(time.time() - start_time, 2),
                    "chunks_analyzed": len(relevant_chunks),
                    "model_used": settings.gemini_model,
                    "parsed_query": asdict(parsed_query),
                    "system_version": "1.0_General_Purpose_Optimized",
                    "clause_details": clause_details,  # Include original text and details
                    "optimization_notes": "Reduced LLM calls, faster search, original text preserved"
                },
                query_id=query_id,
                timestamp=time.time()
            )
            
            logger.info(f"Document query processed: {final_decision.decision} with confidence {final_decision.confidence}")
            return final_decision
            
        except Exception as e:
            logger.error(f"Error processing document query {query_id}: {str(e)}")
            return self._error_fallback_decision(query, str(e), query_id, start_time)
    
    def _parse_general_query(self, query: str) -> ParsedQuery:
        """Parse any type of document query using LLM"""
        
        parsing_prompt = f"""
You are an expert document analysis system. Parse this natural language query to understand what information the user needs from documents (policies, contracts, emails, etc.).

QUERY: "{query}"

Extract and structure the following information. Respond ONLY with valid JSON:

{{
    "query_type": "<claim_processing|contract_review|policy_inquiry|compliance_check|email_analysis|general_inquiry>",
    "entity_details": {{
        "age": <number or null>,
        "gender": "<M/F or null>",
        "location": "<city/state or null>",
        "occupation": "<profession or null>",
        "other_relevant_details": {{}}
    }},
    "action_requested": "<what the user wants to know - approval, coverage, terms, etc.>",
    "document_context": "<insurance|contract|policy|email|legal|hr|financial>",
    "amount_mentioned": <monetary amount in the query or null>,
    "time_constraints": {{
        "policy_duration": "<X months/years or null>",
        "deadlines": "<any time limits mentioned>",
        "effective_dates": "<relevant dates>"
    }},
    "keywords": ["<relevant", "keywords", "for", "search>"]
}}

EXAMPLES:
- "46-year-old male, knee surgery in Pune, 3-month-old insurance policy" 
  → query_type: "claim_processing", action_requested: "coverage verification"
  
- "What are the termination clauses in the employment contract for remote workers?"
  → query_type: "contract_review", action_requested: "termination clause analysis"
  
- "Email about deadline for submitting quarterly reports - what's the policy?"
  → query_type: "email_analysis", action_requested: "deadline policy verification"

Focus on extracting the core intent and relevant details for document search.
"""
        
        try:
            response = self.llm.invoke(parsing_prompt)
            response_text = str(response).strip()
            
            # Extract JSON from response
            json_text = self._extract_json_from_response(response_text)
            parsed_data = json.loads(json_text)
            
            # Create ParsedQuery object
            return ParsedQuery(
                raw_query=query,
                query_type=parsed_data.get("query_type"),
                entity_details=parsed_data.get("entity_details"),
                action_requested=parsed_data.get("action_requested"),
                document_context=parsed_data.get("document_context"),
                amount_mentioned=parsed_data.get("amount_mentioned"),
                time_constraints=parsed_data.get("time_constraints"),
                keywords=parsed_data.get("keywords", [])
            )
            
        except Exception as e:
            logger.error(f"Error in query parsing: {str(e)}")
            # Fallback to basic parsing
            return self._basic_query_parse(query)
    
    def _basic_query_parse(self, query: str) -> ParsedQuery:
        """Fallback parsing if LLM fails"""
        keywords = []
        amount = None
        
        # Extract basic keywords
        words = query.lower().split()
        keywords = [word for word in words if len(word) > 3]
        
        # Try to extract amount
        amount_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:lakhs?|thousands?|k|lakh)', query, re.IGNORECASE)
        if amount_match:
            amount = float(amount_match.group(1))
        
        return ParsedQuery(
            raw_query=query,
            query_type="general_inquiry",
            action_requested="information_retrieval",
            keywords=keywords[:10],  # Limit keywords
            amount_mentioned=amount
        )
    
    def _search_relevant_content(self, parsed_query: ParsedQuery) -> List[Dict]:
        """Search for relevant content with max 5 chunks - OPTIMIZED"""
        
        # Faster search with optimized prompts
        search_queries = self._generate_optimized_search_queries(parsed_query)
        
        # Execute searches with limited results
        all_results = []
        for query in search_queries[:2]:  # Reduced to 2 queries for speed
            results = self.vector_store.search_similar(
                query=query,
                top_k=3  # Max 3 results per query
            )
            all_results.extend(results)
        
        # Deduplicate and limit to top 5 chunks
        return self._deduplicate_results(all_results)[:5]
    
    def _generate_optimized_search_queries(self, parsed_query: ParsedQuery) -> List[str]:
        """Generate optimized search queries without LLM for speed"""
        queries = []
        
        # Primary query based on parsed content
        if parsed_query.keywords:
            primary_keywords = " ".join(parsed_query.keywords[:3])
            queries.append(primary_keywords)
        
        # Secondary query for specific context
        if parsed_query.query_type and parsed_query.action_requested:
            context_query = f"{parsed_query.action_requested} {parsed_query.document_context or ''}"
            queries.append(context_query)
        
        # Fallback to raw query
        if not queries:
            queries.append(parsed_query.raw_query)
        
        return queries
    
    def _basic_search(self, parsed_query: ParsedQuery) -> List[Dict]:
        """Fallback search using basic keywords"""
        if parsed_query.keywords:
            search_query = " ".join(parsed_query.keywords[:5])
            return self.vector_store.search_similar(
                query=search_query,
                top_k=5
            )
        return []
    
    def _make_informed_decision(self, parsed_query: ParsedQuery, relevant_chunks: List[Dict]) -> Dict:
        """Make decision based on retrieved content - OPTIMIZED with original text"""
        
        # Extract clause information with original text
        clauses_info = self._extract_clause_information(relevant_chunks)
        
        # Format context from chunks with original text
        context_text = self._format_optimized_context(relevant_chunks)
        
        # Simplified decision prompt for faster processing
        decision_prompt = f"""
Analyze this query against document content and decide:

QUERY: {parsed_query.raw_query}
TYPE: {parsed_query.query_type}
ACTION: {parsed_query.action_requested}

DOCUMENT CLAUSES:
{context_text}

Respond ONLY with JSON:
{{
    "decision": "approved|rejected|conditional|review_required",
    "amount": <number or null>,
    "justification": "Brief explanation referencing specific clauses",
    "clause_references": ["clause_1", "clause_2"],
    "confidence": <0.0_to_1.0>
}}

RULES:
- "approved": Clearly covered
- "rejected": Clearly excluded  
- "conditional": Covered with conditions
- "review_required": Unclear/insufficient info

Be concise and reference clauses by their IDs.
"""
        
        try:
            response = self.llm.invoke(decision_prompt)
            response_text = str(response).strip()
            
            json_text = self._extract_json_from_response(response_text)
            decision_data = json.loads(json_text)
            
            # Add clause details with original text
            enhanced_decision = {
                "decision": decision_data.get("decision", "review_required"),
                "amount": decision_data.get("amount"),
                "justification": decision_data.get("justification", "Decision based on document analysis"),
                "clause_references": decision_data.get("clause_references", []),
                "confidence": float(decision_data.get("confidence", 0.5)),
                "clause_details": clauses_info  # Include original text and details
            }
            
            return enhanced_decision
            
        except Exception as e:
            logger.error(f"Error in decision making: {str(e)}")
            return self._fallback_decision(parsed_query)
    
    def _extract_clause_information(self, relevant_chunks: List[Dict]) -> List[Dict]:
        """Extract clause information including original text"""
        clauses_info = []
        
        for i, chunk in enumerate(relevant_chunks):
            metadata = chunk.get('metadata', {})
            
            clause_info = {
                "clause_id": f"CLAUSE_{i+1}",
                "clause_reference": metadata.get('clause_reference', f"REF_{i+1}"),
                "original_text": metadata.get('original_text', metadata.get('content', '')),
                "summary": metadata.get('summary', ''),
                "source_document": metadata.get('source_document', 'Unknown'),
                "source_page": metadata.get('source_page', 0),
                "relevance_score": chunk.get('score', 0.0),
                "keywords": metadata.get('keywords', [])
            }
            
            clauses_info.append(clause_info)
        
        return clauses_info
    
    def _fallback_decision(self, parsed_query: ParsedQuery) -> Dict:
        """Fallback decision if LLM fails"""
        return {
            "decision": "review_required",
            "amount": parsed_query.amount_mentioned,
            "justification": f"Unable to process request: {parsed_query.action_requested}. Manual review required.",
            "clause_references": [],
            "confidence": 0.3
        }
    
    def _error_fallback_decision(self, query: str, error_msg: str, query_id: str, start_time: float) -> DocumentDecision:
        """Error fallback decision"""
        return DocumentDecision(
            decision="review_required",
            amount=None,
            justification=f"System error occurred: {error_msg}. Manual review required.",
            clause_references=[],
            confidence=0.0,
            processing_metadata={
                "query_id": query_id,
                "error": error_msg,
                "processing_time_seconds": time.time() - start_time
            },
            query_id=query_id,
            timestamp=time.time()
        )
    
    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON from LLM response"""
        response_text = str(response_text).strip()
        
        # Look for JSON code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
        
        # Find content between first { and last }
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            return response_text[start_idx:end_idx]
        
        # Look for array format
        start_idx = response_text.find('[')
        end_idx = response_text.rfind(']') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            return response_text[start_idx:end_idx]
        
        raise ValueError("No valid JSON found in response")
    
    def _format_optimized_context(self, relevant_chunks: List[Dict]) -> str:
        """Format document context for LLM - OPTIMIZED for speed"""
        context_sections = []
        
        for i, chunk in enumerate(relevant_chunks):
            metadata = chunk.get('metadata', {})
            summary = metadata.get('summary', '')
            original_content = metadata.get('original_text', metadata.get('content', ''))
            score = chunk.get('score', 0.0)
            doc_name = metadata.get('source_document', 'Unknown')
            
            # Use summary for faster processing, keep original available
            display_content = summary if summary else original_content[:300]
            
            context_sections.append(f"""
CLAUSE_{i+1} (Score: {score:.2f}, Doc: {doc_name}):
{display_content}
""")
        
        return "\n".join(context_sections)
    
    def _format_context(self, relevant_chunks: List[Dict]) -> str:
        """Format document context for LLM - LEGACY METHOD"""
        context_sections = []
        
        for i, chunk in enumerate(relevant_chunks):
            content = chunk.get('metadata', {}).get('content', '')
            score = chunk.get('score', 0.0)
            doc_name = chunk.get('metadata', {}).get('source_document', 'Unknown')
            
            context_sections.append(f"""
DOCUMENT CLAUSE {i+1} (Relevance: {score:.3f}):
Source: {doc_name}
Content: {content}
---""")
        
        return "\n".join(context_sections)
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results"""
        unique_results = []
        seen_content = set()
        
        for result in results:
            content = result.get('metadata', {}).get('content', '')
            content_hash = hash(content[:100])
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
        
        return sorted(unique_results, key=lambda x: x.get('score', 0), reverse=True)
