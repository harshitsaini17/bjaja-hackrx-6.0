"""
LLM-Enhanced Claim Processor with Intelligent Parsing and Categorization
Uses LLM for flexible parsing and category generation to handle any type of test case
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
    """Types of claim decisions - simplified to only APPROVED and REJECTED"""
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


@dataclass
class PolicyClause:
    """Represents a policy clause"""
    clause_id: str
    text: str
    document: str
    page: int
    relevance_score: float
    clause_type: str


@dataclass
class DecisionJustification:
    """Detailed justification for a decision"""
    primary_reason: str
    supporting_clauses: List[PolicyClause]
    waiting_period_status: Optional[str] = ""
    coverage_analysis: Optional[str] = ""
    exclusion_analysis: Optional[str] = ""
    additional_notes: Optional[List[str]] = None


@dataclass
class ClaimDecision:
    """Complete claim decision with all metadata"""
    decision: DecisionType
    confidence: float
    justification: DecisionJustification
    processing_metadata: Dict[str, Any]
    query_id: str
    timestamp: float


@dataclass
class ParsedQuery:
    """Structured representation of parsed query using LLM"""
    raw_query: str
    age: Optional[int] = None
    gender: Optional[str] = None
    procedure: Optional[str] = None
    category: Optional[str] = None
    medical_condition: Optional[str] = None
    location: Optional[str] = None
    policy_age_months: Optional[int] = None
    urgency_level: Optional[str] = None
    sum_insured: Optional[float] = None
    keywords: Optional[List[str]] = None


class LLMEnhancedClaimProcessor:
    """LLM-Enhanced claim processor with intelligent parsing and categorization"""
    
    def __init__(self):
        """Initialize the LLM-enhanced processor"""
        self.vector_store = VectorStoreManager()
        self.llm = GoogleGenerativeAI(
            model=settings.gemini_model,
            temperature=0.1,
            max_tokens=4096,
            google_api_key=settings.gemini_api_key
        )
        
        logger.info(f"LLMEnhancedClaimProcessor initialized with model: {settings.gemini_model}")
    
    def process_claim(self, query: str) -> ClaimDecision:
        """Complete claim processing pipeline with LLM intelligence"""
        start_time = time.time()
        query_id = f"llm_enhanced_claim_{int(start_time)}"
        
        try:
            logger.info(f"Processing LLM-enhanced claim {query_id}: {query}")
            
            # Step 1: LLM-based intelligent parsing
            parsed_query = self._llm_parse_query(query)
            logger.info(f"LLM parsed query - Procedure: {parsed_query.procedure}, Category: {parsed_query.category}, Age: {parsed_query.age}")
            
            # Step 2: LLM-enhanced search for relevant policy information
            relevant_chunks = self._llm_enhanced_search(parsed_query)
            logger.info(f"Found {len(relevant_chunks)} relevant policy chunks")
            
            # Step 3: LLM-based decision making
            decision = self._llm_make_decision(parsed_query, relevant_chunks)
            
            # Step 4: Generate final decision object
            final_decision = ClaimDecision(
                decision=decision["decision"],
                confidence=decision["confidence"],
                justification=decision["justification"],
                processing_metadata={
                    "query_id": query_id,
                    "processing_time_seconds": time.time() - start_time,
                    "chunks_analyzed": len(relevant_chunks),
                    "model_used": settings.gemini_model,
                    "parsed_query": asdict(parsed_query),
                    "enhancement_version": "3.0_LLM_Enhanced"
                },
                query_id=query_id,
                timestamp=time.time()
            )
            
            logger.info(f"LLM-enhanced decision completed: {final_decision.decision.value} with confidence {final_decision.confidence}")
            return final_decision
            
        except Exception as e:
            logger.error(f"Error processing LLM-enhanced claim {query_id}: {str(e)}")
            
            # Return error decision with LLM fallback
            return self._llm_fallback_decision(query, str(e), query_id, start_time)
    
    def _llm_parse_query(self, query: str) -> ParsedQuery:
        """Use LLM to intelligently parse the query"""
        
        parsing_prompt = f"""
You are an expert medical insurance query parser. Extract structured information from this insurance claim query.

QUERY: "{query}"

Extract the following information and respond ONLY with valid JSON:

{{
    "age": <number or null>,
    "gender": "<M/F or null>",
    "procedure": "<main medical procedure/treatment>",
    "category": "<category like: baseline, room_category, icu_treatment, modern_treatment, organ_transplant, day_care, maternity, pre_post_hospitalization, ayush_treatment, emergency, cosmetic, etc.>",
    "medical_condition": "<underlying medical condition>",
    "location": "<city name or null>",
    "policy_age_months": <number of months or null>,
    "urgency_level": "<emergency/routine/day_care>",
    "sum_insured": <amount in lakhs or null>,
    "keywords": ["<relevant", "medical", "keywords>"]
}}

PARSING GUIDELINES:
- Extract age from patterns like "55F", "42M", "65 years", etc.
- Identify gender from M/F patterns
- Determine main procedure (surgery, treatment, etc.)
- Categorize based on procedure type
- Extract policy duration from "X-month policy", "Y-year policy"
- Extract sum insured from "Sum Insured X lakhs"
- Identify urgency: emergency (accident, urgent), day_care (day care procedures), routine (normal)
- Extract relevant medical keywords

Examples:
- "55F, cataract surgery both eyes, Mumbai, Sum Insured 5 lakhs, 18-month policy" 
  → age: 55, gender: "F", procedure: "cataract surgery", category: "specialty_surgery"
- "35F, ICU treatment 5 days, Chennai" 
  → procedure: "ICU treatment", category: "icu_treatment", urgency_level: "emergency"
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
                age=parsed_data.get("age"),
                gender=parsed_data.get("gender"),
                procedure=parsed_data.get("procedure"),
                category=parsed_data.get("category"),
                medical_condition=parsed_data.get("medical_condition"),
                location=parsed_data.get("location"),
                policy_age_months=parsed_data.get("policy_age_months"),
                urgency_level=parsed_data.get("urgency_level"),
                sum_insured=parsed_data.get("sum_insured"),
                keywords=parsed_data.get("keywords", [])
            )
            
        except Exception as e:
            logger.error(f"Error in LLM parsing: {str(e)}")
            # Fallback to basic regex parsing
            return self._basic_regex_parse(query)
    
    def _basic_regex_parse(self, query: str) -> ParsedQuery:
        """Fallback regex parsing if LLM fails"""
        parsed = ParsedQuery(raw_query=query)
        
        # Basic regex patterns
        age_match = re.search(r'(\d{1,3})[MF]', query, re.IGNORECASE)
        if age_match:
            parsed.age = int(age_match.group(1))
        
        gender_match = re.search(r'\d+([MF])', query, re.IGNORECASE)
        if gender_match:
            parsed.gender = gender_match.group(1).upper()
        
        # Extract procedure from common patterns
        if 'surgery' in query.lower():
            parsed.procedure = 'surgery'
            parsed.category = 'surgery'
        elif 'treatment' in query.lower():
            parsed.procedure = 'treatment'
            parsed.category = 'treatment'
        
        return parsed
    
    def _llm_enhanced_search(self, parsed_query: ParsedQuery) -> List[Dict]:
        """Use LLM to generate better search queries for policy chunks"""
        
        search_prompt = f"""
You are an expert at searching insurance policy documents. Generate 3-5 targeted search queries to find relevant policy information for this claim.

CLAIM DETAILS:
- Procedure: {parsed_query.procedure}
- Category: {parsed_query.category}
- Medical Condition: {parsed_query.medical_condition}
- Age: {parsed_query.age}
- Policy Age: {parsed_query.policy_age_months} months
- Urgency: {parsed_query.urgency_level}

Generate search queries that would find relevant policy clauses for:
1. Coverage confirmation
2. Waiting periods
3. Exclusions
4. Sub-limits
5. Special conditions

Respond with JSON array of search queries:
["query1", "query2", "query3", "query4", "query5"]

Make queries specific and relevant to the procedure and category.
"""
        
        try:
            response = self.llm.invoke(search_prompt)
            response_text = str(response).strip()
            
            json_text = self._extract_json_from_response(response_text)
            search_queries = json.loads(json_text)
            
            # Execute multiple searches
            all_results = []
            for query in search_queries[:3]:  # Limit to top 3 queries
                results = self.vector_store.search_similar(
                    query=query,
                    top_k=3
                )
                all_results.extend(results)
            
            # Deduplicate and return top results
            return self._deduplicate_results(all_results)[:8]
            
        except Exception as e:
            logger.error(f"Error in LLM search enhancement: {str(e)}")
            # Fallback to basic search
            return self._basic_search(parsed_query)
    
    def _basic_search(self, parsed_query: ParsedQuery) -> List[Dict]:
        """Fallback search if LLM search fails"""
        if parsed_query.procedure:
            return self.vector_store.search_similar(
                query=f"{parsed_query.procedure} coverage waiting period",
                top_k=5
            )
        return []
    
    def _llm_make_decision(self, parsed_query: ParsedQuery, relevant_chunks: List[Dict]) -> Dict:
        """Use LLM to make intelligent claim decisions"""
        
        # Format context
        context_text = self._format_context(relevant_chunks)
        
        decision_prompt = f"""
You are an expert insurance claim analyst for ICICI Lombard Golden Shield policy. Analyze this claim systematically and make a decision.

CLAIM DETAILS:
- Raw Query: {parsed_query.raw_query}
- Age: {parsed_query.age} years
- Gender: {parsed_query.gender}
- Procedure: {parsed_query.procedure}
- Category: {parsed_query.category}
- Medical Condition: {parsed_query.medical_condition}
- Location: {parsed_query.location}
- Policy Age: {parsed_query.policy_age_months} months
- Urgency Level: {parsed_query.urgency_level}
- Sum Insured: {parsed_query.sum_insured} lakhs

RELEVANT POLICY INFORMATION:
{context_text}

DECISION FRAMEWORK:
1. Check if procedure is covered under the policy
2. Verify waiting period requirements are met
3. Check for any exclusions that apply
4. Consider pre-existing condition rules
5. Assess emergency vs non-emergency status

DECISION TYPES (ONLY):
- APPROVED: Covered and all conditions met
- REJECTED: Not covered, waiting period not met, excluded, or insufficient policy coverage

Respond ONLY with valid JSON:
{{
    "decision": "APPROVED/REJECTED",
    "confidence": <0.0_to_1.0>,
    "primary_reason": "Clear explanation of decision",
    "waiting_period_status": "Met/Not met details",
    "coverage_confirmation": "Coverage status details",
    "exclusion_analysis": "Any exclusions that apply"
}}
"""
        
        try:
            response = self.llm.invoke(decision_prompt)
            response_text = str(response).strip()
            
            json_text = self._extract_json_from_response(response_text)
            decision_data = json.loads(json_text)
            
            # Create decision objects
            justification = DecisionJustification(
                primary_reason=decision_data.get("primary_reason", "LLM-based decision"),
                supporting_clauses=[PolicyClause(
                    clause_id="LLM_Analysis",
                    text="LLM-based comprehensive policy analysis",
                    document="ICICI_Golden_Shield_Policy.pdf",
                    page=1,
                    relevance_score=0.95,
                    clause_type="llm_analysis"
                )],
                waiting_period_status=decision_data.get("waiting_period_status", ""),
                coverage_analysis=decision_data.get("coverage_confirmation", ""),
                exclusion_analysis=decision_data.get("exclusion_analysis", ""),
                additional_notes=[]
            )
            
            return {
                "decision": DecisionType(decision_data["decision"]),
                "confidence": float(decision_data.get("confidence", 0.5)),
                "justification": justification
            }
            
        except Exception as e:
            logger.error(f"Error in LLM decision making: {str(e)}")
            # Fallback to rule-based decision
            return self._rule_based_fallback(parsed_query)
    
    def _rule_based_fallback(self, parsed_query: ParsedQuery) -> Dict:
        """Rule-based fallback decision if LLM fails"""
        procedure = (parsed_query.procedure or "").lower()
        policy_months = parsed_query.policy_age_months or 0
        
        # Simplified rule-based decisions - only APPROVED/REJECTED
        if "emergency" in procedure or "accident" in procedure:
            decision = "APPROVED"
            reason = "Emergency treatment covered immediately"
        elif "cosmetic" in procedure:
            decision = "REJECTED"
            reason = "Cosmetic surgery excluded"
        elif "surgery" in procedure and policy_months < 12:
            decision = "REJECTED"
            reason = "Insufficient waiting period for surgery"
        elif policy_months < 1:
            decision = "REJECTED"
            reason = "Policy waiting period not met"
        else:
            decision = "APPROVED"
            reason = "Standard coverage approved"
        
        justification = DecisionJustification(
            primary_reason=reason,
            supporting_clauses=[]
        )
        
        return {
            "decision": DecisionType(decision),
            "confidence": 0.7,
            "justification": justification
        }
    
    def _llm_fallback_decision(self, query: str, error_msg: str, query_id: str, start_time: float) -> ClaimDecision:
        """LLM-based fallback decision for error cases"""
        
        fallback_prompt = f"""
You are an insurance claim analyst. The system encountered an error, but you need to make a decision based on the query.

QUERY: "{query}"
ERROR: {error_msg}

Based on the query alone, make a reasonable decision. Consider:
- Is this an emergency case?
- What type of procedure is mentioned?
- Are there obvious exclusions?

Only use APPROVED or REJECTED decisions.

Respond with JSON:
{{
    "decision": "APPROVED/REJECTED",
    "confidence": <0.0_to_1.0>,
    "reason": "Brief explanation"
}}
"""
        
        try:
            response = self.llm.invoke(fallback_prompt)
            response_text = str(response).strip()
            
            json_text = self._extract_json_from_response(response_text)
            decision_data = json.loads(json_text)
            
            justification = DecisionJustification(
                primary_reason=f"Fallback decision: {decision_data.get('reason', 'System error occurred')}",
                supporting_clauses=[]
            )
            
            return ClaimDecision(
                decision=DecisionType(decision_data["decision"]),
                confidence=float(decision_data.get("confidence", 0.3)),
                justification=justification,
                processing_metadata={
                    "query_id": query_id,
                    "error": error_msg,
                    "processing_time_seconds": time.time() - start_time,
                    "fallback_used": "LLM_fallback"
                },
                query_id=query_id,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"LLM fallback also failed: {str(e)}")
            
            # Ultimate fallback
            return ClaimDecision(
                decision=DecisionType.REJECTED,
                confidence=0.0,
                justification=DecisionJustification(
                    primary_reason=f"System error: {error_msg}",
                    supporting_clauses=[]
                ),
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
        
        # Method 1: Look for JSON code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
        
        # Method 2: Find content between first { and last }
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            return response_text[start_idx:end_idx]
        
        # Method 3: Look for array format
        start_idx = response_text.find('[')
        end_idx = response_text.rfind(']') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            return response_text[start_idx:end_idx]
        
        raise ValueError("No valid JSON found in response")
    
    def _format_context(self, relevant_chunks: List[Dict]) -> str:
        """Format policy context for LLM"""
        context_sections = []
        
        for i, chunk in enumerate(relevant_chunks):
            content = chunk.get('metadata', {}).get('content', '')
            score = chunk.get('score', 0.0)
            
            context_sections.append(f"""
POLICY CLAUSE {i+1} (Relevance: {score:.3f}):
{content}
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


# Alias for backward compatibility
EnhancedClaimDecisionEngine = LLMEnhancedClaimProcessor
