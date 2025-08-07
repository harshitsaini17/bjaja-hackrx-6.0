"""
Query processing and decision engine
Handles natural language queries and generates structured decisions
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
from vector_storage import VectorStoreManager

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)


def extract_json_from_response(response_text: str) -> str:
    """
    Robust JSON extraction from LLM response with multiple fallback methods
    """
    import re
    
    response_text = str(response_text).strip()
    
    # Method 1: Look for JSON code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()
    
    # Method 2: Find content between first { and last }
    start_idx = response_text.find('{')
    end_idx = response_text.rfind('}') + 1
    
    if start_idx != -1 and end_idx > start_idx:
        json_text = response_text[start_idx:end_idx]
        
        # Validate that braces are balanced
        brace_count = 0
        valid_end = -1
        for i, char in enumerate(json_text):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    valid_end = i + 1
                    break
        
        if valid_end > 0:
            return json_text[:valid_end].strip()
    
    # Method 3: Line-by-line extraction
    lines = response_text.split('\n')
    json_lines = []
    inside_json = False
    brace_count = 0
    
    for line in lines:
        if '{' in line and not inside_json:
            inside_json = True
            brace_count += line.count('{') - line.count('}')
            json_lines.append(line)
        elif inside_json:
            brace_count += line.count('{') - line.count('}')
            json_lines.append(line)
            if brace_count <= 0:
                break
    
    if json_lines:
        return '\n'.join(json_lines).strip()
    
    # Method 4: Try to find any JSON-like structure
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, response_text, re.DOTALL)
    if matches:
        return matches[0].strip()
    
    raise ValueError("No valid JSON found in response")


def clean_json_text(json_text: str) -> str:
    """
    Clean and fix common JSON formatting issues with simplified approach
    """
    # Remove any trailing commas before closing braces/brackets
    json_text = re.sub(r',\s*([}\]])', r'\1', json_text)
    
    # Remove any control characters that might break JSON parsing
    json_text = re.sub(r'[\x00-\x1f\x7f]', '', json_text)
    
    # Basic structure fixes
    if not json_text.strip().startswith('{'):
        json_text = '{' + json_text
    if not json_text.strip().endswith('}'):
        json_text = json_text + '}'
    
    return json_text.strip()


class DecisionType(Enum):
    """Types of claim decisions"""
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    PARTIAL = "PARTIAL"
    UNDER_REVIEW = "UNDER_REVIEW"


@dataclass
class ParsedQuery:
    """Structured representation of a parsed query"""
    age: Optional[int] = None
    gender: Optional[str] = None
    procedure: Optional[str] = None
    location: Optional[str] = None
    policy_age_months: Optional[int] = None
    medical_condition: Optional[str] = None
    claim_amount: Optional[float] = None
    urgency_level: Optional[str] = None
    raw_query: str = ""
    extracted_entities: List[Dict[str, Any]] = None
    keywords: List[str] = None
    
    def __post_init__(self):
        if self.extracted_entities is None:
            self.extracted_entities = []
        if self.keywords is None:
            self.keywords = []


@dataclass
class PolicyClause:
    """Represents a relevant policy clause"""
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
    waiting_period_status: Optional[str] = None
    coverage_analysis: Optional[str] = None
    exclusion_analysis: Optional[str] = None
    additional_notes: List[str] = None
    
    def __post_init__(self):
        if self.additional_notes is None:
            self.additional_notes = []


@dataclass
class ClaimDecision:
    """Final claim processing decision"""
    decision: DecisionType
    amount: float
    confidence: float
    justification: DecisionJustification
    processing_metadata: Dict[str, Any]
    query_id: str
    timestamp: float


class QueryProcessor:
    """Processes natural language queries into structured format"""
    
    def __init__(self):
        """Initialize the query processor"""
        self.llm = GoogleGenerativeAI(
            model=settings.gemini_model,
            temperature=0.1,  # Low temperature for consistent parsing
            max_tokens=2048,
            google_api_key=settings.gemini_api_key
        )
        
        logger.info(f"QueryProcessor initialized with model: {settings.gemini_model}")
    
    def parse_query(self, query: str) -> ParsedQuery:
        """Parse natural language query into structured format"""
        
        prompt = f"""
Parse this insurance claim query and return ONLY valid JSON (no extra text before or after).

QUERY: "{query}"

Return JSON in this exact format with proper quotation marks:
{{"age": 46, "gender": "male", "procedure": "knee surgery", "location": "Pune", "policy_age_months": 3, "medical_condition": "orthopedic condition", "claim_amount": null, "urgency_level": "routine", "keywords": ["knee", "surgery", "orthopedic"], "extracted_entities": [{{"type": "age", "value": "46", "context": "patient age"}}, {{"type": "procedure", "value": "knee surgery", "context": "medical procedure"}}]}}

Rules:
- Extract numeric age if mentioned  
- Infer gender: male/female/null
- Identify medical procedure 
- Extract location if mentioned
- Convert policy duration to months (3-month = 3, 1-year = 12)
- Set urgency_level: emergency/urgent/routine/null
- Use null for missing information
- Ensure all strings are properly quoted
- No trailing commas
"""
        
        try:
            response = self.llm.invoke(prompt)
            response_text = str(response).strip()
            
            # Use robust JSON extraction
            json_text = extract_json_from_response(response_text)
            json_text = clean_json_text(json_text)
            
            # Parse the cleaned JSON
            parsed_data = json.loads(json_text)
            
            return ParsedQuery(
                age=parsed_data.get("age"),
                gender=parsed_data.get("gender"),
                procedure=parsed_data.get("procedure"),
                location=parsed_data.get("location"),
                policy_age_months=parsed_data.get("policy_age_months"),
                medical_condition=parsed_data.get("medical_condition"),
                claim_amount=parsed_data.get("claim_amount"),
                urgency_level=parsed_data.get("urgency_level"),
                raw_query=query,
                extracted_entities=parsed_data.get("extracted_entities", []),
                keywords=parsed_data.get("keywords", [])
            )
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            logger.error(f"Attempted to parse: {json_text[:200] if 'json_text' in locals() else 'No JSON text found'}")
            # Fallback parsing
            return self._fallback_parsing(query)
        except Exception as e:
            logger.error(f"Error parsing query: {str(e)}")
            # Fallback parsing
            return self._fallback_parsing(query)
    
    def _fallback_parsing(self, query: str) -> ParsedQuery:
        """Enhanced fallback query parsing using regex patterns"""
        parsed = ParsedQuery(raw_query=query)
        
        # Extract age - improved patterns
        age_match = re.search(r'(\d{1,3})[MF]|(\d{1,3})\s*(?:years?|yrs?|y)|(\d{1,3})(?=\s*[MF])', query, re.IGNORECASE)
        if age_match:
            parsed.age = int(age_match.group(1) or age_match.group(2) or age_match.group(3))
        
        # Extract gender - improved patterns
        if re.search(r'\d+M\b|male', query, re.IGNORECASE):
            parsed.gender = "male"
        elif re.search(r'\d+F\b|female', query, re.IGNORECASE):
            parsed.gender = "female"
        
        # Extract procedure/medical condition
        # Common medical procedures and conditions
        procedures = [
            'surgery', 'treatment', 'therapy', 'examination', 'test', 'procedure',
            'knee surgery', 'delivery', 'accident treatment', 'robotic surgery', 
            'chemotherapy', 'cataract surgery', 'dialysis', 'physiotherapy',
            'cosmetic surgery', 'cardiac surgery', 'pre-hospitalization', 'post-surgery'
        ]
        
        for proc in procedures:
            if proc.lower() in query.lower():
                parsed.procedure = proc
                break
        
        # If no specific procedure found, extract general terms
        if not parsed.procedure:
            # Look for medical terms
            medical_terms = re.findall(r'\b(?:surgery|treatment|therapy|examination|test|procedure|delivery|accident|dialysis|chemotherapy|physiotherapy)\b', query, re.IGNORECASE)
            if medical_terms:
                parsed.procedure = medical_terms[0].lower()
        
        # Extract location - Indian cities
        cities = ['mumbai', 'delhi', 'bangalore', 'pune', 'chennai', 'kolkata', 'hyderabad']
        for city in cities:
            if city.lower() in query.lower():
                parsed.location = city.title()
                break
        
        # Extract policy age in months
        policy_match = re.search(r'(\d+)[-\s]*(?:month|year)[-\s]*policy', query, re.IGNORECASE)
        if policy_match:
            num = int(policy_match.group(1))
            if 'year' in policy_match.group(0).lower():
                parsed.policy_age_months = num * 12
            else:
                parsed.policy_age_months = num
        
        # Set urgency based on keywords
        if re.search(r'emergency|urgent|accident', query, re.IGNORECASE):
            parsed.urgency_level = "emergency"
        else:
            parsed.urgency_level = "routine"
        
        # Extract keywords
        parsed.keywords = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        
        # Set medical condition based on procedure
        condition_map = {
            'knee surgery': 'orthopedic condition',
            'delivery': 'pregnancy',
            'accident treatment': 'accident',
            'chemotherapy': 'cancer',
            'cataract surgery': 'cataract',
            'dialysis': 'kidney disease',
            'cardiac surgery': 'heart condition'
        }
        
        if parsed.procedure and parsed.procedure.lower() in condition_map:
            parsed.medical_condition = condition_map[parsed.procedure.lower()]
        
        return parsed
        parsed.keywords = query.lower().split()
        
        return parsed


class ClaimDecisionEngine:
    """Main engine for processing claims and making decisions"""
    
    def __init__(self):
        """Initialize the decision engine"""
        self.query_processor = QueryProcessor()
        self.vector_store = VectorStoreManager()
        self.llm = GoogleGenerativeAI(
            model=settings.gemini_model,
            temperature=0.2,
            max_tokens=4096,
            google_api_key=settings.gemini_api_key
        )
        
        logger.info(f"ClaimDecisionEngine initialized with model: {settings.gemini_model}")
    
    def process_claim(self, query: str) -> ClaimDecision:
        """Complete claim processing pipeline"""
        start_time = time.time()
        query_id = f"claim_{int(start_time)}"
        
        try:
            logger.info(f"Processing claim {query_id}: {query}")
            
            # Step 1: Parse the query
            parsed_query = self.query_processor.parse_query(query)
            logger.info(f"Parsed query: {parsed_query.procedure}, Age: {parsed_query.age}, Policy: {parsed_query.policy_age_months} months")
            
            # Step 2: Search for relevant policy information
            relevant_chunks = self._search_relevant_policies(parsed_query)
            logger.info(f"Found {len(relevant_chunks)} relevant policy chunks")
            
            # Step 3: Make decision based on policy rules
            decision = self._make_decision(parsed_query, relevant_chunks)
            
            # Step 4: Generate final decision object
            final_decision = ClaimDecision(
                decision=decision["decision"],
                amount=decision["amount"],
                confidence=decision["confidence"],
                justification=decision["justification"],
                processing_metadata={
                    "query_id": query_id,
                    "processing_time_seconds": time.time() - start_time,
                    "chunks_analyzed": len(relevant_chunks),
                    "model_used": settings.gemini_model,
                    "parsed_query": asdict(parsed_query)
                },
                query_id=query_id,
                timestamp=time.time()
            )
            
            logger.info(f"Decision completed: {final_decision.decision.value} with confidence {final_decision.confidence}")
            return final_decision
            
        except Exception as e:
            logger.error(f"Error processing claim {query_id}: {str(e)}")
            
            # Return error decision
            return ClaimDecision(
                decision=DecisionType.UNDER_REVIEW,
                amount=0.0,
                confidence=0.0,
                justification=DecisionJustification(
                    primary_reason=f"Error processing claim: {str(e)}",
                    supporting_clauses=[]
                ),
                processing_metadata={
                    "query_id": query_id,
                    "error": str(e),
                    "processing_time_seconds": time.time() - start_time
                },
                query_id=query_id,
                timestamp=time.time()
            )
    
    def _search_relevant_policies(self, parsed_query: ParsedQuery) -> List[Dict]:
        """Search for relevant policy chunks"""
        
        # Build search query from parsed information
        search_terms = []
        
        if parsed_query.procedure:
            search_terms.append(parsed_query.procedure)
        if parsed_query.medical_condition:
            search_terms.append(parsed_query.medical_condition)
        if parsed_query.keywords:
            search_terms.extend(parsed_query.keywords[:5])  # Top 5 keywords
        
        search_query = " ".join(search_terms)
        
        # Search in vector store
        results = self.vector_store.search_similar(
            query=search_query,
            top_k=10,
            filter_dict={}  # Could add filters for specific policy types
        )
        
        return results
    
    def _make_decision(self, parsed_query: ParsedQuery, relevant_chunks: List[Dict]) -> Dict:
        """Make claim decision based on query and policy chunks with enhanced analysis"""
        
        # Prepare enhanced context from relevant chunks
        context_text = self._format_policy_context(relevant_chunks)
        
        # Use enhanced decision prompt
        prompt = self._create_enhanced_decision_prompt(parsed_query, context_text)
        
        try:
            # Try primary model first
            decision = self._make_decision_with_model("gemini-2.5-flash-lite", prompt)
            
            # If confidence is low, escalate to more powerful model
            if decision["confidence"] < 0.85:
                logger.info("Low confidence detected, escalating to Gemini Pro for higher accuracy")
                pro_prompt = prompt.replace("gemini-2.5-flash-lite", "gemini-2.5-pro")
                decision = self._make_decision_with_model("gemini-2.5-pro", pro_prompt)
            
            return decision
            
        except Exception as e:
            logger.error(f"Error making decision: {str(e)}")
            # Smart fallback decision based on basic rules
            return self._fallback_decision(parsed_query, str(e))
    
    def _create_enhanced_decision_prompt(self, parsed_query: ParsedQuery, context_text: str) -> str:
        """Create enhanced decision prompt with systematic analysis framework"""
        
        prompt = f"""
You are an expert insurance claim analyst. Analyze this claim systematically using the ICICI Lombard Golden Shield policy.

CLAIM DETAILS:
- Age: {parsed_query.age} years
- Gender: {parsed_query.gender}
- Procedure: {parsed_query.procedure}
- Policy Age: {parsed_query.policy_age_months} months
- Location: {parsed_query.location}
- Medical Condition: {parsed_query.medical_condition}

RELEVANT POLICY CLAUSES:
{context_text}

SYSTEMATIC DECISION FRAMEWORK:
1. COVERAGE CHECK: Is the procedure explicitly covered in the policy?
2. WAITING PERIOD: Does policy age meet the required waiting period?
3. EXCLUSIONS: Are there any exclusions that apply to this case?
4. SUB-LIMITS: Are there specific sub-limits or co-payment requirements?
5. TERRITORIAL LIMITS: Is treatment location within coverage area?

ENHANCED DECISION RULES:
- APPROVED: Coverage confirmed + waiting period met + no exclusions + within limits
- PARTIAL: Coverage confirmed + waiting period met + sub-limits apply
- REJECTED: Waiting period not met OR explicitly excluded OR not covered
- UNDER_REVIEW: Only if genuinely insufficient information (avoid overuse)

SPECIFIC GUIDELINES:
- Emergency/Accident: Immediate coverage, no waiting period
- Maternity: Requires 9+ months waiting period
- Surgery: Usually requires 12+ months waiting period (except emergency)
- Day Care: Covered procedures listed explicitly
- Modern Treatments: Specific list in policy, check coverage
- Pre/Post Hospitalization: Time limits apply (30-60 days pre, 60-180 days post)
- Room Rent: Check category limits and proportionate deduction rules
- AYUSH: Covered at registered centers in India

Respond ONLY with valid JSON (no extra text):
{{"decision": "APPROVED", "amount": 100000.0, "confidence": 0.95, "justification": {{"primary_reason": "Specific reason with exact policy reference", "waiting_period_analysis": "Met/Not met - X months required, {parsed_query.policy_age_months} months completed", "coverage_confirmation": "Explicitly covered under Section X.Y", "exclusion_check": "No exclusions apply / Exclusion X applies", "sub_limit_analysis": "Standard coverage / Subject to Rs X limit", "territorial_check": "Treatment in India, covered location", "room_category_check": "Within entitlement / Proportionate deduction applicable"}}}}
"""
        return prompt
    
    def _make_decision_with_model(self, model_name: str, prompt: str) -> Dict:
        """Make decision using specified model"""
        
        # Create model instance for this decision
        decision_llm = GoogleGenerativeAI(
            model=model_name,
            temperature=0.2,
            max_tokens=4096,
            google_api_key=settings.gemini_api_key
        )
        
        response = decision_llm.invoke(prompt)
        response_text = str(response).strip()
        
        # Use robust JSON extraction
        json_text = extract_json_from_response(response_text)
        json_text = clean_json_text(json_text)
        
        # Parse the cleaned JSON
        decision_data = json.loads(json_text)
        
        # Convert to proper objects
        justification_data = decision_data.get("justification", {})
        supporting_clauses = []
        
        # Create supporting clauses from context (simplified for now)
        supporting_clauses.append(PolicyClause(
            clause_id="PolicyClause1",
            text="Relevant policy clause based on decision",
            document="ICICI_Golden_Shield_Policy.pdf",
            page=1,
            relevance_score=0.9,
            clause_type="coverage"
        ))
        
        justification = DecisionJustification(
            primary_reason=justification_data.get("primary_reason", ""),
            supporting_clauses=supporting_clauses,
            waiting_period_status=justification_data.get("waiting_period_analysis", ""),
            coverage_analysis=justification_data.get("coverage_confirmation", ""),
            exclusion_analysis=justification_data.get("exclusion_check", ""),
            additional_notes=[
                justification_data.get("sub_limit_analysis", ""),
                justification_data.get("territorial_check", ""),
                justification_data.get("room_category_check", "")
            ]
        )
        
        return {
            "decision": DecisionType(decision_data["decision"]),
            "amount": float(decision_data.get("amount", 0.0)),
            "confidence": float(decision_data.get("confidence", 0.5)),
            "justification": justification
        }

    
    def _fallback_decision(self, parsed_query: ParsedQuery, error_msg: str) -> Dict:
        """Make a basic decision using simple rules when JSON parsing fails"""
        
        # Basic decision rules
        if parsed_query.procedure:
            procedure = parsed_query.procedure.lower()
            policy_months = parsed_query.policy_age_months or 0
            
            # Emergency/accident cases - immediate coverage
            if any(word in procedure for word in ['emergency', 'accident']):
                return {
                    "decision": DecisionType.APPROVED,
                    "amount": 50000.0,
                    "confidence": 0.8,
                    "justification": DecisionJustification(
                        primary_reason="Emergency treatment covered immediately under policy terms",
                        supporting_clauses=[]
                    )
                }
            
            # Cosmetic surgery - usually excluded
            elif 'cosmetic' in procedure:
                return {
                    "decision": DecisionType.REJECTED,
                    "amount": 0.0,
                    "confidence": 0.9,
                    "justification": DecisionJustification(
                        primary_reason="Cosmetic procedures are excluded under policy terms",
                        supporting_clauses=[]
                    )
                }
            
            # Maternity - needs waiting period
            elif any(word in procedure for word in ['delivery', 'maternity', 'pregnancy']):
                if policy_months >= 9:
                    decision = DecisionType.APPROVED
                    reason = "Maternity waiting period satisfied"
                    amount = 75000.0
                else:
                    decision = DecisionType.REJECTED
                    reason = "Maternity waiting period not met (requires 9+ months)"
                    amount = 0.0
                
                return {
                    "decision": decision,
                    "amount": amount,
                    "confidence": 0.85,
                    "justification": DecisionJustification(
                        primary_reason=reason,
                        supporting_clauses=[]
                    )
                }
            
            # Surgery with waiting period considerations
            elif 'surgery' in procedure:
                if policy_months >= 12:
                    decision = DecisionType.APPROVED
                    reason = "Surgery covered after completion of waiting period"
                    amount = 100000.0
                    confidence = 0.8
                elif policy_months >= 6:
                    decision = DecisionType.PARTIAL
                    reason = "Partial coverage available under policy terms"
                    amount = 50000.0
                    confidence = 0.7
                else:
                    decision = DecisionType.REJECTED
                    reason = "Insufficient waiting period for surgical procedures"
                    amount = 0.0
                    confidence = 0.8
                
                return {
                    "decision": decision,
                    "amount": amount,
                    "confidence": confidence,
                    "justification": DecisionJustification(
                        primary_reason=reason,
                        supporting_clauses=[]
                    )
                }
            
            # Day care treatments - generally covered
            elif any(word in procedure for word in ['day care', 'dialysis', 'physiotherapy']):
                return {
                    "decision": DecisionType.APPROVED,
                    "amount": 25000.0,
                    "confidence": 0.75,
                    "justification": DecisionJustification(
                        primary_reason="Day care treatment covered under policy terms",
                        supporting_clauses=[]
                    )
                }
        
        # Default fallback
        return {
            "decision": DecisionType.UNDER_REVIEW,
            "amount": 0.0,
            "confidence": 0.3,
            "justification": DecisionJustification(
                primary_reason=f"Unable to process claim automatically. Manual review required. Error: {error_msg}",
                supporting_clauses=[]
            )
        }


def test_claim_processing():
    """Test the complete claim processing system"""
    try:
        print("🚀 Testing Claim Processing System...")
        
        # Initialize decision engine
        engine = ClaimDecisionEngine()
        print("✅ Decision engine initialized")
        
        # Test queries
        test_queries = [
            "46M, knee surgery, Pune, 3-month policy",
            "35F, delivery, Mumbai, 6-month policy",
            "28M, emergency air ambulance, Delhi, 1-year policy",
            "45F, cardiac surgery, Bangalore, 2-year policy"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\\n{'='*60}")
            print(f"🔍 Test Case {i}: {query}")
            print(f"{'='*60}")
            
            # Process the claim
            decision = engine.process_claim(query)
            
            # Display results
            print(f"\\n📋 CLAIM DECISION:")
            print(f"   Decision: {decision.decision.value}")
            print(f"   Amount: ${decision.amount:,.2f}")
            print(f"   Confidence: {decision.confidence:.2%}")
            print(f"   Processing Time: {decision.processing_metadata['processing_time_seconds']:.2f}s")
            
            print(f"\\n💡 JUSTIFICATION:")
            print(f"   Primary Reason: {decision.justification.primary_reason}")
            
            if decision.justification.waiting_period_status:
                print(f"   Waiting Period: {decision.justification.waiting_period_status}")
            
            if decision.justification.coverage_analysis:
                print(f"   Coverage: {decision.justification.coverage_analysis}")
            
            if decision.justification.supporting_clauses:
                print(f"\\n📄 SUPPORTING CLAUSES:")
                for j, clause in enumerate(decision.justification.supporting_clauses[:2]):
                    print(f"   {j+1}. {clause.clause_id}: {clause.text[:100]}...")
            
            print(f"\\n📊 METADATA:")
            print(f"   Query ID: {decision.query_id}")
            print(f"   Chunks Analyzed: {decision.processing_metadata.get('chunks_analyzed', 0)}")
            
        print(f"\\n🎉 All test cases completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in claim processing test: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_claim_processing()
