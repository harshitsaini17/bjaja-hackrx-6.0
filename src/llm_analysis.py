"""
LLM-based content analysis using Gemini
Handles page-level content analysis, table processing, and intelligent chunking
"""

import os
import sys
import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

# LLM imports
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import our document processing module
from .document_ingestion import DocumentPage, ProcessedDocument

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Types of content in insurance policy documents"""
    CLAUSE = "clause"
    TABLE = "table"
    EXCLUSION = "exclusion"
    WAITING_PERIOD = "waiting_period"
    COVERAGE = "coverage"
    PREMIUM = "premium"
    CLAIM_PROCESS = "claim_process"
    ELIGIBILITY = "eligibility"
    DEFINITIONS = "definitions"
    CONTACT_INFORMATION = "contact_information"
    DEFINITION = "definition"
    POLICY_PREAMBLE = "policy_preamble"
    POLICY_SCHEDULE = "policy_schedule"
    POLICY_SCOPE = "policy_scope"
    TERMS_CONDITIONS = "terms_conditions"
    EXAMPLE = "example"
    CLAIM_CALCULATION_ILLUSTRATION = "claim_calculation_illustration"
    OFFICE_JURISDICTION_DETAILS = "office_jurisdiction_details"
    BENEFITS = "benefits"
    SUB_LIMITS = "sub_limits"
    OTHER = "other"


@dataclass
class AnalyzedContent:
    """Represents analyzed content with LLM insights"""
    original_content: str
    content_type: ContentType
    keywords: List[str]
    entities: List[Dict[str, Any]]
    summary: str
    importance_score: float
    medical_terms: List[str]
    policy_terms: List[str]
    metadata: Dict[str, Any]


@dataclass
class IntelligentChunk:
    """Represents an intelligently created chunk"""
    chunk_id: str
    content: str
    chunk_type: ContentType
    source_page: int
    source_document: str
    keywords: List[str]
    entities: List[str]
    summary: str
    importance_score: float
    metadata: Dict[str, Any]
    relationships: List[str]  # Related chunk IDs


class GeminiContentAnalyzer:
    """Handles LLM-based content analysis using Gemini"""
    
    def __init__(self):
        """Initialize Gemini API and configure models"""
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=settings.gemini_api_key)
        
        # Initialize models
        self.llm = GoogleGenerativeAI(
            model=settings.gemini_model,
            temperature=settings.gemini_temperature,
            max_tokens=settings.gemini_max_tokens,
            google_api_key=settings.gemini_api_key
        )
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=settings.gemini_api_key
        )
        
        # Text splitter for intelligent chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        logger.info("GeminiContentAnalyzer initialized successfully")
    
    def analyze_page_content(self, page: DocumentPage) -> AnalyzedContent:
        """Analyze a single page using Gemini for content insights"""
        
        prompt = f"""
        Analyze this insurance policy page content and provide structured insights.
        
        PAGE CONTENT:
        {page.content[:2000]}
        
        Please respond with ONLY a valid JSON object in this exact format:
        {{
            "content_type": "clause",
            "keywords": ["keyword1", "keyword2", "keyword3"],
            "medical_terms": ["term1", "term2"],
            "policy_terms": ["term1", "term2"],
            "entities": [{{"type": "amount", "value": "50000", "context": "coverage limit"}}],
            "summary": "Brief summary of the page content",
            "importance_score": 0.8
        }}
        
        Content types: clause, table, exclusion, waiting_period, coverage, premium, claim_process, eligibility, definitions, other
        """
        
        try:
            response = self.llm.invoke(prompt)
            
            # Clean the response to extract JSON
            response_text = str(response).strip()
            
            # Try to find JSON content
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_text = response_text[start_idx:end_idx]
                analysis_data = json.loads(json_text)
            else:
                raise ValueError("No JSON found in response")
            
            return AnalyzedContent(
                original_content=page.content,
                content_type=ContentType(analysis_data.get("content_type", "other")),
                keywords=analysis_data.get("keywords", []),
                entities=analysis_data.get("entities", []),
                summary=analysis_data.get("summary", ""),
                importance_score=analysis_data.get("importance_score", 0.5),
                medical_terms=analysis_data.get("medical_terms", []),
                policy_terms=analysis_data.get("policy_terms", []),
                metadata={
                    "page_number": page.page_number,
                    "source_document": page.document_source,
                    "analysis_timestamp": time.time(),
                    "has_tables": page.metadata.get("has_tables", False)
                }
            )
            
        except Exception as e:
            logger.error(f"Error analyzing page {page.page_number}: {str(e)}")
            return self._fallback_analysis(page)
    
    def _fallback_analysis(self, page: DocumentPage) -> AnalyzedContent:
        """Fallback analysis when LLM fails"""
        content_lower = page.content.lower()
        
        # Simple keyword-based content type detection
        if any(word in content_lower for word in ["exclusion", "excluded", "not covered"]):
            content_type = ContentType.EXCLUSION
        elif any(word in content_lower for word in ["waiting period", "wait", "months after"]):
            content_type = ContentType.WAITING_PERIOD
        elif any(word in content_lower for word in ["coverage", "covered", "benefit"]):
            content_type = ContentType.COVERAGE
        elif "table" in content_lower or page.metadata.get("has_tables", False):
            content_type = ContentType.TABLE
        else:
            content_type = ContentType.CLAUSE
        
        return AnalyzedContent(
            original_content=page.content,
            content_type=content_type,
            keywords=content_lower.split()[:10],  # First 10 words as keywords
            entities=[],
            summary="Content analysis failed, fallback used",
            importance_score=0.5,
            medical_terms=[],
            policy_terms=[],
            metadata={
                "page_number": page.page_number,
                "source_document": page.document_source,
                "fallback_used": True
            }
        )
    
    def process_table_content(self, table_data: List[List[str]], page_number: int) -> str:
        """Convert table data to searchable natural language using Gemini"""
        
        # Convert table to text representation
        table_text = "\\n".join([" | ".join(row) for row in table_data])
        
        prompt = f"""
        Convert this insurance policy table to searchable natural language while preserving exact data relationships:
        
        TABLE DATA:
        {table_text[:2000]}  # Limit for API
        
        REQUIREMENTS:
        1. Preserve all numerical values and conditions exactly
        2. Convert to natural language descriptions
        3. Maintain policy term relationships
        4. Make it searchable for claim processing queries
        5. Keep clause references and cross-links intact
        
        Generate a natural language description that maintains all the critical information for insurance claim processing.
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            logger.error(f"Error processing table on page {page_number}: {str(e)}")
            # Fallback: return formatted table text
            return f"Table content: {table_text}"
    
    def create_intelligent_chunks(self, analyzed_content: AnalyzedContent) -> List[IntelligentChunk]:
        """Create intelligent chunks using Gemini for context-aware chunking"""
        
        prompt = f"""
        Create intelligent chunks from this insurance content for optimal claim processing.
        
        CONTENT TYPE: {analyzed_content.content_type.value}
        CONTENT: {analyzed_content.original_content[:1500]}
        
        Please respond with ONLY a valid JSON object in this exact format:
        {{
            "chunks": [
                {{
                    "content": "chunk content here",
                    "chunk_type": "clause",
                    "keywords": ["keyword1", "keyword2"],
                    "entities": ["entity1", "entity2"],
                    "summary": "chunk summary",
                    "importance_score": 0.8,
                    "relationships": ["related concept 1"]
                }}
            ]
        }}
        
        Create 2-4 semantic chunks maximum. Each chunk should be self-contained.
        """
        
        try:
            response = self.llm.invoke(prompt)
            
            # Clean the response to extract JSON
            response_text = str(response).strip()
            
            # Try to find JSON content
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_text = response_text[start_idx:end_idx]
                chunk_data = json.loads(json_text)
            else:
                raise ValueError("No JSON found in response")
            
            chunks = []
            for i, chunk_info in enumerate(chunk_data.get("chunks", [])):
                chunk_id = f"{analyzed_content.metadata['page_number']}_{i}_{int(time.time())}"
                
                chunk = IntelligentChunk(
                    chunk_id=chunk_id,
                    content=chunk_info.get("content", ""),
                    chunk_type=ContentType(chunk_info.get("chunk_type", "other")),
                    source_page=analyzed_content.metadata["page_number"],
                    source_document=analyzed_content.metadata["source_document"],
                    keywords=chunk_info.get("keywords", []),
                    entities=chunk_info.get("entities", []),
                    summary=chunk_info.get("summary", ""),
                    importance_score=chunk_info.get("importance_score", 0.5),
                    metadata={
                        "original_content_type": analyzed_content.content_type.value,
                        "creation_timestamp": time.time()
                    },
                    relationships=chunk_info.get("relationships", [])
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating intelligent chunks: {str(e)}")
            return self._fallback_chunking(analyzed_content)
    
    def _fallback_chunking(self, analyzed_content: AnalyzedContent) -> List[IntelligentChunk]:
        """Fallback chunking using text splitter"""
        text_chunks = self.text_splitter.split_text(analyzed_content.original_content)
        
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk_id = f"{analyzed_content.metadata['page_number']}_{i}_{int(time.time())}"
            
            chunk = IntelligentChunk(
                chunk_id=chunk_id,
                content=chunk_text,
                chunk_type=analyzed_content.content_type,
                source_page=analyzed_content.metadata["page_number"],
                source_document=analyzed_content.metadata["source_document"],
                keywords=analyzed_content.keywords[:5],  # Use top keywords
                entities=[],
                summary=analyzed_content.summary,
                importance_score=analyzed_content.importance_score,
                metadata={
                    "fallback_chunking": True,
                    "creation_timestamp": time.time()
                },
                relationships=[]
            )
            chunks.append(chunk)
        
        return chunks


def test_gemini_analysis():
    """Test the Gemini content analysis system"""
    try:
        # Initialize analyzer
        analyzer = GeminiContentAnalyzer()
        
        print("✅ GeminiContentAnalyzer initialized successfully")
        
        # Test with a simple page (using first sample document)
        from document_ingestion import DocumentIngestionSystem
        
        ingestion_system = DocumentIngestionSystem()
        sample_file = "sample_data/EDLHLGA23009V012223.pdf"  # Smallest file for testing
        
        if os.path.exists(sample_file):
            print(f"📄 Processing {sample_file} for analysis testing...")
            
            processed_doc = ingestion_system.process_document(sample_file)
            
            # Analyze first page
            if processed_doc.pages:
                first_page = processed_doc.pages[0]
                print(f"🔍 Analyzing page 1 content (first 200 chars): {first_page.content[:200]}...")
                
                analyzed_content = analyzer.analyze_page_content(first_page)
                
                print(f"✅ Page analysis complete:")
                print(f"   Content Type: {analyzed_content.content_type.value}")
                print(f"   Keywords: {analyzed_content.keywords[:5]}")
                print(f"   Summary: {analyzed_content.summary}")
                print(f"   Importance Score: {analyzed_content.importance_score}")
                
                # Test intelligent chunking
                print(f"🧩 Creating intelligent chunks...")
                chunks = analyzer.create_intelligent_chunks(analyzed_content)
                
                print(f"✅ Created {len(chunks)} intelligent chunks:")
                for i, chunk in enumerate(chunks):
                    print(f"   Chunk {i+1}: {chunk.summary[:100]}...")
                    print(f"   Keywords: {chunk.keywords[:3]}")
                    print(f"   Type: {chunk.chunk_type.value}")
                
        else:
            print(f"❌ Sample file not found: {sample_file}")
            
    except Exception as e:
        print(f"❌ Error in Gemini analysis testing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_gemini_analysis()
