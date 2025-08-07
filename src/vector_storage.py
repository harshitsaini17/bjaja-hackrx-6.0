"""
Vector embedding and storage system using Pinecone
Handles embedding generation and vector database operations
"""

import os
import sys
import json
import logging
import time
import traceback
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict
import uuid

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

# Vector database imports
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Import our analysis modules
from .llm_analysis import IntelligentChunk, AnalyzedContent, GeminiContentAnalyzer
from .document_ingestion import ProcessedDocument, DocumentIngestionSystem

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages vector embeddings and Pinecone operations"""
    
    def __init__(self):
        """Initialize Pinecone and embedding models"""
        if not settings.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=settings.gemini_api_key
        )
        
        # Index configuration
        self.index_name = settings.pinecone_index_name
        self.dimension = settings.pinecone_dimension
        
        # Initialize or connect to index
        self._setup_index()
        
        logger.info("VectorStoreManager initialized successfully")
    
    def _setup_index(self):
        """Setup or connect to Pinecone index"""
        try:
            # Check if index exists
            existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                
                # Create index with serverless spec
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=settings.pinecone_metric,
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                
                # Wait for index to be ready
                time.sleep(10)
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
            # Get index stats
            stats = self.index.describe_index_stats()
            logger.info(f"Index stats: {stats}")
            
        except Exception as e:
            logger.error(f"Error setting up Pinecone index: {str(e)}")
            raise
    
    def generate_embedding(self, text: str, max_retries: int = 3) -> List[float]:
        """Generate embedding for text using Gemini with retry logic"""
        for attempt in range(max_retries):
            try:
                embedding = self.embeddings.embed_query(text)
                
                # Validate embedding dimensions
                if len(embedding) != self.dimension:
                    raise ValueError(f"Embedding dimension mismatch: expected {self.dimension}, got {len(embedding)}")
                
                return embedding
                
            except Exception as e:
                logger.warning(f"Embedding generation attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to generate embedding after {max_retries} attempts: {str(e)}")
                    raise
                
                # Wait before retry with exponential backoff
                wait_time = (2 ** attempt) + 1
                time.sleep(wait_time)
    
    def store_chunk(self, chunk: IntelligentChunk, max_retries: int = 3) -> str:
        """Store a single chunk in Pinecone with retry logic"""
        for attempt in range(max_retries):
            try:
                # Generate embedding for chunk content
                content_for_embedding = f"""
                {chunk.content}
                
                Summary: {chunk.summary}
                Keywords: {', '.join(chunk.keywords)}
                Entities: {', '.join(chunk.entities)}
                Type: {chunk.chunk_type.value}
                """
                
                embedding = self.generate_embedding(content_for_embedding)
                
                # Prepare metadata with size limits
                metadata = {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content[:2000] if len(chunk.content) > 2000 else chunk.content,
                    "chunk_type": chunk.chunk_type.value,
                    "source_page": chunk.source_page,
                    "source_document": os.path.basename(chunk.source_document),
                    "keywords": chunk.keywords[:10],
                    "entities": chunk.entities[:10], 
                    "summary": chunk.summary[:500] if len(chunk.summary) > 500 else chunk.summary,
                    "importance_score": chunk.importance_score,
                    "creation_timestamp": chunk.metadata.get("creation_timestamp", time.time()),
                    "relationships": chunk.relationships[:5]
                }
                
                # Upsert to Pinecone with timeout
                vector_id = chunk.chunk_id
                self.index.upsert(
                    vectors=[(vector_id, embedding, metadata)],
                    timeout=30  # 30 second timeout
                )
                
                logger.debug(f"Successfully stored chunk {chunk.chunk_id} in Pinecone")
                return vector_id
                
            except Exception as e:
                logger.warning(f"Store chunk attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to store chunk {chunk.chunk_id} after {max_retries} attempts: {str(e)}")
                    raise
                
                # Wait before retry
                wait_time = (2 ** attempt) + 1
                time.sleep(wait_time)
    
    def store_chunks_batch(self, chunks: List[IntelligentChunk], batch_size: int = 100) -> List[str]:
        """Store multiple chunks in batches"""
        stored_ids = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_vectors = []
            
            for chunk in batch:
                try:
                    # Generate embedding
                    content_for_embedding = f"""
                    {chunk.content}
                    
                    Summary: {chunk.summary}
                    Keywords: {', '.join(chunk.keywords)}
                    Entities: {', '.join(chunk.entities)}
                    Type: {chunk.chunk_type.value}
                    """
                    
                    embedding = self.generate_embedding(content_for_embedding)
                    
                    # Prepare metadata with original text preservation
                    metadata = {
                        "chunk_id": chunk.chunk_id,
                        "content": chunk.content[:1000],  # Processed/summarized content
                        "original_text": chunk.content,  # Store full original text
                        "chunk_type": chunk.chunk_type.value,
                        "source_page": chunk.source_page,
                        "source_document": os.path.basename(chunk.source_document),
                        "keywords": chunk.keywords[:10],
                        "entities": chunk.entities[:10],
                        "summary": chunk.summary[:500],
                        "importance_score": chunk.importance_score,
                        "creation_timestamp": chunk.metadata.get("creation_timestamp", time.time()),
                        "relationships": chunk.relationships[:5],
                        "clause_reference": f"{os.path.basename(chunk.source_document)}_PAGE_{chunk.source_page}_CHUNK_{chunk.chunk_id}"
                    }
                    
                    batch_vectors.append((chunk.chunk_id, embedding, metadata))
                    
                except Exception as e:
                    logger.error(f"Error preparing chunk {chunk.chunk_id} for batch: {str(e)}")
                    continue
            
            # Upsert batch
            if batch_vectors:
                try:
                    self.index.upsert(vectors=batch_vectors)
                    stored_ids.extend([vec[0] for vec in batch_vectors])
                    logger.info(f"Stored batch of {len(batch_vectors)} chunks")
                except Exception as e:
                    logger.error(f"Error storing batch: {str(e)}")
        
        return stored_ids
    
    def search_similar(self, query: str, top_k: int = None, filter_dict: Dict = None) -> List[Dict]:
        """Search for similar chunks in Pinecone with max 5 results"""
        if top_k is None:
            top_k = min(settings.search_top_k, 5)  # Ensure max 5 chunks
        else:
            top_k = min(top_k, 5)  # Enforce max 5 limit
        
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            
            # Search in Pinecone
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Process results
            results = []
            for match in search_results['matches']:
                result = {
                    "id": match['id'],
                    "score": match['score'],
                    "metadata": match['metadata']
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} similar chunks for query (max 5 enforced)")
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {str(e)}")
            return []
    
    def delete_by_source(self, source_document: str):
        """Delete all chunks from a specific source document"""
        try:
            # Query all vectors from the source
            filter_dict = {"source_document": os.path.basename(source_document)}
            
            # Get all IDs (Pinecone doesn't support delete by filter directly)
            all_results = self.index.query(
                vector=[0.0] * self.dimension,  # Dummy vector
                top_k=10000,  # Large number to get all
                include_metadata=True,
                filter=filter_dict
            )
            
            # Extract IDs and delete
            ids_to_delete = [match['id'] for match in all_results['matches']]
            
            if ids_to_delete:
                self.index.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} chunks from {source_document}")
            
        except Exception as e:
            logger.error(f"Error deleting chunks from {source_document}: {str(e)}")
    
    def get_index_stats(self) -> Dict:
        """Get current index statistics"""
        try:
            stats = self.index.describe_index_stats()
            # Convert IndexDescription to serializable dict
            namespaces_dict = {}
            for ns_name, ns_summary in stats.namespaces.items():
                namespaces_dict[ns_name] = {
                    'vector_count': ns_summary.vector_count
                }
            
            return {
                'dimension': stats.dimension,
                'index_fullness': stats.index_fullness,
                'metric': stats.metric,
                'namespaces': namespaces_dict,
                'total_vector_count': stats.total_vector_count,
                'vector_type': stats.vector_type
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {}


class DocumentProcessingPipeline:
    """Complete pipeline for processing documents and storing in vector database"""
    
    def __init__(self):
        """Initialize the complete pipeline"""
        self.ingestion_system = DocumentIngestionSystem()
        self.analyzer = GeminiContentAnalyzer()
        self.vector_store = VectorStoreManager()
        
        logger.info("DocumentProcessingPipeline initialized")
    
    def process_document_to_vectors(self, file_path: str) -> Dict[str, Any]:
        """Complete pipeline: document -> analysis -> chunks -> vectors"""
        start_time = time.time()
        
        try:
            logger.info(f"Starting complete processing for: {file_path}")
            
            # Step 1: Document Ingestion
            processed_doc = self.ingestion_system.process_document(file_path)
            logger.info(f"Ingested {processed_doc.total_pages} pages")
            
            # Step 2: Analyze pages and create chunks
            all_chunks = []
            total_chunks_created = 0
            all_stored_ids = []
            
            for page in processed_doc.pages:  # Process all pages
                logger.info(f"Analyzing page {page.page_number}")
                
                # Analyze page content
                analyzed_content = self.analyzer.analyze_page_content(page)
                
                # Create intelligent chunks
                chunks = self.analyzer.create_intelligent_chunks(analyzed_content)
                all_chunks.extend(chunks)
                total_chunks_created += len(chunks)
                
                logger.info(f"Created {len(chunks)} chunks from page {page.page_number}")
                
                # Process in batches to avoid memory issues
                if len(all_chunks) >= 50:  # Store in batches of 50 chunks
                    logger.info(f"Storing batch of {len(all_chunks)} chunks...")
                    batch_stored_ids = self.vector_store.store_chunks_batch(all_chunks)
                    all_stored_ids.extend(batch_stored_ids)
                    all_chunks = []  # Reset for next batch
            
            # Step 3: Store remaining chunks in vector database
            if all_chunks:  # Store any remaining chunks
                logger.info(f"Storing final batch of {len(all_chunks)} chunks...")
                final_stored_ids = self.vector_store.store_chunks_batch(all_chunks)
                all_stored_ids.extend(final_stored_ids)
            
            # Calculate processing stats
            processing_time = time.time() - start_time
            
            result = {
                "status": "success",
                "document_id": processed_doc.document_id,
                "total_pages": processed_doc.total_pages,
                "pages_processed": len(processed_doc.pages),
                "chunks_created": total_chunks_created,
                "chunks_stored": len(all_stored_ids),
                "processing_time_seconds": processing_time,
                "stored_chunk_ids": all_stored_ids
            }
            
            logger.info(f"Complete processing finished: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in complete document processing: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time_seconds": time.time() - start_time
            }


def test_vector_storage():
    """Test the vector storage system"""
    try:
        print("🚀 Testing Vector Storage System...")
        
        # Initialize pipeline
        pipeline = DocumentProcessingPipeline()
        print("✅ Pipeline initialized successfully")
        
        # Test with sample document
        sample_file = "sample_data/EDLHLGA23009V012223.pdf"
        
        if os.path.exists(sample_file):
            print(f"📄 Processing {sample_file} through complete pipeline...")
            
            result = pipeline.process_document_to_vectors(sample_file)
            
            if result["status"] == "success":
                print(f"✅ Complete pipeline successful!")
                print(f"   Pages processed: {result['pages_processed']}")
                print(f"   Chunks created: {result['chunks_created']}")
                print(f"   Chunks stored: {result['chunks_stored']}")
                print(f"   Processing time: {result['processing_time_seconds']:.2f}s")
                
                # Test search functionality
                print(f"\\n🔍 Testing search functionality...")
                query = "air ambulance coverage emergency"
                search_results = pipeline.vector_store.search_similar(query, top_k=3)
                
                print(f"✅ Found {len(search_results)} results for query: '{query}'")
                for i, result in enumerate(search_results):
                    print(f"   Result {i+1}: Score {result['score']:.3f}")
                    print(f"   Content: {result['metadata']['content'][:100]}...")
                    print(f"   Type: {result['metadata']['chunk_type']}")
                
                # Get index stats
                stats = pipeline.vector_store.get_index_stats()
                print(f"\\n📊 Index Statistics:")
                print(f"   Total vectors: {stats.get('total_vector_count', 0)}")
                print(f"   Dimension: {stats.get('dimension', 0)}")
                
            else:
                print(f"❌ Pipeline failed: {result['error']}")
        
        else:
            print(f"❌ Sample file not found: {sample_file}")
    
    except Exception as e:
        print(f"❌ Error in vector storage testing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_vector_storage()
