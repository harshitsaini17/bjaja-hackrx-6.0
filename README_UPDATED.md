# LLM Document Processing System

A general-purpose system that uses Large Language Models (LLMs) to process natural language queries and retrieve relevant information from large unstructured documents such as policy documents, contracts, and emails.

## 🎯 Objective

The system takes natural language queries and:

1. **Parses and structures** the query to identify key details
2. **Searches and retrieves** relevant clauses using semantic understanding (max 5 chunks)
3. **Evaluates** the information to determine decisions based on document content
4. **Returns structured responses** with decision, amount, justification, and clause mapping

## 🚀 Key Features

- **General Purpose**: Works with insurance policies, contracts, emails, and other documents
- **Semantic Search**: Uses vector embeddings for intelligent content retrieval
- **Limited Context**: Fetches only top 5 most relevant chunks for efficient processing
- **Structured Output**: Returns decisions with specific clause references
- **Multiple Document Types**: Handles PDFs, Word files, and text documents

## 📊 API Response Format

The system returns structured JSON responses matching the problem statement:

```json
{
    "decision": "approved|rejected|conditional|review_required",
    "amount": 150000.0,
    "justification": "Explanation with specific clause references",
    "clause_references": ["SEC_4_2", "WAIT_2_1", "GEOG_3_1"],
    "confidence": 0.92,
    "processing_time_seconds": 18.5,
    "query_id": "doc_query_1754568526",
    "timestamp": "2025-08-07T10:30:00Z",
    "metadata": {
        "chunks_analyzed": 5,
        "model_used": "gemini-2.5-flash",
        "query_type": "claim_processing"
    }
}
```

## 🔧 Setup and Installation

### Prerequisites
- Python 3.8+
- Google Gemini API Key
- Pinecone API Key

### Environment Variables
Create a `.env` file with:
```
GEMINI_API_KEY=your_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python main.py
```

The API will be available at `http://localhost:8000`

## 📚 API Endpoints

### Process Document Query
**POST** `/process-query`

Process a natural language query against documents.

**Request:**
```json
{
    "query": "46-year-old male, knee surgery in Pune, 3-month-old insurance policy",
    "query_id": "optional_external_id"
}
```

**Response:**
```json
{
    "decision": "approved",
    "amount": 150000.0,
    "justification": "Knee surgery is covered under surgical procedures...",
    "clause_references": ["SEC_4_2", "WAIT_2_1"],
    "confidence": 0.92,
    "processing_time_seconds": 18.5,
    "query_id": "doc_query_1754568526",
    "timestamp": "2025-08-07T10:30:00Z"
}
```

### Other Endpoints
- **GET** `/health` - API health check
- **GET** `/query/{query_id}` - Retrieve specific query result
- **GET** `/queries` - List all processed queries
- **GET** `/stats` - Processing statistics
- **DELETE** `/queries` - Clear all queries (testing)

## 🧪 Testing

### Run Test Suite
```bash
python test_api.py
```

### Interactive Testing
The test script includes an interactive mode for manual query testing.

### Sample Queries

1. **Insurance Claims:**
   - "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
   - "55F, breast cancer chemotherapy, Mumbai, 18-month policy, 10 lakh sum insured"

2. **Contract Reviews:**
   - "What are the termination clauses for remote employees in the employment contract?"
   - "Show me the non-compete agreement terms for senior managers"

3. **Policy Inquiries:**
   - "Are maternity benefits covered for a 28-year-old female with 6-month policy?"
   - "What's the waiting period for pre-existing conditions?"

4. **Email Analysis:**
   - "Email mentions deadline for quarterly reports, what's the company policy?"
   - "Client email about refund request, what are our terms?"

## 🏗️ Architecture

### Components

1. **GeneralDocumentProcessor** - Main processing engine
2. **VectorStoreManager** - Pinecone vector database integration
3. **Query Parser** - LLM-powered query understanding
4. **Content Retrieval** - Semantic search with max 5 chunks
5. **Decision Engine** - LLM-based decision making

### Processing Pipeline

1. **Query Parsing**: Extract entities, intent, and context
2. **Search Strategy**: Generate targeted search queries
3. **Content Retrieval**: Fetch top 5 relevant document chunks
4. **Decision Analysis**: Analyze content and make structured decisions
5. **Response Generation**: Format results with clause references

## 🎯 Applications

- **Insurance**: Claim processing, policy inquiries, coverage verification
- **Legal**: Contract analysis, compliance checking, clause extraction
- **HR**: Policy questions, employee handbook queries, benefits information
- **Finance**: Terms verification, limit checking, approval workflows

## ⚙️ Configuration

Key settings in `config.py`:

```python
# Search Configuration - Limited to max 5 chunks
search_top_k: int = 5
similarity_threshold: float = 0.7

# LLM Configuration
gemini_model: str = "gemini-2.5-flash"
gemini_temperature: float = 0.1

# Document Processing
chunk_size: int = 1000
chunk_overlap: int = 200
```

## 🔍 Example Use Cases

### Insurance Claim Processing
**Query:** "46M, knee surgery, Pune, 3-month policy"
**Response:** "Yes, knee surgery is covered under the policy."

### Contract Review
**Query:** "Termination clause for remote workers"
**Response:** Analysis of employment termination terms with specific clause references.

### Policy Verification
**Query:** "Maternity benefits, 28F, 6-month policy"
**Response:** Coverage determination based on policy terms and waiting periods.

## 📈 Performance

- **Max Context**: 5 document chunks per query
- **Average Response Time**: 15-25 seconds
- **Confidence Scoring**: 0.0-1.0 scale
- **Supported Documents**: PDF, DOCX, TXT

## 🔒 Security

- API key authentication for LLM services
- Input validation and sanitization
- Error handling and fallback mechanisms
- Configurable CORS settings

## 📝 API Documentation

Interactive API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🤝 Contributing

This system is designed to be extensible for various document types and use cases. Key areas for enhancement:

- Additional document format support
- Custom prompt templates for specific domains
- Enhanced caching mechanisms
- Batch processing capabilities

## 📄 License

This project is part of the Bajaj 2025 Hackathon submission.
