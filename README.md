# ICICI Lombard Insurance Claim Processor API

## 🏥 Overview

A FastAPI-based REST API for processing insurance claims using LLM-enhanced decision making. The system provides intelligent claim analysis with 81.8% accuracy, focusing on binary APPROVED/REJECTED decisions.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
python main.py
```

The API will start on `http://localhost:8000`

### 3. Test the API

```bash
python test_client.py
```

## 📚 API Documentation

Once the server is running, visit:
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔗 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API information |
| `GET` | `/health` | Health check |
| `POST` | `/process-claim` | Process a new claim |
| `GET` | `/claim/{claim_id}` | Get specific claim |
| `GET` | `/claims` | List all claims |
| `GET` | `/stats` | Processing statistics |
| `DELETE` | `/claims` | Clear all claims |

### Process Claim

**POST** `/process-claim`

Process an insurance claim using natural language input.

**Request Body:**
```json
{
  "query": "45F, breast cancer, chemotherapy treatment, oncology, Delhi, Sum Insured 15 lakhs, 2-year policy",
  "claim_id": "CLAIM_2024_001"
}
```

**Response:**
```json
{
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
```

## 💻 Example Usage

### Using cURL

```bash
# Process a claim
curl -X POST "http://localhost:8000/process-claim" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "35M, road accident, emergency hospitalization, Mumbai, Sum Insured 5 lakhs, 6-month policy",
       "claim_id": "EMERGENCY_001"
     }'

# Get statistics
curl "http://localhost:8000/stats"

# List recent claims
curl "http://localhost:8000/claims?limit=5"
```

### Using Python Requests

```python
import requests

# Process a claim
response = requests.post("http://localhost:8000/process-claim", json={
    "query": "65F, cataract surgery, day care procedure, Sum Insured 5 lakhs, 2-year policy",
    "claim_id": "CATARACT_001"
})

result = response.json()
print(f"Decision: {result['decision']}")
print(f"Confidence: {result['confidence']}")
print(f"Reason: {result['primary_reason']}")
```

## 🎯 Features

- **LLM-Enhanced Processing**: Uses Google Gemini for intelligent claim analysis
- **High Accuracy**: 81.8% decision accuracy in testing
- **Binary Decisions**: Simple APPROVED/REJECTED outcomes
- **Fast Processing**: Average 25 second response time
- **RESTful API**: Standard HTTP methods and JSON
- **Interactive Documentation**: Swagger UI and ReDoc
- **Statistics**: Real-time processing metrics
- **Error Handling**: Comprehensive error responses

## 📊 Sample Test Cases

The system handles various claim types:

### ✅ Typically Approved
- Emergency procedures (road accidents, heart attacks)
- Covered surgeries after waiting periods
- Cancer treatments
- Day care procedures

### ❌ Typically Rejected
- Pre-existing conditions within waiting period
- Cosmetic procedures
- Excluded treatments
- Policy violations

## 🔧 Configuration

The system uses:
- **LLM Model**: Google Gemini 2.5 Flash
- **Vector Database**: Pinecone for policy knowledge
- **Policy Documents**: ICICI Lombard Golden Shield
- **Decision Framework**: Binary classification only

## 📈 Performance Metrics

- **Overall Accuracy**: 81.8% (18/22 test cases)
- **Emergency Cases**: 100% accuracy
- **Complex Medical**: 100% accuracy
- **Policy Edge Cases**: 100% accuracy
- **Average Processing**: 24.8 seconds

## 🛠️ Development

### Project Structure
```
├── main.py                    # FastAPI application
├── llm_enhanced_processor.py  # Core claim processor
├── test_client.py            # API test client
├── src/                      # Core modules
│   ├── vector_storage.py     # Vector database
│   ├── llm_analysis.py       # LLM integration
│   └── document_ingestion.py # Policy processing
└── requirements.txt          # Dependencies
```

### Environment Variables
Set up your `.env` file:
```
GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=insurance-policies
```

## 🚀 Deployment

For production deployment:

1. Use a production ASGI server
2. Configure CORS properly
3. Add authentication/authorization
4. Use a persistent database
5. Add monitoring and logging
6. Scale with load balancers

## 📞 Support

For issues or questions about the claim processor API, please check the interactive documentation at `/docs` or review the test client examples.
