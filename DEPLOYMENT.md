# Vercel Deployment Guide for LLM Document Processing System

## Overview
This FastAPI application is optimized for deployment on Vercel with serverless functions.

## Project Structure
```
bajaj-2025/
├── vercel.json                 # Vercel configuration
├── api/                        # Vercel API directory
│   ├── main.py                 # FastAPI application
│   ├── requirements.txt        # Python dependencies
│   ├── general_document_processor.py
│   ├── config.py
│   ├── src/                    # Source modules
│   └── .env.example           # Environment variables template
└── README.md
```

## Deployment Steps

### 1. Environment Variables Setup
In Vercel dashboard, add these environment variables:

**Required:**
- `GOOGLE_API_KEY`: Your Google Gemini API key
- `PINECONE_API_KEY`: Your Pinecone API key  
- `PINECONE_ENVIRONMENT`: Your Pinecone environment
- `PINECONE_INDEX_NAME`: Your Pinecone index name

**Optional:**
- `DEFAULT_MODEL`: gemini-2.0-flash-exp
- `MAX_CHUNKS`: 5
- `LOG_LEVEL`: INFO

### 2. Deploy to Vercel

**Option A: Via Vercel CLI**
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy from project root
vercel

# Follow prompts to configure project
```

**Option B: Via GitHub Integration**
1. Push code to GitHub repository
2. Connect repository to Vercel dashboard
3. Configure environment variables
4. Deploy automatically

### 3. Verify Deployment

**Health Check:**
```
GET https://your-app.vercel.app/health
```

**API Documentation:**
```
GET https://your-app.vercel.app/docs
```

**Sample Request:**
```bash
curl -X POST "https://your-app.vercel.app/process-query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "46-year-old male, knee surgery in Pune, 3-month-old insurance policy",
    "query_id": "DEMO_001"
  }'
```

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /process-query` - Process document query
- `GET /query/{query_id}` - Retrieve specific query result
- `GET /queries` - List all processed queries
- `GET /stats` - Processing statistics
- `GET /docs` - Interactive API documentation

## Performance Optimizations

- **Max 5 chunks**: Limits vector search results for faster processing
- **Optimized LLM calls**: Reduced from 2-3 to 1-2 calls per query
- **Original text preservation**: Clause details include full original text
- **Caching**: In-memory storage for processed queries
- **Async processing**: FastAPI async support for better concurrency

## Monitoring

- Check `/health` endpoint for system status
- Use `/stats` endpoint for processing metrics
- Vercel dashboard provides deployment logs and analytics
- Response times average 15-20 seconds per query

## Troubleshooting

**Common Issues:**
1. **503 Service Unavailable**: Check environment variables configuration
2. **Timeout errors**: Vercel has 60-second timeout limit for serverless functions
3. **Import errors**: Ensure all dependencies are in requirements.txt
4. **Vector DB connection**: Verify Pinecone credentials and index exists

**Debug Steps:**
1. Check Vercel function logs in dashboard
2. Test `/health` endpoint first
3. Verify environment variables are set correctly
4. Test with simple query before complex ones

## Local Development

```bash
# Install dependencies
pip install -r api/requirements.txt

# Set environment variables
cp api/.env.example api/.env
# Edit .env with your API keys

# Run locally
cd api
python main.py

# Test at http://localhost:8000
```

## Production Considerations

- **Rate Limiting**: Consider implementing rate limiting for production
- **Authentication**: Add API key authentication for production use
- **CORS**: Configure specific origins instead of wildcard
- **Monitoring**: Set up error tracking and performance monitoring
- **Scaling**: Vercel automatically scales serverless functions
- **Cost**: Monitor API usage for Google Gemini and Pinecone

## Support

For issues:
1. Check Vercel deployment logs
2. Verify environment variables
3. Test API endpoints individually
4. Review error responses for specific issues
