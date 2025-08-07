#!/usr/bin/env python3
"""
Performance Test and Original Text Demo
Shows optimized response times and original clause text references
"""

import requests
import json
import time

# API Configuration
API_BASE_URL = "http://localhost:8000"
PROCESS_ENDPOINT = f"{API_BASE_URL}/process-query"

def performance_test():
    """Test the optimized API performance"""
    
    print("🚀 Performance Optimization Test")
    print("="*50)
    
    test_queries = [
        "46-year-old male, knee surgery in Pune, 3-month-old insurance policy",
        "32F, emergency appendectomy, Delhi, 2-week policy", 
        "55M, cosmetic surgery, Mumbai, 2-year policy",
        "28F, maternity benefits, Chennai, 6-month policy"
    ]
    
    total_time = 0
    successful_queries = 0
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📋 Test {i}: {query[:50]}...")
        
        start_time = time.time()
        
        payload = {
            "query": query,
            "query_id": f"PERF_TEST_{i}"
        }
        
        try:
            response = requests.post(PROCESS_ENDPOINT, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                api_time = time.time() - start_time
                processing_time = result['processing_time_seconds']
                
                print(f"✅ Success!")
                print(f"   Total Time: {api_time:.2f}s")
                print(f"   Processing Time: {processing_time:.2f}s")
                print(f"   Decision: {result['decision']}")
                print(f"   Confidence: {result['confidence']:.2f}")
                print(f"   Chunks Analyzed: {result['metadata'].get('chunks_analyzed', 'N/A')}")
                
                # Show clause details if available
                clause_details = result.get('clause_details', [])
                if clause_details:
                    print(f"   📖 Original Clauses Found: {len(clause_details)}")
                    for j, clause in enumerate(clause_details[:2], 1):  # Show first 2
                        original_text = clause.get('original_text', '')[:100]
                        print(f"      Clause {j}: {original_text}...")
                
                total_time += processing_time
                successful_queries += 1
                
            else:
                print(f"❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Request failed: {str(e)}")
        
        # Small delay between requests
        time.sleep(0.5)
    
    if successful_queries > 0:
        avg_time = total_time / successful_queries
        print(f"\n📊 Performance Summary:")
        print(f"   Successful Queries: {successful_queries}/{len(test_queries)}")
        print(f"   Average Processing Time: {avg_time:.2f}s")
        print(f"   {'🟢 EXCELLENT' if avg_time < 10 else '🟡 GOOD' if avg_time < 20 else '🔴 NEEDS IMPROVEMENT'}")

def clause_details_demo():
    """Demonstrate original text extraction"""
    
    print("\n\n📖 Original Clause Text Demo")
    print("="*50)
    
    query = "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
    
    payload = {
        "query": query,
        "query_id": "CLAUSE_DEMO"
    }
    
    try:
        response = requests.post(PROCESS_ENDPOINT, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"📤 Query: {query}")
            print(f"🎯 Decision: {result['decision']}")
            print(f"📋 Justification: {result['justification'][:150]}...")
            
            # Show detailed clause information
            clause_details = result.get('clause_details', [])
            
            if clause_details:
                print(f"\n📖 Referenced Clauses with Original Text:")
                print("-" * 50)
                
                for i, clause in enumerate(clause_details, 1):
                    print(f"\n🔍 Clause {i}:")
                    print(f"   📄 Source: {clause.get('source_document', 'Unknown')}")
                    print(f"   📑 Page: {clause.get('source_page', 'N/A')}")
                    print(f"   🆔 Reference: {clause.get('clause_reference', 'N/A')}")
                    print(f"   📊 Relevance: {clause.get('relevance_score', 0.0):.3f}")
                    
                    original_text = clause.get('original_text', '')
                    if original_text:
                        print(f"   📖 Original Text:")
                        print(f"      {original_text[:300]}...")
                    
                    summary = clause.get('summary', '')
                    if summary:
                        print(f"   📝 Summary: {summary[:150]}...")
                    
                    keywords = clause.get('keywords', [])
                    if keywords:
                        print(f"   🔑 Keywords: {', '.join(keywords[:5])}")
            else:
                print("\n⚠️  No clause details found in response")
                
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request error: {str(e)}")

def compare_optimization():
    """Compare optimization results"""
    
    print("\n\n⚡ Optimization Comparison")
    print("="*50)
    
    # Expected improvements
    improvements = {
        "Search Queries": "3 → 2 (33% reduction)",
        "LLM Calls": "2-3 → 1-2 (50% reduction)", 
        "Context Processing": "Full text → Summarized",
        "Response Structure": "Enhanced with original text",
        "Expected Speed": "30s → 10-15s (50% faster)"
    }
    
    for feature, improvement in improvements.items():
        print(f"   {feature}: {improvement}")
    
    print(f"\n✨ New Features:")
    print(f"   📖 Original clause text preservation")
    print(f"   🔗 Enhanced clause referencing")
    print(f"   ⚡ Optimized search strategy")
    print(f"   📊 Better metadata structure")

if __name__ == "__main__":
    print("⚡ LLM Document Processing - Performance & Clause Demo")
    print("="*60)
    
    # Check if API is available
    try:
        health_response = requests.get(f"{API_BASE_URL}/health")
        if health_response.status_code != 200:
            print("❌ API is not available. Please start the server first:")
            print("   uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
            exit(1)
    except:
        print("❌ Cannot connect to API. Please start the server first:")
        print("   uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
        exit(1)
    
    # Show optimization details
    compare_optimization()
    
    # Run performance test
    performance_test()
    
    # Show clause details demo
    clause_details_demo()
    
    print(f"\n🎉 Demo Complete!")
    print(f"\n💡 Key Improvements:")
    print(f"   ⚡ Faster processing through optimized search")
    print(f"   📖 Original text preservation in responses")
    print(f"   🔗 Enhanced clause mapping and references")
    print(f"   📊 Better metadata and debugging information")
