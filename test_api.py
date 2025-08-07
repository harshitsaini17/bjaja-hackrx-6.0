#!/usr/bin/env python3
"""
Test script for the General Document Processing System
Demonstrates the API with various query types matching the problem statement
"""

import requests
import json
import time
from typing import Dict, Any

# API Configuration
API_BASE_URL = "http://localhost:8000"
PROCESS_ENDPOINT = f"{API_BASE_URL}/process-query"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

def test_api_health():
    """Test if the API is healthy"""
    print("🔍 Testing API Health...")
    try:
        response = requests.get(HEALTH_ENDPOINT)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is healthy: {data['status']}")
            print(f"   Processor ready: {data['processor_ready']}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {str(e)}")
        return False

def send_query(query: str, query_id: str = None) -> Dict[str, Any]:
    """Send a query to the document processing API"""
    payload = {
        "query": query,
        "query_id": query_id
    }
    
    print(f"\n📤 Sending query: {query}")
    print("   " + "="*50)
    
    try:
        response = requests.post(PROCESS_ENDPOINT, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ Decision: {result['decision']}")
            print(f"💰 Amount: {result.get('amount', 'N/A')}")
            print(f"📋 Justification: {result['justification']}")
            print(f"📖 Clause References: {result.get('clause_references', [])}")
            print(f"🎯 Confidence: {result['confidence']:.2f}")
            print(f"⏱️  Processing Time: {result['processing_time_seconds']:.2f}s")
            print(f"🆔 Query ID: {result['query_id']}")
            
            return result
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return {}
            
    except Exception as e:
        print(f"❌ Request error: {str(e)}")
        return {}

def run_test_suite():
    """Run a comprehensive test suite"""
    print("🚀 Starting LLM Document Processing System Test Suite")
    print("="*60)
    
    # Check API health first
    if not test_api_health():
        print("❌ API is not available. Please start the server first.")
        return
    
    # Test cases from problem statement
    test_cases = [
        {
            "name": "Insurance Claim - Knee Surgery",
            "query": "46-year-old male, knee surgery in Pune, 3-month-old insurance policy",
            "expected_type": "claim_processing"
        },
        {
            "name": "Insurance Claim - Cancer Treatment",
            "query": "55F, breast cancer chemotherapy treatment, Mumbai, 18-month policy, sum insured 10 lakhs",
            "expected_type": "claim_processing"
        },
        {
            "name": "Contract Review - Employment Terms",
            "query": "What are the termination clauses for remote employees in the employment contract?",
            "expected_type": "contract_review"
        },
        {
            "name": "Policy Inquiry - Maternity Benefits",
            "query": "Are maternity benefits covered for a 28-year-old female with 6-month policy?",
            "expected_type": "policy_inquiry"
        },
        {
            "name": "Email Analysis - Deadline Policy",
            "query": "Email mentions deadline for quarterly report submission, what's the company policy?",
            "expected_type": "email_analysis"
        },
        {
            "name": "Emergency Treatment",
            "query": "32M, emergency appendectomy surgery, Delhi, 2-week old policy",
            "expected_type": "claim_processing"
        },
        {
            "name": "Cosmetic Surgery",
            "query": "25F, cosmetic rhinoplasty surgery, Bangalore, 2-year policy",
            "expected_type": "claim_processing"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['name']}")
        
        result = send_query(
            query=test_case['query'],
            query_id=f"TEST_{i:02d}"
        )
        
        if result:
            results.append({
                "test_name": test_case['name'],
                "query": test_case['query'],
                "decision": result.get('decision'),
                "confidence": result.get('confidence'),
                "amount": result.get('amount'),
                "processing_time": result.get('processing_time_seconds')
            })
        
        # Small delay between requests
        time.sleep(1)
    
    # Summary
    print(f"\n📊 Test Summary")
    print("="*60)
    
    if results:
        approved = sum(1 for r in results if r['decision'] == 'approved')
        rejected = sum(1 for r in results if r['decision'] == 'rejected')
        conditional = sum(1 for r in results if r['decision'] == 'conditional')
        review_required = sum(1 for r in results if r['decision'] == 'review_required')
        
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        avg_processing_time = sum(r['processing_time'] for r in results) / len(results)
        
        print(f"✅ Approved: {approved}")
        print(f"❌ Rejected: {rejected}")
        print(f"⚠️  Conditional: {conditional}")
        print(f"🔍 Review Required: {review_required}")
        print(f"🎯 Average Confidence: {avg_confidence:.2f}")
        print(f"⏱️  Average Processing Time: {avg_processing_time:.2f}s")
        
        # Detailed results
        print(f"\n📋 Detailed Results:")
        for result in results:
            print(f"   {result['test_name']}: {result['decision']} (conf: {result['confidence']:.2f})")
    
    print(f"\n🎉 Test Suite Complete!")

def interactive_mode():
    """Interactive mode for manual testing"""
    print("\n🎮 Interactive Mode")
    print("="*30)
    print("Enter queries to test the system. Type 'quit' to exit.")
    
    if not test_api_health():
        print("❌ API is not available.")
        return
    
    query_counter = 1
    
    while True:
        query = input(f"\n[Query {query_counter}] Enter your query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not query:
            print("⚠️  Please enter a valid query.")
            continue
        
        send_query(query, f"INTERACTIVE_{query_counter:03d}")
        query_counter += 1

if __name__ == "__main__":
    print("LLM Document Processing System - Test Client")
    print("="*50)
    
    mode = input("Choose mode: (1) Test Suite (2) Interactive (3) Both [1]: ").strip()
    
    if mode in ['2', 'interactive']:
        interactive_mode()
    elif mode in ['3', 'both']:
        run_test_suite()
        interactive_mode()
    else:
        run_test_suite()
