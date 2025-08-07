#!/usr/bin/env python3
"""
Demo script showing the exact problem statement example
Demonstrates the API matching the required format
"""

import requests
import json

# API Configuration
API_BASE_URL = "http://localhost:8000"
PROCESS_ENDPOINT = f"{API_BASE_URL}/process-query"

def demo_problem_statement_example():
    """
    Demonstrate the exact example from the problem statement:
    
    Input: "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
    Expected: "Yes, knee surgery is covered under the policy."
    """
    
    print("🎯 Problem Statement Demo")
    print("="*50)
    
    # The exact query from problem statement
    query = "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
    
    print(f"📤 Input Query: {query}")
    print("\n🔄 Processing...")
    
    payload = {
        "query": query,
        "query_id": "PROBLEM_STATEMENT_DEMO"
    }
    
    try:
        response = requests.post(PROCESS_ENDPOINT, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n✅ API Response (Structured JSON):")
            print("="*50)
            print(json.dumps(result, indent=2))
            
            print(f"\n📋 Summary Response:")
            print("="*30)
            print(f"Decision: {result['decision']}")
            print(f"Amount: {result.get('amount', 'N/A')}")
            print(f"Justification: {result['justification']}")
            
            # Show the simple format mentioned in problem statement
            if result['decision'] == 'approved':
                simple_response = "Yes, knee surgery is covered under the policy."
            else:
                simple_response = "No, knee surgery is not covered under the policy."
                
            print(f"\n🎯 Simple Response (as in problem statement):")
            print(f'"{simple_response}"')
            
            print(f"\n📖 Clause References:")
            for clause in result.get('clause_references', []):
                print(f"   - {clause}")
                
            print(f"\n📊 Processing Details:")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Processing Time: {result['processing_time_seconds']:.2f}s")
            print(f"   Chunks Analyzed: {result['metadata'].get('chunks_analyzed', 'N/A')}")
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Connection error: {str(e)}")
        print("\n💡 Make sure the API server is running:")
        print("   python main.py")

def demo_multiple_examples():
    """Show multiple examples matching the problem statement format"""
    
    examples = [
        {
            "input": "46-year-old male, knee surgery in Pune, 3-month-old insurance policy",
            "expected_pattern": "knee surgery coverage"
        },
        {
            "input": "32F, emergency appendectomy, Delhi, 2-week policy",
            "expected_pattern": "emergency surgery coverage"
        },
        {
            "input": "55M, cosmetic surgery, Mumbai, 2-year policy",
            "expected_pattern": "cosmetic surgery exclusion"
        }
    ]
    
    print("\n🎯 Multiple Demo Examples")
    print("="*50)
    
    for i, example in enumerate(examples, 1):
        print(f"\n📋 Example {i}:")
        print(f"Input: {example['input']}")
        
        payload = {
            "query": example['input'],
            "query_id": f"DEMO_{i}"
        }
        
        try:
            response = requests.post(PROCESS_ENDPOINT, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                # Generate simple response
                decision = result['decision']
                justification = result['justification']
                
                if decision == 'approved':
                    simple_response = f"Yes, {example['expected_pattern']} is covered."
                elif decision == 'rejected':
                    simple_response = f"No, {example['expected_pattern']} is not covered."
                else:
                    simple_response = f"Coverage for {example['expected_pattern']} requires review."
                
                print(f"Output: \"{simple_response}\"")
                print(f"Details: {justification[:100]}...")
                print(f"Confidence: {result['confidence']:.2f}")
                
            else:
                print(f"Error: {response.status_code}")
                
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    print("LLM Document Processing System - Problem Statement Demo")
    print("="*60)
    
    # Check if API is available
    try:
        health_response = requests.get(f"{API_BASE_URL}/health")
        if health_response.status_code != 200:
            print("❌ API is not available. Please start the server first:")
            print("   python main.py")
            exit(1)
    except:
        print("❌ Cannot connect to API. Please start the server first:")
        print("   python main.py")
        exit(1)
    
    # Run the exact problem statement demo
    demo_problem_statement_example()
    
    # Ask if user wants to see more examples
    more_examples = input(f"\n🤔 Show more examples? (y/n) [y]: ").strip().lower()
    if more_examples != 'n':
        demo_multiple_examples()
    
    print(f"\n🎉 Demo Complete!")
    print(f"\n💡 For interactive testing, run: python test_api.py")
    print(f"📚 For API docs, visit: http://localhost:8000/docs")
