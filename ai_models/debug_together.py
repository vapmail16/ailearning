import os
import together
from dotenv import load_dotenv
import json
import time

def check_response_structure(response):
    """Validate Together API response structure"""
    checks = {
        "Has choices": False,
        "Has valid text": False,
        "Has usage info": False,
        "Text is non-empty": False
    }
    
    try:
        # Check basic structure
        if isinstance(response, dict) and 'choices' in response:
            checks["Has choices"] = True
            if len(response['choices']) > 0 and 'text' in response['choices'][0]:
                checks["Has valid text"] = True
                if response['choices'][0]['text'].strip():
                    checks["Text is non-empty"] = True
        
        # Check usage information
        if 'usage' in response and 'total_tokens' in response['usage']:
            checks["Has usage info"] = True
            
        return checks
    except Exception as e:
        print(f"Error in structure check: {str(e)}")
        return checks

def test_different_prompts():
    print("\nTesting different prompts...")
    prompts = [
        "Say hello",
        "Explain quantum computing in simple terms",
        "Write a short poem about AI"
    ]
    
    all_passed = True
    for prompt in prompts:
        print(f"\nTesting prompt: {prompt}")
        try:
            response = together.Complete.create(
                prompt=prompt,
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                max_tokens=128,
                temperature=0.7
            )
            
            # Validate response
            checks = check_response_structure(response)
            all_checks_passed = all(checks.values())
            
            print(f"Response validation:")
            for check, passed in checks.items():
                print(f"  {check}: {'✅ PASS' if passed else '❌ FAIL'}")
            
            if all_checks_passed:
                print(f"✅ PASS - First few words: {response['choices'][0]['text'][:50]}...")
            else:
                print("❌ FAIL - Invalid response structure")
                all_passed = False
                
        except Exception as e:
            print(f"❌ FAIL - Error: {str(e)}")
            all_passed = False
    
    return all_passed

def test_parameters():
    print("\nTesting different parameters...")
    test_params = [
        {'temperature': 0.1, 'max_tokens': 50},
        {'temperature': 0.7, 'max_tokens': 100},
        {'temperature': 1.0, 'max_tokens': 200}
    ]
    
    all_passed = True
    for params in test_params:
        print(f"\nTesting with parameters: {params}")
        try:
            response = together.Complete.create(
                prompt="Tell me a story",
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                **params
            )
            
            # Validate response length matches requested max_tokens
            actual_length = len(response['choices'][0]['text'].split())
            checks = check_response_structure(response)
            
            if all(checks.values()):
                print(f"✅ PASS - Response length: {actual_length} words")
            else:
                print(f"❌ FAIL - Response validation failed")
                all_passed = False
                
        except Exception as e:
            print(f"❌ FAIL - Error: {str(e)}")
            all_passed = False
    
    return all_passed

def debug_together_api():
    """Debug Together AI API integration"""
    try:
        # Initialize Together AI
        together.api_key = os.getenv('TOGETHER_API_KEY')
        print(f"API Key configured: {'✓' if together.api_key else '✗'}")

        # Test basic completion
        print("\n1. Testing basic completion...")
        test_prompt = "What is 2+2?"
        start_time = time.time()
        response = together.Complete.create(
            prompt=test_prompt,
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            max_tokens=100,
            temperature=0.7,
        )
        completion_time = time.time() - start_time
        
        print(f"Response time: {completion_time:.2f}s")
        print(f"Response: {json.dumps(response, indent=2)}")

        # Test error handling
        print("\n2. Testing error handling...")
        try:
            # Intentionally cause an error with invalid model
            response = together.Complete.create(
                prompt=test_prompt,
                model="invalid-model",
                max_tokens=100
            )
        except Exception as e:
            print(f"Expected error caught: {str(e)}")

        # Test parameter variations
        print("\n3. Testing parameter variations...")
        parameters = [
            {"temperature": 0.1},
            {"temperature": 0.9},
            {"max_tokens": 50},
            {"max_tokens": 200}
        ]

        for params in parameters:
            param_name = list(params.keys())[0]
            param_value = params[param_name]
            print(f"\nTesting with {param_name}={param_value}")
            
            try:
                response = together.Complete.create(
                    prompt=test_prompt,
                    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                    **params
                )
                print(f"Success - Response length: {len(str(response))}")
            except Exception as e:
                print(f"Error: {str(e)}")

        print("\n✓ Debug tests completed")

    except Exception as e:
        print(f"\n✗ Fatal error in debug_together_api: {str(e)}")
        raise

if __name__ == "__main__":
    print("Starting Together AI API Debug...")
    debug_together_api() 