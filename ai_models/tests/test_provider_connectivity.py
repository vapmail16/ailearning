import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_comparison import ModelComparison, MODEL_TIERS
import json
from datetime import datetime

def test_provider_connectivity():
    model_comparison = ModelComparison()
    test_prompt = "Tell me a short joke about programming."
    results = {}
    
    print("\n=== Testing Provider Connectivity ===\n")
    
    for tier, providers in MODEL_TIERS.items():
        print(f"\nTesting {tier.upper()} Models:")
        print("-" * 50)
        
        tier_results = {}
        for provider, model in providers.items():
            print(f"\nTesting {provider} ({model}):")
            try:
                result = model_comparison.get_model_response(
                    provider, 
                    test_prompt,
                    {'tier': tier}
                )
                
                is_error = isinstance(result.get('response'), str) and result['response'].startswith('Error:')
                
                print(f"Status: {'ERROR' if is_error else 'SUCCESS'}")
                print(f"Response: {result['response'][:100]}...")
                print(f"Time taken: {result['metrics']['time_taken']:.2f}s")
                print(f"Cost: ${result['metrics']['cost']:.4f}")
                
                tier_results[provider] = {
                    'status': 'error' if is_error else 'success',
                    'model': model,
                    'response': result['response'][:100],
                    'metrics': result['metrics']
                }
                
            except Exception as e:
                print(f"Status: FAILED")
                print(f"Error: {str(e)}")
                tier_results[provider] = {
                    'status': 'failed',
                    'model': model,
                    'error': str(e)
                }
        
        results[tier] = tier_results
    
    # Save diagnostic results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"test_results/provider_diagnostic_{timestamp}.json"
    os.makedirs("test_results", exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Diagnostic Results Summary ===\n")
    for tier, tier_results in results.items():
        print(f"\n{tier.upper()}:")
        for provider, data in tier_results.items():
            status = data['status']
            status_color = {
                'success': '\033[92m',  # Green
                'error': '\033[93m',    # Yellow
                'failed': '\033[91m'    # Red
            }.get(status, '')
            print(f"{status_color}{provider}: {status.upper()}\033[0m")
    
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    test_provider_connectivity() 