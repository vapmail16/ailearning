import time
import asyncio
import traceback
import json

class TestPerformance:
    def __init__(self, model_comparison):
        self.model_comparison = model_comparison

    def test_simple_vs_complex_latency(self, simple_prompt, complex_prompt):
        """Test latency differences between simple and complex queries"""
        results = {}
        
        print(f"Testing simple prompt: {simple_prompt}")
        print(f"Testing complex prompt: {complex_prompt}")
        
        for model in ["openai", "anthropic", "gemini", "together"]:
            try:
                print(f"Testing model: {model}")
                
                # Get simple query response
                simple_start = time.time()
                simple_result = self.model_comparison.get_model_response(model, simple_prompt)
                simple_time = time.time() - simple_start
                
                # Get complex query response
                complex_start = time.time()
                complex_result = self.model_comparison.get_model_response(model, complex_prompt)
                complex_time = time.time() - complex_start
                
                results[model] = {
                    'simple_latency': simple_time,
                    'complex_latency': complex_time,
                    'simple_response': simple_result['response'],
                    'complex_response': complex_result['response'],
                    'simple_cost': simple_result['metrics']['cost'],
                    'complex_cost': complex_result['metrics']['cost']
                }
                
                print(f"Results for {model}: {results[model]}")
                
            except Exception as e:
                print(f"Error testing {model}: {str(e)}")
                results[model] = {
                    'simple_latency': 0,
                    'complex_latency': 0,
                    'simple_response': f"Error: {str(e)}",
                    'complex_response': f"Error: {str(e)}",
                    'simple_cost': 0,
                    'complex_cost': 0
                }
        
        return results

    def test_streaming_response(self, prompt, model="openai"):
        """Test streaming response capabilities"""
        try:
            print(f"Starting streaming test with {model} for prompt: {prompt}")
            
            # Get streaming response from model
            for chunk in self.model_comparison.get_streaming_response(model, prompt):
                # Format chunk as SSE data
                yield f"data: {json.dumps(chunk)}\n\n"
                
        except Exception as e:
            error_msg = f"Error in streaming test: {str(e)}"
            print(error_msg)
            yield f"data: {json.dumps({'error': True, 'content': error_msg})}\n\n"

    def test_batch_vs_individual(self, batch_size=100, base_prompt=None, model="openai"):
        """Compare batch processing vs individual requests"""
        try:
            # Set default prompt if none provided
            if not base_prompt:
                base_prompt = "Default test prompt"

            # Generate prompts
            prompts = [base_prompt for _ in range(batch_size)]
            individual_results = []
            individual_costs = 0
            
            print(f"Testing with batch size: {batch_size}")
            print(f"Base prompt: {base_prompt}")
            print(f"Model: {model}")
            
            # Test individual requests first to warm up cache
            print(f"Running {batch_size} individual requests with {model} for prompt: {base_prompt[:50]}...")
            
            start_time = time.time()
            
            # Run individual requests
            for i, prompt in enumerate(prompts):
                try:
                    result = self.model_comparison.get_model_response(model, prompt)
                    if result and 'response' in result:
                        individual_results.append(result)
                        individual_costs += result['metrics']['cost']
                        print(f"Individual request {i+1}/{batch_size} completed. Cost: ${result['metrics']['cost']:.4f}")
                except Exception as e:
                    print(f"Error in individual request {i+1}: {str(e)}")
            
            individual_time = time.time() - start_time
            print(f"Individual requests completed in {individual_time:.2f}s")
            print(f"Individual total cost: ${individual_costs:.4f}")
            
            # Test batch requests (should use cache)
            print(f"Running batch request with {batch_size} prompts using {model}...")
            batch_start_time = time.time()
            batch_results = self.model_comparison.batch_process_requests(prompts, model)
            batch_time = time.time() - batch_start_time
            
            if batch_results and 'results' in batch_results:
                batch_costs = batch_results['metrics']['total_cost']
                cache_hits = batch_results['cache_stats']['hits']
                cache_misses = batch_results['cache_stats']['misses']
                
                print(f"Batch requests completed in {batch_time:.2f}s")
                print(f"Batch total cost: ${batch_costs:.4f}")
                print(f"Cache performance - Hits: {cache_hits}, Misses: {cache_misses}")
                
                return {
                    'individual': {
                        'total_time': individual_time,
                        'cost': individual_costs,
                        'requests_completed': len(individual_results)
                    },
                    'batch': {
                        'total_time': batch_time,
                        'cost': batch_costs,
                        'requests_completed': len(batch_results['results']),
                        'cache_stats': batch_results['cache_stats']
                    },
                    'improvements': {
                        'time_saved_percent': ((individual_time - batch_time) / individual_time) * 100,
                        'cost_saved_percent': ((individual_costs - batch_costs) / individual_costs) * 100
                    }
                }
            else:
                raise Exception("Batch processing failed to return valid results")
            
        except Exception as e:
            print(f"Error in batch test: {str(e)}")
            traceback.print_exc()
            return {
                'individual': {
                    'total_time': 0,
                    'cost': 0,
                    'requests_completed': 0
                },
                'batch': {
                    'total_time': 0,
                    'cost': 0,
                    'requests_completed': 0
                },
                'improvements': {
                    'time_saved_percent': 0,
                    'cost_saved_percent': 0
                }
            } 