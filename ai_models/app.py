from flask import Flask, render_template, request, jsonify, current_app, Response
from src.model_comparison import ModelComparison, MODEL_TIERS
import time
import os
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime
from src.function_registry import AVAILABLE_FUNCTIONS, get_weather, calculator, wikipedia_search, currency_convert
from dotenv import load_dotenv
import traceback
from tests.test_performance import TestPerformance  # Add this import at the top

app = Flask(__name__)
model_comparison = ModelComparison()

# Load environment variables at app startup
load_dotenv()
print("Environment variables loaded in app.py")
print(f"Exchange Rate API Key available: {'Yes' if os.getenv('EXCHANGERATE_API_KEY') else 'No'}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare_models():
    try:
        prompt = request.json.get('prompt')
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400

        print(f"Received prompt: {prompt}")

        responses = {}
        start_time = time.time()

        # Get responses from all models
        models_to_test = ["openai", "anthropic", "gemini", "together"]  # Added back all models
        
        for model in models_to_test:
            try:
                print(f"Attempting to get response from {model}...")
                
                # Verify API keys before making calls
                api_key_map = {
                    "openai": "OPENAI_API_KEY",
                    "anthropic": "ANTHROPIC_API_KEY",
                    "gemini": "GOOGLE_API_KEY",
                    "together": "TOGETHER_API_KEY"
                }
                
                api_key = os.getenv(api_key_map[model])
                if not api_key:
                    raise ValueError(f"Missing API key for {model}")
                
                result = model_comparison.get_model_response(model, prompt)
                print(f"Raw result from {model}:", result)
                
                if isinstance(result, dict) and 'response' in result:
                    responses[model] = {
                        'text': result['response'],
                        'metrics': {
                            'time_taken': result.get('metrics', {}).get('time_taken', 0),
                            'length': result.get('metrics', {}).get('length', 0),
                            'accuracy': result.get('metrics', {}).get('accuracy', 0),
                            'creativity_score': result.get('metrics', {}).get('creativity', 0),
                            'logical_score': result.get('metrics', {}).get('logical_score', 0),
                            'factual_score': result.get('metrics', {}).get('accuracy', 0),
                            'cost': result.get('metrics', {}).get('cost', 0.0)
                        }
                    }
                else:
                    print(f"Invalid response format from {model}:", result)
                    responses[model] = {
                        'text': 'Error: Invalid response format',
                        'metrics': {
                            'time_taken': 0,
                            'length': 0,
                            'accuracy': 0,
                            'creativity_score': 0,
                            'logical_score': 0,
                            'factual_score': 0,
                            'cost': 0.0
                        }
                    }
                    
            except Exception as e:
                print(f"Error with {model}: {str(e)}")
                print(f"Traceback: {traceback.format_exc()}")
                responses[model] = {
                    'text': f'Error: {str(e)}',
                    'metrics': {
                        'time_taken': 0,
                        'length': 0,
                        'accuracy': 0,
                        'creativity_score': 0,
                        'logical_score': 0,
                        'factual_score': 0,
                        'cost': 0.0
                    }
                }

        # Calculate summary
        if any(not r['text'].startswith('Error:') for r in responses.values()):
            summary = calculate_summary(responses)
        else:
            summary = {
                'fastest_model': 'N/A',
                'most_accurate_model': 'N/A',
                'best_overall_model': 'N/A',
                'analysis': 'All models returned errors'
            }

        return jsonify({
            'responses': responses,
            'summary': summary,
            'total_time': time.time() - start_time
        })

    except Exception as e:
        print(f"General error in compare_models: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': f'An error occurred while comparing models: {str(e)}',
            'details': traceback.format_exc()
        }), 500

@app.route('/parameter-test', methods=['POST'])
def parameter_test():
    try:
        data = request.json
        model = data.get('model')
        prompt = data.get('prompt')
        params = data.get('params', {})
        
        result = model_comparison.get_model_response(model, prompt, params)
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in parameter test: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/tier-test', methods=['POST'])
def test_tiers():
    data = request.json
    prompt = data.get('prompt')
    tier = data.get('tier', 'tier_1')

    if not prompt:
        return jsonify({'error': 'Missing prompt'}), 400

    results = model_comparison.compare_model_tiers(prompt, tier)
    
    # Save tier comparison results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"test_results/tier_test_{tier}_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    return jsonify(results)

@app.route('/compare-tiers', methods=['POST'])
def compare_tiers():
    data = request.json
    prompt = data.get('prompt')
    tier = data.get('tier', 'tier_1')
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    try:
        # Get responses from models in the selected tier
        responses = {}
        for provider, model_name in MODEL_TIERS[tier].items():
            try:
                result = model_comparison.get_model_response(provider, prompt, {'tier': tier})
                
                if not isinstance(result.get('response'), str) or not result['response'].startswith('Error:'):
                    responses[provider] = {
                        'text': result['response'],
                        'metrics': {
                            'time_taken': result['metrics']['time_taken'],
                            'accuracy': result['metrics']['accuracy'],
                            'creativity_score': result['metrics']['creativity'],
                            'logical_score': result['metrics'].get('logical_score', 0),
                            'cost': result['metrics']['cost']
                        }
                    }
            except Exception as provider_error:
                print(f"Error with provider {provider}: {str(provider_error)}")
                continue

        if not responses:
            return jsonify({'error': 'No valid responses from any provider in this tier'}), 500

        return jsonify({
            'responses': responses,
            'tier': tier
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test-function', methods=['POST'])
def test_function():
    data = request.json
    function_type = data.get('function')
    parameters = data.get('parameters', {})
    selected_models = data.get('models', [])
    
    if not function_type or not selected_models:
        return jsonify({'error': 'Missing function type or models'}), 400

    try:
        results = {}
        actual_result = None  # Initialize actual_result at the top
        
        # Get actual function result first
        try:
            if function_type == 'get_weather':
                actual_result = get_weather(parameters)
            elif function_type == 'calculator':
                actual_result = calculator(parameters)
            elif function_type == 'wikipedia_search':
                if not parameters.get('query'):
                    return jsonify({'error': 'Search query is required'}), 400
                actual_result = wikipedia_search(parameters)
            elif function_type == 'currency_convert':
                # Validate currency parameters
                required_params = ['amount', 'from_currency', 'to_currency']
                missing_params = [p for p in required_params if p not in parameters]
                
                if missing_params:
                    return jsonify({
                        'error': f'Missing required parameters: {", ".join(missing_params)}'
                    }), 400
                
                try:
                    amount = float(parameters['amount'])
                    if amount <= 0:
                        return jsonify({'error': 'Amount must be greater than 0'}), 400
                except ValueError:
                    return jsonify({'error': 'Invalid amount value - must be a number'}), 400
                
                # Validate currency codes
                from_currency = str(parameters['from_currency']).strip().upper()
                to_currency = str(parameters['to_currency']).strip().upper()
                
                if not from_currency or not to_currency:
                    return jsonify({'error': 'Invalid currency codes'}), 400
                
                # Update parameters with validated values
                parameters.update({
                    'amount': amount,
                    'from_currency': from_currency,
                    'to_currency': to_currency
                })
                
                actual_result = currency_convert(parameters)
                
                if 'error' in actual_result:
                    print(f"Currency conversion error: {actual_result['error']}")
                    return jsonify(actual_result), 400
            
            if isinstance(actual_result, dict) and 'error' in actual_result:
                return jsonify({'error': actual_result['error']}), 400

        except Exception as e:
            print(f"Function execution error: {str(e)}")
            return jsonify({'error': f"Function execution error: {str(e)}"}), 500

        # Test each selected model
        for model in selected_models:
            start_time = time.time()
            prompt = f"Use the {function_type} function with these parameters: {json.dumps(parameters)}"
            
            try:
                if model == 'openai':
                    response = model_comparison.openai_client.chat.completions.create(
                        model="gpt-4-turbo-preview",
                        messages=[{"role": "user", "content": prompt}],
                        functions=[AVAILABLE_FUNCTIONS[function_type]],
                        function_call={"name": function_type}
                    )
                    
                    if hasattr(response.choices[0].message, 'function_call'):
                        success = True
                        response_text = actual_result  # Use actual result for display
                    else:
                        success = False
                        response_text = "No function call made"
                    
                elif model == 'anthropic':
                    # Format tool schema according to Claude's requirements
                    tool_schema = {
                        "type": "object",
                        "properties": AVAILABLE_FUNCTIONS[function_type]["parameters"]["properties"],
                        "required": AVAILABLE_FUNCTIONS[function_type]["parameters"]["required"]
                    }

                    response = model_comparison.anthropic_client.messages.create(
                        model="claude-3-opus-20240229",
                        max_tokens=1024,
                        messages=[{"role": "user", "content": prompt}],
                        tools=[{
                            "type": "custom",
                            "function": AVAILABLE_FUNCTIONS[function_type]["name"],
                            "description": AVAILABLE_FUNCTIONS[function_type]["description"],
                            "parameters": tool_schema
                        }]
                    )
                    
                    try:
                        # Try to parse tool calls from response
                        if hasattr(response.content[0], 'tool_calls') and response.content[0].tool_calls:
                            success = True
                            response_text = actual_result
                        else:
                            # Try to parse the response as JSON
                            try:
                                parsed_response = json.loads(response.content[0].text)
                                if isinstance(parsed_response, dict) and all(k in parsed_response for k in tool_schema["properties"].keys()):
                                    success = True
                                    response_text = actual_result
                                else:
                                    success = False
                                    response_text = response.content[0].text
                            except json.JSONDecodeError:
                                success = False
                                response_text = response.content[0].text
                    except Exception as e:
                        print(f"Error processing Anthropic response: {str(e)}")
                        success = False
                        response_text = f"Error: {str(e)}"
                    
                elif model == 'gemini':
                    response = model_comparison.get_model_response(
                        'gemini',
                        prompt,
                        {'function_def': AVAILABLE_FUNCTIONS[function_type]}
                    )
                    success = True
                    response_text = actual_result  # Use actual result for display
                
                results[model] = {
                    'function_called': function_type,
                    'parameters': parameters,
                    'response': response_text,
                    'success': success,
                    'metrics': {
                        'time_taken': time.time() - start_time,
                        'cost': model_comparison.calculate_cost(model, prompt, str(response_text))
                    }
                }
                
            except Exception as e:
                print(f"Error with {model}: {str(e)}")
                results[model] = {
                    'function_called': function_type,
                    'parameters': parameters,
                    'response': f"Error: {str(e)}",
                    'success': False,
                    'metrics': {
                        'time_taken': time.time() - start_time,
                        'cost': 0
                    }
                }

        return jsonify({
            'results': results,
            'actual_data': actual_result
        })

    except Exception as e:
        print(f"General error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/test-performance', methods=['POST'])
def test_performance():
    try:
        data = request.json
        test_type = data.get('test_type')
        test_performance = TestPerformance(model_comparison)

        if test_type == 'streaming':
            print(f"Running streaming test with model: {data.get('model', 'openai')}")
            return Response(
                test_performance.test_streaming_response(
                    data.get('prompt', ''), 
                    data.get('model', 'openai')
                ),
                mimetype='text/event-stream'
            )
            
        elif test_type == 'latency':
            simple_prompt = data.get('simple_prompt')
            complex_prompt = data.get('complex_prompt')
            print(f"Running latency test with simple prompt: {simple_prompt}")
            print(f"Complex prompt: {complex_prompt}")
            
            results = test_performance.test_simple_vs_complex_latency(
                simple_prompt=simple_prompt,
                complex_prompt=complex_prompt
            )
            return jsonify(results)
            
        elif test_type == 'batch':
            batch_size = int(data.get('batch_size', 100))
            model = data.get('model', 'openai')
            prompt = data.get('prompt', '')
            
            print(f"Running batch test with size: {batch_size} using model: {model}")
            print(f"Prompt: {prompt}")
            
            try:
                results = test_performance.test_batch_vs_individual(
                    batch_size=batch_size,
                    base_prompt=prompt,
                    model=model
                )
                return jsonify(results)
            except Exception as e:
                print(f"Error in batch test: {str(e)}")
                traceback.print_exc()
                return jsonify({'error': str(e)}), 500
            
        return jsonify({'error': 'Invalid test type'}), 400
        
    except Exception as e:
        print(f"Error in test_performance: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def calculate_summary(responses):
    try:
        # Find the best performing models
        valid_responses = {k: v for k, v in responses.items() if v['metrics']['time_taken'] > 0}
        
        if valid_responses:
            fastest_model = min(valid_responses.items(), key=lambda x: x[1]['metrics']['time_taken'])[0]
            most_accurate = max(valid_responses.items(), key=lambda x: x[1]['metrics']['accuracy'])[0]
            
            # Calculate overall score
            overall_scores = {}
            for model, data in valid_responses.items():
                metrics = data['metrics']
                overall_scores[model] = (
                    0.4 * metrics['accuracy'] +
                    0.4 * (1 - min(metrics['time_taken'] / 10, 1)) +  # Normalize time
                    0.2 * min(metrics['length'] / 1000, 1)  # Normalize length
                )
            
            best_overall = max(overall_scores.items(), key=lambda x: x[1])[0]
            
            return {
                'fastest_model': fastest_model,
                'most_accurate_model': most_accurate,
                'best_overall_model': best_overall,
                'analysis': generate_analysis(valid_responses, overall_scores)
            }
        else:
            return {
                'fastest_model': 'N/A',
                'most_accurate_model': 'N/A',
                'best_overall_model': 'N/A',
                'analysis': 'No valid responses received from models.'
            }
    except Exception as e:
        print(f"Error in calculate_summary: {str(e)}")
        return {
            'fastest_model': 'Error',
            'most_accurate_model': 'Error',
            'best_overall_model': 'Error',
            'analysis': f'Error calculating summary: {str(e)}'
        }

def generate_analysis(responses, overall_scores):
    try:
        best_model = max(overall_scores.items(), key=lambda x: x[1])[0]
        fastest = min(responses.items(), key=lambda x: x[1]['metrics']['time_taken'])[0]
        most_detailed = max(responses.items(), key=lambda x: x[1]['metrics']['length'])[0]
        
        return (f"{best_model} performed best overall, with a good balance of speed and accuracy. "
                f"{fastest} was particularly fast in responding, while {most_detailed} provided "
                f"the most detailed responses.")
    except Exception as e:
        return f"Error generating analysis: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, port=5001) 