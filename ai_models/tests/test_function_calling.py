import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_comparison import ModelComparison
from src.function_registry import AVAILABLE_FUNCTIONS, get_weather
import json
from datetime import datetime

def test_function_calling():
    model_comparison = ModelComparison()
    
    print("\n=== Testing Function Calling Across Models ===\n")
    
    test_cases = [
        {
            "location": "London",
            "units": "celsius"
        }
    ]
    
    for params in test_cases:
        print(f"\nTesting weather function for {params['location']}:")
        print("-" * 50)
        
        # Get actual weather data first
        weather_data = get_weather(params)
        if "error" in weather_data:
            print(f"Error getting weather data: {weather_data['error']}")
            continue
            
        print("\nActual Weather Data:")
        print(json.dumps(weather_data, indent=2))
        
        # Test each model's function calling
        models = ['openai', 'anthropic', 'gemini']
        function_def = AVAILABLE_FUNCTIONS['get_weather']
        
        for model in models:
            print(f"\nTesting {model.upper()}:")
            try:
                if model == 'openai':
                    response = model_comparison.openai_client.chat.completions.create(
                        model="gpt-4-turbo-preview",
                        messages=[{
                            "role": "user",
                            "content": f"Get the weather for {params['location']}"
                        }],
                        functions=[function_def],
                        function_call={"name": "get_weather"}
                    )
                    
                    function_call = response.choices[0].message.function_call
                    print(f"Function Called: {function_call.name if function_call else 'None'}")
                    print(f"Parameters: {function_call.arguments if function_call else 'None'}")
                    
                elif model == 'anthropic':
                    response = model_comparison.anthropic_client.messages.create(
                        model="claude-3-opus-20240229",
                        max_tokens=1024,
                        messages=[{
                            "role": "user",
                            "content": f"Get the weather for {params['location']}. Use the get_weather function."
                        }],
                        tools=[{
                            "type": "custom",
                            "name": "get_weather",
                            "description": "Get current weather for a location",
                            "parameters": {
                                "$schema": "http://json-schema.org/draft-07/schema#",
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "City name or location"
                                    },
                                    "units": {
                                        "type": "string",
                                        "enum": ["celsius", "fahrenheit"],
                                        "description": "Temperature unit"
                                    }
                                },
                                "required": ["location"]
                            }
                        }]
                    )
                    
                    print(f"Response: {response.content[0].text[:200]}...")
                    
                elif model == 'gemini':
                    response = model_comparison.get_model_response(
                        'gemini',
                        f"Get the weather for {params['location']}. Use the get_weather function.",
                        {'function_def': function_def}
                    )
                    print(f"Response: {response['response'][:200]}...")
                
                print("Status: SUCCESS")
                
            except Exception as e:
                print(f"Status: FAILED")
                print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_function_calling() 