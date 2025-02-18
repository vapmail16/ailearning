import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from src.function_registry import get_weather

def test_weather_api():
    # Ensure environment variables are loaded
    load_dotenv()
    
    # Print API key for debugging (remove in production)
    api_key = os.getenv('OPENWEATHER_API_KEY')
    print(f"Using API Key: {api_key[:5]}..." if api_key else "No API key found!")
    
    print("\n=== Testing Weather API Integration ===\n")
    
    test_cases = [
        {"location": "London", "units": "celsius"},
        {"location": "New York", "units": "fahrenheit"},
        {"location": "Tokyo", "units": "celsius"},
        {"location": "NonExistentCity", "units": "celsius"}  # Should handle error
    ]
    
    for params in test_cases:
        print(f"\nTesting weather for {params['location']}:")
        print("-" * 50)
        
        try:
            result = get_weather(params)
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Temperature: {result['temperature']}°{'C' if params['units'] == 'celsius' else 'F'}")
                print(f"Conditions: {result['conditions']}")
                print(f"Humidity: {result['humidity']}%")
                print(f"Wind Speed: {result['wind_speed']} {'m/s' if params['units'] == 'celsius' else 'mph'}")
        except Exception as e:
            print(f"Exception occurred: {str(e)}")

if __name__ == "__main__":
    test_weather_api() 