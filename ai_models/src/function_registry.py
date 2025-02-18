import requests
import os
from typing import Dict, Any, List
import wikipediaapi
from dotenv import load_dotenv
import operator

# Load environment variables with debug info
print("Loading environment variables...")
load_dotenv()

# API Keys with verification
WEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
EXCHANGERATE_API_KEY = os.getenv('EXCHANGERATE_API_KEY')

# Debug environment variables (only showing first few characters for security)
print(f"Weather API Key loaded: {'Yes' if WEATHER_API_KEY else 'No'}")
print(f"Exchange Rate API Key loaded: {'Yes' if EXCHANGERATE_API_KEY else 'No'}")
if not EXCHANGERATE_API_KEY:
    print("WARNING: Exchange Rate API Key not found in environment variables!")
    print(f"Available environment variables: {[k for k in os.environ.keys() if 'KEY' in k]}")

# Verify API key format
if EXCHANGERATE_API_KEY:
    print(f"Exchange Rate API Key format: {EXCHANGERATE_API_KEY[:5]}...")
else:
    # Try loading directly
    EXCHANGERATE_API_KEY = '759fcd3c6397fdcbb1f279d0'
    print("Using hardcoded Exchange Rate API Key as fallback")

# Initialize Wikipedia API with proper user agent and error handling
try:
    wiki_wiki = wikipediaapi.Wikipedia(
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        user_agent='AIModelComparison/1.0 (https://github.com/yourusername/ai_models)'
    )
except Exception as e:
    print(f"Error initializing Wikipedia API: {str(e)}")
    wiki_wiki = None

AVAILABLE_FUNCTIONS = {
    "get_weather": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
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
    },
    "calculator": {
        "name": "calculator",
        "description": "Perform basic mathematical calculations",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "Mathematical operation to perform"
                },
                "numbers": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Numbers to perform operation on"
                }
            },
            "required": ["operation", "numbers"]
        }
    },
    "wikipedia_search": {
        "name": "wikipedia_search",
        "description": "Search Wikipedia for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search term or topic"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": 1
                }
            },
            "required": ["query"]
        }
    },
    "currency_convert": {
        "name": "currency_convert",
        "description": "Convert between currencies",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "number",
                    "description": "Amount to convert"
                },
                "from_currency": {
                    "type": "string",
                    "description": "Source currency code (e.g., USD)"
                },
                "to_currency": {
                    "type": "string",
                    "description": "Target currency code (e.g., EUR)"
                }
            },
            "required": ["amount", "from_currency", "to_currency"]
        }
    },
    "amazon_order": {
        "name": "amazon_order",
        "description": "Simulate placing an order on Amazon (mock function, no real orders)",
        "parameters": {
            "type": "object",
            "properties": {
                "product": {
                    "type": "string",
                    "description": "Product name or description"
                },
                "quantity": {
                    "type": "integer",
                    "description": "Number of items to order",
                    "minimum": 1
                },
                "max_price": {
                    "type": "number",
                    "description": "Maximum price willing to pay per item (in GBP)",
                    "minimum": 0
                },
                "priority_shipping": {
                    "type": "boolean",
                    "description": "Whether to use priority shipping",
                    "default": False
                }
            },
            "required": ["product", "quantity", "max_price"]
        }
    }
}

# Function implementations
def calculator(params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform basic mathematical calculations"""
    try:
        operation = params.get('operation')
        numbers = params.get('numbers', [])
        
        if not numbers or len(numbers) < 2:
            return {"error": "At least two numbers are required"}
        
        ops = {
            'add': operator.add,
            'subtract': operator.sub,
            'multiply': operator.mul,
            'divide': operator.truediv
        }
        
        if operation not in ops:
            return {"error": f"Invalid operation: {operation}"}
        
        # Reduce the list using the selected operation
        result = numbers[0]
        for num in numbers[1:]:
            if operation == 'divide' and num == 0:
                return {"error": "Division by zero"}
            result = ops[operation](result, num)
        
        return {
            "result": result,
            "operation": operation,
            "numbers": numbers
        }
    except Exception as e:
        return {"error": str(e)}

def wikipedia_search(params: Dict[str, Any]) -> Dict[str, Any]:
    """Search Wikipedia for information"""
    try:
        if wiki_wiki is None:
            return {"error": "Wikipedia API not initialized"}
            
        query = params.get('query')
        if not query:
            return {"error": "Query parameter is required"}
            
        # Search for the page
        try:
            page = wiki_wiki.page(query)
            
            if not page.exists():
                return {
                    "result": f"No Wikipedia page found for: {query}",
                    "query": query,
                    "found": False
                }
            
            # Get summary and full text
            return {
                "result": {
                    "title": page.title,
                    "summary": page.summary[:500],  # First 500 chars of summary
                    "url": page.fullurl
                },
                "query": query,
                "found": True
            }
        except Exception as page_error:
            return {"error": f"Error accessing Wikipedia: {str(page_error)}"}
            
    except Exception as e:
        return {"error": f"Wikipedia search error: {str(e)}"}

def currency_convert(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert between currencies using ExchangeRate API"""
    try:
        # Debug: Print incoming parameters
        print(f"Currency conversion parameters: {params}")
        print(f"Using Exchange Rate API Key: {EXCHANGERATE_API_KEY[:5]}...")  # Only show first 5 chars
        
        amount = float(params.get('amount', 0))
        from_currency = str(params.get('from_currency', '')).upper()
        to_currency = str(params.get('to_currency', '')).upper()
        
        # Debug: Print parsed values
        print(f"Parsed values - Amount: {amount}, From: {from_currency}, To: {to_currency}")
        
        if amount <= 0:
            return {"error": "Amount must be greater than 0"}
            
        if not from_currency or not to_currency:
            return {"error": "Both source and target currencies are required"}
        
        # Get exchange rate
        try:
            if not EXCHANGERATE_API_KEY:
                print("ExchangeRate API key is missing!")
                return {"error": "ExchangeRate API key not found"}

            base_url = "https://v6.exchangerate-api.com/v6"
            url = f"{base_url}/{EXCHANGERATE_API_KEY}/pair/{from_currency}/{to_currency}"
            
            # Debug: Print request URL (remove sensitive info)
            print(f"Making request to: {base_url}/<API_KEY>/pair/{from_currency}/{to_currency}")
            
            response = requests.get(url, timeout=10)  # Add timeout
            
            # Debug: Print full response details
            print(f"Response Status: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            print(f"Response Body: {response.text}")
            
            if response.status_code != 200:
                return {
                    "error": f"API Error: Status {response.status_code}",
                    "details": response.text
                }
            
            data = response.json()
            
            # Check for API-specific error responses
            if data.get('result') == 'error':
                return {
                    "error": "API Error",
                    "details": data.get('error', 'Unknown API error')
                }
            
            conversion_rate = data.get('conversion_rate')
            if not conversion_rate:
                return {
                    "error": "Missing conversion rate",
                    "details": "The API response did not include a conversion rate"
                }
            
            converted_amount = amount * conversion_rate
            
            result = {
                "result": {
                    "from_amount": amount,
                    "from_currency": from_currency,
                    "to_currency": to_currency,
                    "rate": conversion_rate,
                    "converted_amount": round(converted_amount, 2)
                },
                "success": True
            }
            
            # Debug: Print final result
            print(f"Conversion successful: {result}")
            return result
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Currency API request failed: {str(e)}"
            print(f"Request Error: {error_msg}")
            return {"error": error_msg}
            
    except Exception as e:
        error_msg = f"Currency conversion error: {str(e)}"
        print(f"General Error: {error_msg}")
        return {"error": error_msg}

def get_weather(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get weather data from OpenWeatherMap API"""
    try:
        location = params.get('location')
        units = 'metric' if params.get('units') == 'celsius' else 'imperial'
        
        # Print URL for debugging (remove in production)
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&units={units}&appid={WEATHER_API_KEY}"
        print(f"Requesting URL: {url}")
        
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        return {
            "temperature": data['main']['temp'],
            "conditions": data['weather'][0]['description'],
            "humidity": data['main']['humidity'],
            "wind_speed": data['wind']['speed']
        }
    except Exception as e:
        return {"error": f"{str(e)}\nAPI Key used: {WEATHER_API_KEY[:5]}..."}

def amazon_order(params: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate placing an order on Amazon (mock function)"""
    try:
        product = params.get('product', '').strip()
        quantity = int(params.get('quantity', 1))
        max_price = float(params.get('max_price', 0))
        priority_shipping = bool(params.get('priority_shipping', False))

        if not product:
            return {"error": "Product description is required"}
        
        if quantity < 1:
            return {"error": "Quantity must be at least 1"}
            
        if max_price <= 0:
            return {"error": "Maximum price must be greater than 0"}

        # Simulate product search and price check
        import random
        found_price = round(random.uniform(max_price * 0.7, max_price * 1.2), 2)
        shipping_days = random.randint(2, 7) if priority_shipping else random.randint(5, 12)
        
        # Check if price is acceptable
        if found_price > max_price:
            return {
                "result": {
                    "status": "price_too_high",
                    "product": product,
                    "found_price": f"£{found_price:.2f}",
                    "max_price": f"£{max_price:.2f}",
                    "message": f"Found price (£{found_price:.2f}) exceeds maximum price (£{max_price:.2f})"
                },
                "success": False
            }

        # Calculate total cost
        shipping_cost = 7.99 if priority_shipping else 3.99  # UK shipping rates
        subtotal = found_price * quantity
        tax = round(subtotal * 0.20, 2)  # UK VAT rate (20%)
        total = subtotal + shipping_cost + tax

        # Simulate successful order
        order_id = f"UK-MOK-{random.randint(100000, 999999)}"
        
        return {
            "result": {
                "status": "order_placed",
                "order_id": order_id,
                "product": product,
                "quantity": quantity,
                "unit_price": f"£{found_price:.2f}",
                "subtotal": f"£{subtotal:.2f}",
                "shipping_cost": f"£{shipping_cost:.2f}",
                "tax": f"£{tax:.2f}",
                "total": f"£{total:.2f}",
                "estimated_delivery": f"{shipping_days} days",
                "shipping_method": "Priority" if priority_shipping else "Standard",
                "success": True,
                "message": f"Order placed successfully! Your order #{order_id} will arrive in {shipping_days} days."
            }
        }

    except ValueError as e:
        return {"error": f"Invalid parameter value: {str(e)}"}
    except Exception as e:
        return {"error": f"Order processing error: {str(e)}"}

print(f"Current working directory: {os.getcwd()}")
print(f"Looking for .env in: {os.path.join(os.getcwd(), '.env')}") 