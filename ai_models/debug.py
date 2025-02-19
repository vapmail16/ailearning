import os
import asyncio
from dotenv import load_dotenv
import openai
from src.model_comparison import ModelComparison

async def test_openai_call(model_comparison):
    print("\n=== Testing OpenAI API ===")
    print("Attempting simple OpenAI API call...")
    response = await model_comparison.get_model_response_async("openai", "Say hello")
    print(f"OpenAI Response: {response}")
    return response

def run_diagnostics():
    print("\n=== Environment Setup ===")
    # Load environment variables
    load_dotenv()
    print("Loading environment variables...")
    
    # Check API Keys
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    together_key = os.getenv('TOGETHER_API_KEY')
    gemini_key = os.getenv('GOOGLE_API_KEY')
    
    print(f"\nAPI Keys Status:")
    print(f"OpenAI API Key: {'Present' if openai_key else 'Missing'}")
    if openai_key:
        print(f"OpenAI Key Format: {openai_key[:6]}...")
    print(f"Anthropic API Key: {'Present' if anthropic_key else 'Missing'}")
    print(f"Together API Key: {'Present' if together_key else 'Missing'}")
    print(f"Gemini API Key: {'Present' if gemini_key else 'Missing'}")
    
    print("\n=== Package Versions ===")
    print(f"OpenAI Package Version: {openai.__version__}")
    
    print("\n=== Model Comparison Setup ===")
    try:
        model_comparison = ModelComparison()
        print("ModelComparison initialized successfully")
        print(f"OpenAI Client Type: {type(model_comparison.openai_client)}")
        
        # Run async test
        asyncio.run(test_openai_call(model_comparison))
        
    except Exception as e:
        print(f"\nError initializing ModelComparison: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    run_diagnostics() 