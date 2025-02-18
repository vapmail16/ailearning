import os
import pytest
from dotenv import load_dotenv
import openai
import anthropic
import together
import requests

# Load environment variables
load_dotenv()

class TestAPIConnections:
    def test_openai_connection(self):
        """Test OpenAI API connection"""
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello, are you working?"}]
            )
            assert response.choices[0].message.content is not None
            print("\n✓ OpenAI connection successful")
        except Exception as e:
            pytest.fail(f"OpenAI connection failed: {str(e)}")

    def test_anthropic_connection(self):
        """Test Anthropic API connection"""
        client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",  # ✅ Corrected model name
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello, are you working?"}]
            )
            assert response.content[0].text is not None
            print("✓ Anthropic connection successful")
        except Exception as e:
            pytest.fail(f"Anthropic connection failed: {str(e)}")

    def test_together_connection(self):
        """Test Together.ai API connection"""
        together.api_key = os.getenv('TOGETHER_API_KEY')
        try:
            response = together.Complete.create(
                prompt="Hello, are you working?",
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",  # ✅ Corrected model name
                max_tokens=256
            )

            # ✅ Fix: Validate response structure
            if "choices" in response and response["choices"]:
                assert response["choices"][0]["text"] is not None
                print("✓ Together.ai connection successful")
            else:
                pytest.fail(f"Together.ai response format error: {response}")
        except Exception as e:
            pytest.fail(f"Together.ai connection failed: {str(e)}")

    def test_google_gemini_connection(self):
        """Test Google Gemini API connection"""
        api_key = os.getenv('GEMINI_API_KEY')

        if not api_key or len(api_key) < 10:
            pytest.fail("Google Gemini API Key is missing or invalid.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": "Hello, are you working?"}]}]
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response_json = response.json()
            assert "candidates" in response_json
            print("✓ Google Gemini connection successful")
        except Exception as e:
            pytest.fail(f"Google Gemini connection failed: {str(e)}")

    # Commenting out DeepSeek test until service is available
    """
    def test_deepseek_connection(self):
        """Test DeepSeek API connection"""
        api_key = os.getenv('DEEPSEEK_API_KEY')

        if not api_key or len(api_key) < 10:
            pytest.skip("DeepSeek skipped due to missing API key.")

        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Hello, are you working?"}],
            "max_tokens": 256
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response_json = response.json()

            if "error" in response_json and "balance" in response_json["error"]["message"]:
                pytest.skip("DeepSeek skipped due to insufficient balance.")
            else:
                assert "choices" in response_json
                print("✓ DeepSeek connection successful")
        except Exception as e:
            pytest.fail(f"DeepSeek connection failed: {str(e)}")
    """
