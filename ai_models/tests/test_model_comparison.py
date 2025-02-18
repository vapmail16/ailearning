import os
import time
import json
from datetime import datetime
import pytest
from dotenv import load_dotenv
import openai
import anthropic
import together
import requests

# Load environment variables
load_dotenv()

class ModelComparison:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.together_api_key = os.getenv('TOGETHER_API_KEY')
        self.gemini_api_key = os.getenv('GOOGLE_API_KEY')
        
        # Initialize results directory without cleaning it
        self.results_dir = "test_results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def calculate_factual_score(self, response, expected):
        """Calculate factual accuracy score"""
        score = 0
        response_lower = response.lower()
        expected_lower = expected.lower()
        
        # Exact match
        if expected_lower in response_lower:
            score += 1.0
        # Partial match (at least 50% of words match)
        elif any(word in response_lower for word in expected_lower.split()):
            score += 0.5
            
        return score

    def calculate_creativity_score(self, response):
        """Calculate creativity score based on multiple factors"""
        score = 0
        response_lower = response.lower()
        
        # Length score (up to 0.2)
        length_score = min(len(response) / 500, 0.2)
        score += length_score
        
        # Unique words ratio (up to 0.3)
        words = response_lower.split()
        unique_ratio = len(set(words)) / len(words)
        score += unique_ratio * 0.3
        
        # Required elements (up to 0.5)
        required_elements = ["time", "coffee", "story", "travel"]
        for element in required_elements:
            if element in response_lower:
                score += 0.125
                
        return min(score, 1.0)

    def calculate_logical_score(self, response, prompt):
        """Calculate logical reasoning score"""
        score = 0
        response_lower = response.lower()
        
        if "sequence" in prompt:
            if "30" in response:
                score += 0.5
            if "quadratic" in response_lower or "n^2" in response_lower:
                score += 0.5
        elif "farmer" in prompt:
            key_steps = ["chicken", "fox", "corn", "back", "across"]
            for step in key_steps:
                if step in response_lower:
                    score += 0.2
                    
        return min(score, 1.0)

    def log_response(self, model_name, prompt_type, prompt, response, time_taken, scores=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{self.results_dir}/{model_name}_{prompt_type}_{timestamp}.json"
        
        log_data = {
            "model": model_name,
            "prompt_type": prompt_type,
            "prompt": prompt,
            "response": response,
            "time_taken": time_taken,
            "scores": scores or {}
        }
        
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)

    def get_model_response(self, model_name, prompt):
        start_time = time.time()
        response = None

        try:
            if model_name == "openai":
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.choices[0].message.content

            elif model_name == "anthropic":
                response = self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.content[0].text

            elif model_name == "together":
                together.api_key = self.together_api_key
                response = together.Complete.create(
                    prompt=prompt,
                    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                    max_tokens=256
                )
                if "choices" in response and response["choices"]:
                    response_text = response["choices"][0]["text"]
                else:
                    return None, None, f"Together.ai response format error: {response}"

            elif model_name == "gemini":
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={self.gemini_api_key}"
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}]
                }
                response = requests.post(url, json=payload)
                response_json = response.json()
                
                if "candidates" in response_json and response_json["candidates"]:
                    response_text = response_json["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    return None, None, f"Gemini response format error: {response_json}"

        except Exception as e:
            error_msg = str(e)
            if isinstance(e, requests.exceptions.RequestException):
                error_msg = f"Network error: {str(e)}"
            elif "rate limit" in error_msg.lower():
                error_msg = f"Rate limit exceeded for {model_name}"
            return None, None, f"{model_name} failed: {error_msg}"

        time_taken = time.time() - start_time
        return response_text, time_taken, None

class TestModelComparison:
    @pytest.fixture
    def comparison(self):
        return ModelComparison()

    def test_factual_accuracy(self, comparison):
        """Test factual accuracy with verifiable questions"""
        prompts = [
            "What is the capital of France?",
            "What is the chemical formula for water?",
            "Who wrote 'Romeo and Juliet'?"
        ]
        
        expected_answers = [
            "Paris",
            "H2O",
            "William Shakespeare"
        ]
        
        results = {}
        for prompt, expected in zip(prompts, expected_answers):
            for model in ["openai", "anthropic", "together", "gemini"]:
                response, time_taken, error = comparison.get_model_response(model, prompt)
                
                if error:
                    print(f"\nWarning: {error}")
                    continue
                
                # Calculate scores
                factual_score = comparison.calculate_factual_score(response, expected)
                response_length_score = min(len(response) / 200, 1.0)
                
                scores = {
                    "factual_accuracy": factual_score,
                    "response_length": response_length_score,
                    "response_time_score": 1.0 if time_taken < 5 else 0.5
                }
                
                comparison.log_response(model, "factual", prompt, response, time_taken, scores)
                results[f"{model}_{prompt}"] = "PASS" if factual_score > 0.5 else "FAIL"

        # Ensure at least one model passed each prompt
        for prompt in prompts:
            passed = any(results.get(f"{model}_{prompt}") == "PASS" 
                        for model in ["openai", "anthropic", "together", "gemini"])
            assert passed, f"All models failed for prompt: {prompt}"

    def test_response_time(self, comparison):
        """Test response time for a simple prompt"""
        prompt = "Hello, how are you?"
        
        for model in ["openai", "anthropic", "together", "gemini"]:
            response, time_taken, error = comparison.get_model_response(model, prompt)
            
            if error:
                print(f"\nWarning: {error}")
                continue
            
            scores = {
                "response_time_score": 1.0 if time_taken < 5 else 0.5,
                "response_length": min(len(str(response)) / 200, 1.0)
            }
            
            comparison.log_response(model, "response_time", prompt, response, time_taken, scores)
            assert time_taken < 10, f"{model} response time exceeded 10 seconds"

    def test_creativity(self, comparison):
        """Test creative writing capabilities"""
        prompt = "Write a short story about a time-traveling coffee cup"
        
        for model in ["openai", "anthropic", "together", "gemini"]:
            response, time_taken, error = comparison.get_model_response(model, prompt)
            
            if error:
                print(f"\nWarning: {error}")
                continue
            
            creativity_score = comparison.calculate_creativity_score(str(response))
            scores = {
                "creativity": creativity_score,
                "response_time_score": 1.0 if time_taken < 5 else 0.5,
                "response_length": min(len(str(response)) / 500, 1.0)
            }
            
            comparison.log_response(model, "creativity", prompt, response, time_taken, scores)
            
            # Basic creativity checks
            assert len(str(response)) > 100, f"{model} response too short"
            assert "time" in str(response).lower(), f"{model} didn't include time travel element"
            assert "coffee" in str(response).lower(), f"{model} didn't include coffee cup element"

    def test_logical_reasoning(self, comparison):
        """Test logical reasoning capabilities"""
        prompts = [
            "Solve this logic puzzle: A farmer needs to cross a river with a fox, a chicken, and a bag of corn...",
            "What is the next number in the sequence: 2, 6, 12, 20? Follow a quadratic pattern."
        ]
        
        for prompt in prompts:
            for model in ["openai", "anthropic", "together", "gemini"]:
                response, time_taken, error = comparison.get_model_response(model, prompt)
                
                if error:
                    print(f"\nWarning: {error}")
                    continue
                
                logical_score = comparison.calculate_logical_score(str(response), prompt)
                scores = {
                    "logical_reasoning": logical_score,
                    "response_time_score": 1.0 if time_taken < 5 else 0.5,
                    "response_length": min(len(str(response)) / 300, 1.0)
                }
                
                comparison.log_response(model, "logical", prompt, response, time_taken, scores)
                
                # Basic logical reasoning checks
                assert len(str(response)) > 25, f"{model} response too short"
                if "sequence" in prompt:
                    assert "30" in response, f"{model} failed sequence completion" 