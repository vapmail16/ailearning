import os
import time
import json
from datetime import datetime
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

        # Cost per 1000 tokens (approximate rates)
        self.cost_rates = {
            "openai": {
                "input": 0.0015,    # GPT-3.5-turbo input cost per 1K tokens
                "output": 0.002     # GPT-3.5-turbo output cost per 1K tokens
            },
            "anthropic": {
                "input": 0.015,     # Claude-3 Opus input cost per 1K tokens
                "output": 0.075     # Claude-3 Opus output cost per 1K tokens
            },
            "together": {
                "input": 0.0007,    # Mixtral-8x7B input cost per 1K tokens
                "output": 0.0007    # Mixtral-8x7B output cost per 1K tokens
            },
            "gemini": {
                "input": 0.001,     # Gemini Pro input cost per 1K tokens
                "output": 0.002     # Gemini Pro output cost per 1K tokens
            }
        }

    def calculate_cost(self, model, input_text, output_text):
        """Calculate cost for the API call"""
        try:
            # Approximate token count (rough estimation)
            input_tokens = len(input_text.split()) * 1.3  # Rough token estimation
            output_tokens = len(output_text.split()) * 1.3
            
            # Convert to thousands of tokens
            input_thousands = input_tokens / 1000
            output_thousands = output_tokens / 1000
            
            # Calculate cost
            cost = (
                input_thousands * self.cost_rates[model]["input"] +
                output_thousands * self.cost_rates[model]["output"]
            )
            
            return round(cost, 6)
        except Exception as e:
            print(f"Error calculating cost: {str(e)}")
            return 0.0

    def get_model_response(self, model, prompt, params=None):
        """Unified method for getting model responses with optional parameters"""
        start_time = time.time()
        try:
            # Use default parameters if none provided
            params = params or {
                'temperature': 0.7,
                'max_tokens': 1000,
                'top_p': 1.0
            }

            if model == "openai":
                response = self.openai_client.chat.completions.create(
                    model=MODEL_TIERS.get(params.get('tier'), {}).get('openai', "gpt-3.5-turbo"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=params.get('temperature', 0.7),
                    max_tokens=params.get('max_tokens', 1000),
                    top_p=params.get('top_p', 1.0)
                )
                response_text = response.choices[0].message.content

            elif model == "anthropic":
                response = self.anthropic_client.messages.create(
                    model=MODEL_TIERS.get(params.get('tier'), {}).get('anthropic', "claude-3-opus-20240229"),
                    max_tokens=params.get('max_tokens', 1000),
                    temperature=params.get('temperature', 0.7),
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.content[0].text

            elif model == "together":
                headers = {
                    "Authorization": f"Bearer {self.together_api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                    "prompt": prompt,
                    "max_tokens": params.get('max_tokens', 1000),
                    "temperature": params.get('temperature', 0.7),
                    "top_p": params.get('top_p', 1.0),
                    "top_k": 50,
                    "repetition_penalty": 1.1
                }
                response = requests.post(
                    "https://api.together.xyz/v1/completions",
                    headers=headers,
                    json=payload
                )
                if response.status_code == 200:
                    response_json = response.json()
                    response_text = response_json.get('choices', [{}])[0].get('text', '')
                else:
                    return None, 0, f"Together.ai error: {response.text}", 0.0

            elif model == "gemini":
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.gemini_api_key}"
                payload = {
                    "contents": [{"parts":[{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": params.get('temperature', 0.7),
                        "maxOutputTokens": params.get('max_tokens', 1000),
                        "topP": params.get('top_p', 1.0)
                    }
                }
                response = requests.post(url, json=payload)
                response_text = response.json()['candidates'][0]['content']['parts'][0]['text']

            time_taken = time.time() - start_time
            cost = self.calculate_cost(model, prompt, response_text)
            
            return {
                'response': response_text,
                'metrics': {
                    'time_taken': time_taken,
                    'accuracy': self.calculate_factual_score(response_text, prompt),
                    'creativity': self.calculate_creativity_score(response_text),
                    'logical_score': self.calculate_logical_score(response_text, prompt),
                    'cost': cost,
                    'length': len(response_text)
                }
            }

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            return {
                'response': error_msg,
                'metrics': {
                    'time_taken': time.time() - start_time,
                    'accuracy': 0,
                    'creativity': 0,
                    'logical_score': 0,
                    'cost': 0,
                    'length': 0
                }
            }

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
        if not response:
            return 0.0
            
        try:
            response_lower = str(response).lower()
            score = 0.0
            
            # Length score (up to 0.3)
            length_score = min(len(response_lower) / 1000, 0.3)
            score += length_score
            
            # Vocabulary diversity (up to 0.3)
            words = set(response_lower.split())
            vocab_score = min(len(words) / 100, 0.3)
            score += vocab_score
            
            # Structure and formatting (up to 0.2)
            structure_indicators = [
                '.' in response,  # Complete sentences
                ',' in response,  # Complex sentences
                '\n' in response,  # Paragraphs
                len(response_lower.split()) > 50  # Substantial content
            ]
            score += 0.2 * (sum(structure_indicators) / len(structure_indicators))
            
            # Engagement elements (up to 0.2)
            engagement_elements = [
                '?' in response,  # Questions
                '!' in response,  # Emphasis
                '"' in response,  # Quotations
                ':' in response   # Explanations
            ]
            score += 0.2 * (sum(engagement_elements) / len(engagement_elements))
            
            return min(score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            print(f"Error calculating creativity score: {str(e)}")
            return 0.0

    def calculate_logical_score(self, response, prompt):
        """Calculate logical reasoning score based on prompt type and response"""
        if not response:
            return 0.0
            
        try:
            response_lower = str(response).lower()
            prompt_lower = str(prompt).lower()
            score = 0.0
            
            # Structure score (up to 0.3)
            structure_elements = [
                'because' in response_lower,
                'therefore' in response_lower,
                'if' in response_lower,
                'then' in response_lower,
                'however' in response_lower
            ]
            score += 0.3 * (sum(structure_elements) / len(structure_elements))
            
            # Reasoning patterns (up to 0.4)
            reasoning_patterns = [
                'first' in response_lower,
                'second' in response_lower,
                'finally' in response_lower,
                'conclusion' in response_lower,
                'result' in response_lower
            ]
            score += 0.4 * (sum(reasoning_patterns) / len(reasoning_patterns))
            
            # Prompt-specific scoring (up to 0.3)
            if 'why' in prompt_lower or 'how' in prompt_lower:
                explanation_score = min(len(response_lower.split()) / 100, 0.3)
                score += explanation_score
            elif any(word in prompt_lower for word in ['solve', 'calculate', 'find']):
                numeric_score = 0.3 if any(char.isdigit() for char in response) else 0.0
                score += numeric_score
            else:
                relevance_score = 0.3 if any(word in response_lower for word in prompt_lower.split()) else 0.0
                score += relevance_score
            
            return min(score, 1.0)
            
        except Exception as e:
            print(f"Error calculating logical score: {str(e)}")
            return 0.0

    def log_response(self, model_name, prompt_type, prompt, response, time_taken, scores=None):
        """Log response and metrics to JSON file"""
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

    def run_parameter_tests(self, model, prompt, parameter_sets):
        """Run tests with different parameter combinations"""
        results = {
            'temperature_tests': [],
            'token_tests': [],
            'top_p_tests': []
        }
        
        # Test temperature variations
        for temp in parameter_sets['temperature']:
            response = self.get_model_response_with_params(
                model, prompt, {'temperature': temp}
            )
            results['temperature_tests'].append({
                'temperature': temp,
                'response': response,
                'metrics': {
                    'creativity': self.calculate_creativity_score(response),
                    'factual': self.calculate_factual_score(response, prompt),
                    'length': len(response)
                }
            })
        
        # Similar tests for max_tokens and top_p
        return results 

    def get_model_response_with_params(self, model, prompt, params):
        """Get response from model with specific parameters"""
        start_time = time.time()
        try:
            if model == "openai":
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=params.get('temperature', 0.7),
                    max_tokens=params.get('max_tokens', 1000),
                    top_p=params.get('top_p', 1.0)
                )
                response_text = response.choices[0].message.content

            elif model == "anthropic":
                response = self.anthropic_client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=params.get('max_tokens', 1000),
                    temperature=params.get('temperature', 0.7),
                    top_p=params.get('top_p', 1.0),
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.content[0].text

            elif model == "together":
                headers = {
                    "Authorization": f"Bearer {self.together_api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                    "prompt": prompt,
                    "max_tokens": params.get('max_tokens', 1000),
                    "temperature": params.get('temperature', 0.7),
                    "top_p": params.get('top_p', 1.0),
                    "top_k": 50,
                    "repetition_penalty": 1.1
                }
                response = requests.post(
                    "https://api.together.xyz/v1/completions",
                    headers=headers,
                    json=payload
                )
                if response.status_code == 200:
                    response_json = response.json()
                    response_text = response_json.get('choices', [{}])[0].get('text', '')
                else:
                    return None, 0, f"Together.ai error: {response.text}", 0.0

            elif model == "gemini":
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.gemini_api_key}"
                payload = {
                    "contents": [{"parts":[{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": params.get('temperature', 0.7),
                        "maxOutputTokens": params.get('max_tokens', 1000),
                        "topP": params.get('top_p', 1.0)
                    }
                }
                response = requests.post(url, json=payload)
                response_text = response.json()['candidates'][0]['content']['parts'][0]['text']

            time_taken = time.time() - start_time
            cost = self.calculate_cost(model, prompt, response_text)
            
            return {
                'response': response_text,
                'metrics': {
                    'time_taken': time_taken,
                    'accuracy': self.calculate_factual_score(response_text, prompt),
                    'creativity': self.calculate_creativity_score(response_text),
                    'cost': cost
                }
            }

        except Exception as e:
            return {
                'response': f"Error: {str(e)}",
                'metrics': {
                    'time_taken': time.time() - start_time,
                    'accuracy': 0,
                    'creativity': 0,
                    'cost': 0
                }
            }

    def compare_model_tiers(self, prompt, tier='tier_1'):
        """Compare models within the same tier"""
        results = {}
        for provider, model_name in MODEL_TIERS[tier].items():
            params = {'tier': tier}
            result = self.get_model_response(provider, prompt, params)
            
            if not result['response'].startswith('Error:'):
                results[provider] = {
                    'response': result['response'],
                    'metrics': result['metrics']
                }
        return results

MODEL_TIERS = {
    'tier_1': {  # Most powerful
        'openai': 'gpt-4',
        'anthropic': 'claude-3-opus-20240229',
        'together': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'gemini': 'gemini-ultra'
    },
    'tier_2': {  # Balanced
        'openai': 'gpt-3.5-turbo',
        'together': 'togethercomputer/llama-2-70b-chat',
        'gemini': 'gemini-pro'
    },
    'tier_3': {  # Fast/Efficient
        'together': 'togethercomputer/CodeLlama-34b-Python',
        'gemini': 'gemini-nano'
    }
}

def compare_model_tiers(self, prompt, tier='tier_1'):
    """Compare models within the same tier"""
    results = {}
    for provider, model_name in MODEL_TIERS[tier].items():
        params = {'tier': tier}
        result = self.get_model_response(provider, prompt, params)
        
        if not result['response'].startswith('Error:'):
            results[provider] = {
                'response': result['response'],
                'metrics': result['metrics']
            }
    return results 