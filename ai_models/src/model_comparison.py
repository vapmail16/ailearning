import os
import time
import json
from datetime import datetime
from dotenv import load_dotenv
import openai
from openai import OpenAI
import anthropic
from anthropic import AsyncAnthropic
import together
import requests
from typing import Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import cachetools
from src.performance_monitor import PerformanceMonitor
import aiohttp
import warnings

# Load environment variables
load_dotenv()

# Initialize performance monitor
performance_monitor = PerformanceMonitor()

# Add after imports
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='The use of together.api_key is deprecated')

class ModelComparison:
    def __init__(self):
        # Initialize clients
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.together_api_key = os.getenv('TOGETHER_API_KEY')
        self.gemini_api_key = os.getenv('GOOGLE_API_KEY')
        
        # Initialize Gemini
        import google.generativeai as genai
        genai.configure(api_key=self.gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-pro')
        
        # Initialize results directory
        self.results_dir = "test_results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        # Initialize cache with longer TTL and larger size
        self.response_cache = cachetools.TTLCache(maxsize=1000, ttl=3600 * 24)  # 24 hour cache
        
        # Initialize performance metrics
        self.performance_metrics = {
            'response_times': [],
            'token_usage': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'total_cost': 0.0
        }

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

    def calculate_metrics(self, response_text: str) -> Dict:
        """Calculate response metrics"""
        try:
            # Word count for basic length
            words = response_text.split()
            length = len(words)
            
            # Accuracy based on grammar and coherence
            accuracy = min(length / 100, 1.0)  # Simplified metric
            
            # Creativity based on unique words ratio
            unique_words = len(set(words))
            creativity = unique_words / length if length > 0 else 0
            
            # Logical score based on sentence structure
            sentences = response_text.count('.') + response_text.count('!') + response_text.count('?')
            logical_score = min(sentences / 5, 1.0)  # Simplified metric
            
            return {
                'accuracy': round(accuracy * 100, 2),
                'creativity': round(creativity * 100, 2),
                'logical_score': round(logical_score * 100, 2)
            }
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            return {
                'accuracy': 0.0,
                'creativity': 0.0,
                'logical_score': 0.0
            }

    def get_model_response(self, model: str, prompt: str, params: Dict = None) -> Dict:
        """Get response from specified model with optional parameters"""
        start_time = time.time()
        try:
            # Use params if provided, otherwise use defaults
            params = params or {}
            
            if model == "together":
                together.api_key = self.together_api_key
                response = together.Complete.create(
                    prompt=prompt,
                    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                    max_tokens=params.get('max_tokens', 1000),
                    temperature=params.get('temperature', 0.7)
                )
                
                # The response structure is directly in 'choices', not in 'output'
                if isinstance(response, dict) and 'choices' in response:
                    response_text = response['choices'][0]['text'].strip()
                    
                    # Get usage information
                    usage = response.get('usage', {})
                    token_count = usage.get('total_tokens', 0)
                    
                    # Calculate metrics
                    elapsed_time = time.time() - start_time
                    basic_metrics = self.calculate_metrics(response_text)
                    
                    metrics = {
                        'time_taken': elapsed_time,
                        'length': token_count,
                        'accuracy': basic_metrics['accuracy'],
                        'creativity': basic_metrics['creativity'],
                        'logical_score': basic_metrics['logical_score'],
                        'cost': self.calculate_cost(model, prompt, response_text)
                    }

                    return {
                        'response': response_text,
                        'metrics': metrics
                    }
                else:
                    raise Exception("Invalid Together API response format")
            
            elif model == "openai":
                response = self.openai_client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.choices[0].message.content
                
            elif model == "anthropic":
                response = self.anthropic_client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.content[0].text
                
            elif model == "gemini":
                response = self.gemini_model.generate_content(prompt)
                response_text = response.text
            
            # Calculate all metrics
            elapsed_time = time.time() - start_time
            basic_metrics = self.calculate_metrics(response_text)
            metrics = {
                'time_taken': elapsed_time,
                'length': len(response_text.split()),
                'accuracy': basic_metrics['accuracy'],
                'creativity': basic_metrics['creativity'],
                'logical_score': basic_metrics['logical_score'],
                'cost': self.calculate_cost(model, prompt, response_text)
            }

            return {
                'response': response_text,
                'metrics': metrics
            }

        except Exception as e:
            print(f"Error in get_model_response for {model}: {str(e)}")
            return {
                'response': f"Error: {str(e)}",
                'metrics': {
                    'time_taken': time.time() - start_time,
                    'length': 0,
                    'accuracy': 0,
                    'creativity': 0,
                    'logical_score': 0,
                    'cost': 0.0
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
                    messages=[{"role": "user", "content": prompt}],
                    cache_control={"ttl": params.get('cache_ttl', 3600)}  # Enable explicit caching
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

    async def get_model_response_streaming(self, model: str, prompt: str, 
                                         params: Optional[Dict[str, Any]] = None) -> Dict:
        """Streaming version of get_model_response"""
        cache_key = f"{model}:{prompt}:{str(params)}"
        
        # Check cache first
        if cache_key in self.response_cache:
            self.performance_metrics['cache_hits'] += 1
            return self.response_cache[cache_key]
            
        self.performance_metrics['cache_misses'] += 1
        start_time = time.time()
        
        try:
            if model == "openai":
                response = self.openai_client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[{"role": "user", "content": prompt}],
                    stream=True
                )
                
            elif model == "anthropic":
                response = self.anthropic_client.messages.create(
                    model="claude-3-opus-20240229",
                    messages=[{"role": "user", "content": prompt}],
                    stream=True
                )
                
            # Add streaming response processing...
            
            elapsed_time = time.time() - start_time
            self.performance_metrics['response_times'].append(elapsed_time)
            
            result = {
                'response': response_text,
                'metrics': {
                    'time_taken': elapsed_time,
                    'tokens': token_count,
                    'cost': self.calculate_cost(model, prompt, response_text)
                }
            }
            
            # Cache the result
            self.response_cache[cache_key] = result
            return result
            
        except Exception as e:
            return {'response': f"Error: {str(e)}", 'metrics': {'time_taken': time.time() - start_time}}

    def get_streaming_response(self, model, prompt):
        """Get streaming response with proper formatting"""
        try:
            if model == "openai":
                response = self.openai_client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[{"role": "user", "content": prompt}],
                    stream=True
                )
                
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield {
                            'model': 'openai',
                            'content': chunk.choices[0].delta.content,
                            'done': False
                        }
                yield {'model': 'openai', 'content': '', 'done': True}
                
            elif model == "anthropic":
                with self.anthropic_client.messages.stream(
                    model="claude-3-opus-20240229",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                ) as stream:
                    for chunk in stream:
                        if chunk.delta.text:
                            yield {
                                'model': 'anthropic',
                                'content': chunk.delta.text,
                                'done': False
                            }
                    yield {'model': 'anthropic', 'content': '', 'done': True}
                
            elif model == "gemini":
                response = self.gemini_model.generate_content(
                    prompt,
                    stream=True
                )
                for chunk in response:
                    if chunk.text:
                        yield {
                            'model': 'gemini',
                            'content': chunk.text,
                            'done': False
                        }
                yield {'model': 'gemini', 'content': '', 'done': True}
                
        except Exception as e:
            error_msg = f"Error in streaming: {str(e)}"
            print(error_msg)  # Log the error
            yield {
                'model': model,
                'content': error_msg,
                'done': True,
                'error': True
            }

    async def get_model_response_async(self, model, prompt):
        """Async version of get_model_response"""
        start_time = time.time()
        try:
            if model == "openai":
                async with OpenAI(api_key=os.getenv('OPENAI_API_KEY')) as client:
                    response = await client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    response_text = response.choices[0].message.content
                    
            elif model == "anthropic":
                async with AsyncAnthropic(api_key=os.getenv('ANTHROPIC_API_KEY')) as client:
                    response = await client.messages.create(
                        model="claude-3-opus-20240229",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    response_text = response.content[0].text
                    
            # Calculate metrics
            elapsed_time = time.time() - start_time
            metrics = {
                'time_taken': elapsed_time,
                'length': len(response_text.split()),
                'accuracy': 0.0,
                'creativity': 0.0,
                'logical_score': 0.0,
                'cost': self.calculate_cost(model, prompt, response_text)
            }

            return {
                'response': response_text,
                'metrics': metrics
            }

        except Exception as e:
            print(f"Error in async response for {model}: {str(e)}")
            return {
                'response': f"Error: {str(e)}",
                'metrics': {
                    'time_taken': time.time() - start_time,
                    'length': 0,
                    'accuracy': 0,
                    'creativity': 0,
                    'logical_score': 0,
                    'cost': 0.0
                }
            }

    def batch_process_requests(self, prompts, model):
        """Process multiple requests with caching"""
        try:
            results = []
            total_cost = 0
            cache_stats = {'hits': 0, 'misses': 0}
            start_time = time.time()
            
            # Create a unique cache key for each prompt
            for prompt in prompts:
                cache_key = f"{model}:{prompt}"
                cached_result = self.response_cache.get(cache_key)
                
                if cached_result:
                    print(f"Cache hit for prompt: {prompt[:50]}...")
                    results.append(cached_result)
                    cache_stats['hits'] += 1
                    continue
                
                print(f"Cache miss for prompt: {prompt[:50]}...")
                result = self.get_model_response(model, prompt)
                if result and 'metrics' in result:
                    total_cost += result['metrics'].get('cost', 0)
                    # Cache the result with the unique key
                    self.response_cache[cache_key] = result
                    cache_stats['misses'] += 1
                results.append(result)
            
            total_time = time.time() - start_time
            
            print(f"Cache stats - Hits: {cache_stats['hits']}, Misses: {cache_stats['misses']}")
            
            return {
                'results': results,
                'metrics': {
                    'total_time': total_time,
                    'average_time': total_time / len(prompts),
                    'total_cost': total_cost,
                    'average_cost': total_cost / len(prompts)
                },
                'cache_stats': cache_stats
            }
        except Exception as e:
            print(f"Error in batch processing: {str(e)}")
            return None

    def calculate_batch_metrics(self, batch_results):
        """Calculate cost and performance metrics for batch processing"""
        try:
            # Only count costs for cache misses
            total_cost = batch_results['metrics']['total_cost']
            total_time = batch_results['metrics']['total_time']
            num_requests = len(batch_results['results'])
            cache_hits = batch_results['cache_stats']['hits']
            cache_misses = batch_results['cache_stats']['misses']
            
            return {
                'total_cost': total_cost,
                'average_cost': total_cost / num_requests if num_requests > 0 else 0,
                'total_time': total_time,
                'average_time': total_time / num_requests if num_requests > 0 else 0,
                'requests_completed': num_requests,
                'cache_stats': {
                    'hits': cache_hits,
                    'misses': cache_misses,
                    'hit_rate': (cache_hits / num_requests * 100) if num_requests > 0 else 0
                }
            }
        except Exception as e:
            print(f"Error calculating batch metrics: {str(e)}")
            return {
                'total_cost': 0,
                'average_cost': 0,
                'total_time': 0,
                'average_time': 0,
                'requests_completed': 0,
                'cache_stats': {'hits': 0, 'misses': 0, 'hit_rate': 0}
            }

    def get_cached_response(self, model, prompt, cache_ttl=3600):
        """Get response with explicit caching for supported models"""
        cache_key = f"{model}:{prompt}"
        
        if model == "anthropic":
            response = self.anthropic_client.messages.create(
                model="claude-3-opus-20240229",
                messages=[{"role": "user", "content": prompt}],
                cache_control={"ttl": cache_ttl}  # Enable explicit caching
            )
        # ... handle other models

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