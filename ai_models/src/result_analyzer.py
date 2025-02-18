import json
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt

class ResultAnalyzer:
    def __init__(self, results_dir="test_results", clean_directory=False):
        self.results_dir = results_dir
        if clean_directory:
            self.cleanup_and_initialize_results_dir()
        elif not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            print(f"Created results directory at {self.results_dir}")

    def cleanup_and_initialize_results_dir(self):
        """Clean up old results and initialize results directory"""
        if os.path.exists(self.results_dir):
            shutil.rmtree(self.results_dir)
        os.makedirs(self.results_dir)
        print(f"Initialized clean results directory at {self.results_dir}")

    def load_results(self):
        results = []
        for filename in os.listdir(self.results_dir):
            if filename.endswith(".json"):
                with open(os.path.join(self.results_dir, filename), 'r') as f:
                    results.append(json.load(f))
        return results

    def analyze_response_times(self):
        results = self.load_results()
        if not results:
            print("No results found to analyze")
            return
            
        df = pd.DataFrame(results)
        
        plt.figure(figsize=(10, 6))
        df.boxplot(column='time_taken', by='model')
        plt.title('Response Time by Model')
        plt.suptitle('')  # This removes the automatic suptitle
        plt.ylabel('Time (seconds)')
        plt.savefig(os.path.join(self.results_dir, 'response_times.png'))
        plt.close()

    def analyze_response_lengths(self):
        results = self.load_results()
        if not results:
            print("No results found to analyze")
            return
            
        df = pd.DataFrame(results)
        # Convert response to string and handle None/NaN values
        df['response_length'] = df['response'].astype(str).apply(lambda x: len(x) if x != 'nan' else 0)
        
        plt.figure(figsize=(10, 6))
        df.boxplot(column='response_length', by='model')
        plt.title('Response Length by Model')
        plt.suptitle('')
        plt.ylabel('Length (characters)')
        plt.savefig(os.path.join(self.results_dir, 'response_lengths.png'))
        plt.close()

    def analyze_scores(self):
        """Analyze and visualize scores across different categories"""
        results = self.load_results()
        if not results:
            print("No results found to analyze")
            return
            
        df = pd.DataFrame(results)
        
        # Extract scores into separate columns
        score_categories = ['factual_accuracy', 'creativity', 'logical_reasoning', 
                          'response_time_score', 'response_length']
        
        # Initialize score columns with 0
        for category in score_categories:
            df[category] = 0.0
            
        # Safely extract scores
        for idx, row in df.iterrows():
            scores = row.get('scores', {})
            if isinstance(scores, dict):
                for category in score_categories:
                    df.at[idx, category] = scores.get(category, 0.0)
        
        # Create score comparison plot
        plt.figure(figsize=(12, 8))
        df.boxplot(column=score_categories, by='model')
        plt.title('Score Comparison Across Categories')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'score_comparison.png'))
        plt.close()

    def generate_report(self):
        results = self.load_results()
        if not results:
            print("No results found to generate report")
            return
            
        df = pd.DataFrame(results)
        
        # Calculate average scores per model
        score_categories = ['factual_accuracy', 'creativity', 'logical_reasoning', 
                          'response_time_score', 'response_length']
        
        scores_by_model = {}
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            scores_by_model[model] = {}
            
            # Safely extract scores for each category
            for category in score_categories:
                scores = []
                for _, row in model_data.iterrows():
                    if isinstance(row['scores'], dict):
                        scores.append(row['scores'].get(category, 0.0))
                    else:
                        scores.append(0.0)
                scores_by_model[model][category] = sum(scores) / len(scores) if scores else 0.0
        
        # Calculate average response length properly
        avg_lengths = {}
        for model in df['model'].unique():
            model_responses = df[df['model'] == model]['response']
            lengths = [len(str(r)) for r in model_responses]
            avg_lengths[model] = sum(lengths) / len(lengths) if lengths else 0
        
        report = {
            "total_tests": len(results),
            "tests_per_model": df['model'].value_counts().to_dict(),
            "average_response_times": df.groupby('model')['time_taken'].mean().to_dict(),
            "average_response_lengths": avg_lengths,
            "scores_by_model": scores_by_model,
            "overall_model_scores": {
                model: sum(scores.values()) / len(scores)
                for model, scores in scores_by_model.items()
            }
        }
        
        with open(os.path.join(self.results_dir, 'analysis_report.json'), 'w') as f:
            json.dump(report, f, indent=2)

def main():
    """Main function to run the analysis"""
    print("Starting AI Model Analysis...")
    
    # Initialize analyzer without cleaning the directory
    analyzer = ResultAnalyzer(clean_directory=False)
    print("\nAnalyzing results...")
    
    try:
        # Run all analyses
        analyzer.analyze_response_times()
        analyzer.analyze_response_lengths()
        analyzer.analyze_scores()
        analyzer.generate_report()
        
        # Print summary
        results = analyzer.load_results()
        if results:
            df = pd.DataFrame(results)
            print("\nSummary:")
            print(f"Total tests run: {len(results)}")
            print("\nTests per model:")
            print(df['model'].value_counts())
            print("\nAverage response times (seconds):")
            print(df.groupby('model')['time_taken'].mean())
            
            # Check if analysis files were created
            files_created = os.listdir(analyzer.results_dir)
            print(f"\nFiles generated in {analyzer.results_dir}:")
            for file in files_created:
                print(f"- {file}")
        else:
            print("\nNo test results found to analyze. Please run the tests first using:")
            print("pytest tests/test_model_comparison.py -v")
            
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 