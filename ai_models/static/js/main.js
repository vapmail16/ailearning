class ModelComparison {
    constructor() {
        this.submitBtn = document.getElementById('submit-btn');
        this.promptInput = document.getElementById('prompt-input');
        this.resultsSection = document.querySelector('.results-section');
        this.loadingSection = document.querySelector('.loading');
        
        this.setupEventListeners();
        this.lastResults = null;
    }

    setupEventListeners() {
        this.submitBtn.addEventListener('click', () => this.handleSubmit());
    }

    async handleSubmit() {
        const prompt = this.promptInput.value.trim();
        if (!prompt) {
            alert('Please enter a prompt');
            return;
        }

        // Show loading state
        this.loadingSection.style.display = 'block';
        this.submitBtn.disabled = true;

        try {
            const response = await fetch('/compare', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ prompt })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.lastResults = data;
            this.displayResults(data);
            
            // Initialize parameter testing with baseline data
            if (window.parameterTesting) {
                window.parameterTesting.initializeWithBaseline(data);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while comparing models');
        } finally {
            this.loadingSection.style.display = 'none';
            this.submitBtn.disabled = false;
        }
    }

    displayResults(data) {
        this.resultsSection.style.display = 'block';
        
        // Create results HTML
        this.resultsSection.innerHTML = `
            <h2>Model Responses</h2>
            <div class="responses-grid">
                ${Object.entries(data.responses).map(([model, response]) => `
                    <div class="response-card">
                        <h3>${model}</h3>
                        <div class="response-content">${response.text}</div>
                        <div class="metrics">
                            <div>Time: ${response.metrics.time_taken.toFixed(2)}s</div>
                            <div>Accuracy: ${(response.metrics.accuracy * 100).toFixed(1)}%</div>
                            <div>Creativity: ${(response.metrics.creativity_score * 100).toFixed(1)}%</div>
                            <div>Cost: $${response.metrics.cost.toFixed(4)}</div>
                        </div>
                    </div>
                `).join('')}
            </div>

            <h2>Performance Comparison</h2>
            <div class="comparison-matrix">
                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Response Time (s)</th>
                            <th>Accuracy</th>
                            <th>Creativity</th>
                            <th>Logical Score</th>
                            <th>Cost ($)</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${Object.entries(data.responses).map(([model, response]) => `
                            <tr>
                                <td>${model}</td>
                                <td>${response.metrics.time_taken.toFixed(2)}</td>
                                <td>${(response.metrics.accuracy * 100).toFixed(1)}%</td>
                                <td>${(response.metrics.creativity_score * 100).toFixed(1)}%</td>
                                <td>${(response.metrics.logical_score * 100).toFixed(1)}%</td>
                                <td>$${response.metrics.cost.toFixed(4)}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>

            <h2>Analysis & Conclusion</h2>
            <div class="analysis-section">
                <div class="analysis-grid">
                    <div class="analysis-item">
                        <strong>Fastest Model:</strong> ${data.summary.fastest_model}
                    </div>
                    <div class="analysis-item">
                        <strong>Most Accurate:</strong> ${data.summary.most_accurate_model}
                    </div>
                    <div class="analysis-item">
                        <strong>Overall Best Model:</strong> ${data.summary.best_overall_model}
                    </div>
                    <div class="analysis-text">
                        ${data.summary.analysis}
                    </div>
                </div>
            </div>
        `;
    }

    getLastResults() {
        return this.lastResults;
    }
}

// Initialize after DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    window.modelComparison = new ModelComparison();
}); 