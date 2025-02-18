class ParameterTesting {
    constructor() {
        this.setupEventListeners();
        this.charts = {};
        this.testHistory = [];
        this.promptInput = document.getElementById('prompt-input');
        this.baselineResults = null;
        this.initializeCharts();
    }

    setupEventListeners() {
        // Test type selection
        const testType = document.getElementById('test-type');
        const parameterOptions = document.getElementById('parameter-options');
        const tierOptions = document.getElementById('tier-options');
        
        if (testType) {
            testType.addEventListener('change', async (e) => {
                if (e.target.value === 'parameter') {
                    parameterOptions.style.display = 'block';
                    tierOptions.style.display = 'none';
                    if (this.baselineResults) {
                        this.displayParameterMetrics();
                    }
                } else if (e.target.value === 'tier') {
                    parameterOptions.style.display = 'none';
                    tierOptions.style.display = 'block';
                    if (this.baselineResults && this.promptInput.value.trim()) {
                        await this.fetchAndDisplayTierResults('tier_1');
                    }
                }
            });
        }

        // Tier selection
        const tierSelect = document.getElementById('tier-select');
        if (tierSelect) {
            tierSelect.addEventListener('change', async (e) => {
                if (this.promptInput.value.trim()) {
                    await this.fetchAndDisplayTierResults(e.target.value);
                }
            });
        }

        // Initialize parameter controls
        this.initializeParameterControls();

        // Add Run Test button handler
        const runTestBtn = document.getElementById('run-test-btn');
        if (runTestBtn) {
            runTestBtn.addEventListener('click', () => this.runParameterTest());
        }
    }

    initializeParameterControls() {
        const temperatureSlider = document.getElementById('temperature');
        const maxTokensSlider = document.getElementById('max-tokens');
        const topPSlider = document.getElementById('top-p');

        if (temperatureSlider) {
            temperatureSlider.addEventListener('input', (e) => {
                document.getElementById('temp-value').textContent = e.target.value;
                this.updateParameterResults();
            });
        }

        if (maxTokensSlider) {
            maxTokensSlider.addEventListener('input', (e) => {
                document.getElementById('tokens-value').textContent = e.target.value;
                this.updateParameterResults();
            });
        }

        if (topPSlider) {
            topPSlider.addEventListener('input', (e) => {
                document.getElementById('top-p-value').textContent = e.target.value;
                this.updateParameterResults();
            });
        }
    }

    async fetchAndDisplayTierResults(selectedTier) {
        if (!this.promptInput.value.trim()) {
            alert('Please enter a prompt first');
            return;
        }

        const tierLoading = document.querySelector('.parameter-loading');
        const loadingText = tierLoading.querySelector('p');
        loadingText.textContent = 'Comparing models in tier...';
        tierLoading.style.display = 'block';

        try {
            const response = await fetch('/compare-tiers', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: this.promptInput.value.trim(),
                    tier: selectedTier
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            // Display results in the table
            const tbody = document.getElementById('results-body');
            let rows = '';

            Object.entries(data.responses).forEach(([model, response]) => {
                rows += `
                    <tr>
                        <td>${model}</td>
                        <td>${this.truncateText(response.text, 100)}</td>
                        <td>${response.metrics.time_taken.toFixed(2)}s</td>
                        <td>${(response.metrics.accuracy * 100).toFixed(1)}%</td>
                        <td>${(response.metrics.creativity_score * 100).toFixed(1)}%</td>
                        <td>${(response.metrics.logical_score * 100).toFixed(1)}%</td>
                        <td>$${response.metrics.cost.toFixed(4)}</td>
                    </tr>
                `;
            });

            tbody.innerHTML = rows;

            // Hide parameter summary
            const parameterSummary = document.querySelector('.parameter-summary');
            if (parameterSummary) {
                parameterSummary.style.display = 'none';
            }

        } catch (error) {
            console.error('Error:', error);
            alert('Error comparing model tiers');
        } finally {
            const tierLoading = document.querySelector('.parameter-loading');
            tierLoading.style.display = 'none';
            loadingText.textContent = 'Running parameter tests...'; // Reset loading text
        }
    }

    displayTierMetrics(selectedTier) {
        if (!this.baselineResults || !this.baselineResults.responses) return;
        
        const tbody = document.getElementById('results-body');
        const tierModels = this.getTierModels(selectedTier);
        let rows = '';

        // Filter and sort models based on tier
        const tierResponses = Object.entries(this.baselineResults.responses)
            .filter(([model]) => tierModels.includes(model))
            .sort((a, b) => {
                // Sort by accuracy, then by cost
                const accuracyDiff = b[1].metrics.accuracy - a[1].metrics.accuracy;
                if (accuracyDiff !== 0) return accuracyDiff;
                return a[1].metrics.cost - b[1].metrics.cost;
            });

        // Generate table rows
        tierResponses.forEach(([model, response]) => {
            rows += `
                <tr>
                    <td>${model}</td>
                    <td>${this.truncateText(response.text, 100)}</td>
                    <td>${response.metrics.time_taken.toFixed(2)}s</td>
                    <td>${(response.metrics.accuracy * 100).toFixed(1)}%</td>
                    <td>${(response.metrics.creativity_score * 100).toFixed(1)}%</td>
                    <td>${(response.metrics.logical_score * 100).toFixed(1)}%</td>
                    <td>$${response.metrics.cost.toFixed(4)}</td>
                    <td>-</td>
                </tr>
            `;
        });

        tbody.innerHTML = rows;

        // Hide parameter summary when showing tier comparison
        const parameterSummary = document.querySelector('.parameter-summary');
        if (parameterSummary) {
            parameterSummary.style.display = 'none';
        }
    }

    getTierModels(tier) {
        const tiers = {
            'tier_1': ['openai', 'anthropic', 'together', 'gemini'],
            'tier_2': ['openai', 'anthropic', 'together'],
            'tier_3': ['openai', 'together']
        };
        return tiers[tier] || [];
    }

    initializeWithBaseline(baselineData) {
        this.baselineResults = baselineData;
        const testType = document.getElementById('test-type');
        
        // Show appropriate view based on current test type
        if (testType && testType.value === 'tier') {
            const tierSelect = document.getElementById('tier-select');
            const parameterOptions = document.getElementById('parameter-options');
            const tierOptions = document.getElementById('tier-options');
            
            if (tierSelect) {
                parameterOptions.style.display = 'none';
                tierOptions.style.display = 'block';
                this.displayTierMetrics(tierSelect.value);
            }
        }
    }

    truncateText(text, maxLength) {
        if (!text) return '';
        return text.length > maxLength ? 
            text.substring(0, maxLength) + '...' : 
            text;
    }

    displayAllModelMetrics(data) {
        if (!data || !data.responses) return;

        const tbody = document.getElementById('results-body');
        let rows = '';

        // Display baseline metrics for all models
        Object.entries(data.responses).forEach(([model, response]) => {
            rows += `
                <tr class="baseline-row">
                    <td>${model}</td>
                    <td>Baseline</td>
                    <td>Default</td>
                    <td>${this.truncateText(response.text, 100)}</td>
                    <td>${response.metrics.time_taken.toFixed(2)}s</td>
                    <td>${(response.metrics.accuracy * 100).toFixed(1)}%</td>
                    <td>${(response.metrics.creativity_score * 100).toFixed(1)}%</td>
                    <td>${(response.metrics.logical_score * 100).toFixed(1)}%</td>
                    <td>$${response.metrics.cost.toFixed(4)}</td>
                    <td>-</td>
                </tr>
            `;
        });

        tbody.innerHTML = rows;
    }

    async runParameterTest() {
        if (!this.promptInput.value.trim()) {
            alert('Please enter a prompt first');
            return;
        }

        const selectedModel = document.getElementById('model-select').value;
        const parameters = {
            temperature: parseFloat(document.getElementById('temperature').value),
            max_tokens: parseInt(document.getElementById('max-tokens').value),
            top_p: parseFloat(document.getElementById('top-p').value)
        };

        const runTestBtn = document.getElementById('run-test-btn');
        const parameterLoading = document.querySelector('.parameter-loading');
        
        try {
            runTestBtn.disabled = true;
            parameterLoading.style.display = 'block';

            // Create an array of parameter tests to run
            const parameterTests = [
                { parameter: 'temperature', value: parameters.temperature },
                { parameter: 'max_tokens', value: parameters.max_tokens },
                { parameter: 'top_p', value: parameters.top_p }
            ];

            const response = await fetch('/parameter-test', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: this.promptInput.value.trim(),
                    model: selectedModel,
                    parameters: parameters,
                    parameterTests: parameterTests  // Add this to ensure all parameters are tested
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            if (data.error) {
                throw new Error(data.error);
            }

            // Update the results table with the new test results
            const tbody = document.getElementById('results-body');
            const baselineMetrics = this.baselineResults.responses[selectedModel].metrics;
            
            let rows = '';
            
            // Baseline row
            rows += `
                <tr class="baseline-row">
                    <td>${selectedModel}</td>
                    <td>Baseline</td>
                    <td>Default</td>
                    <td>${this.truncateText(this.baselineResults.responses[selectedModel].text, 100)}</td>
                    <td>${baselineMetrics.time_taken.toFixed(2)}s</td>
                    <td>${(baselineMetrics.accuracy * 100).toFixed(1)}%</td>
                    <td>${(baselineMetrics.creativity_score * 100).toFixed(1)}%</td>
                    <td>${(baselineMetrics.logical_score * 100).toFixed(1)}%</td>
                    <td>$${baselineMetrics.cost.toFixed(4)}</td>
                    <td>-</td>
                </tr>
            `;

            // Add test result rows
            data.results.forEach(result => {
                const changes = this.calculateChanges(result.metrics, baselineMetrics);
                rows += `
                    <tr>
                        <td>${selectedModel}</td>
                        <td>${result.parameter}</td>
                        <td>${result.value}</td>
                        <td>${this.truncateText(result.response, 100)}</td>
                        <td>${result.metrics.time_taken.toFixed(2)}s</td>
                        <td>${(result.metrics.accuracy * 100).toFixed(1)}%</td>
                        <td>${(result.metrics.creativity_score * 100).toFixed(1)}%</td>
                        <td>${(result.metrics.logical_score * 100).toFixed(1)}%</td>
                        <td>$${result.metrics.cost.toFixed(4)}</td>
                        <td>${this.formatChanges(changes)}</td>
                    </tr>
                `;
            });

            tbody.innerHTML = rows;

            // Update the parameter impact summary
            this.updateParameterSummary(data, parameters);

        } catch (error) {
            console.error('Error:', error);
            alert(`Error running parameter test: ${error.message}`);
        } finally {
            runTestBtn.disabled = false;
            parameterLoading.style.display = 'none';
        }
    }

    calculateChanges(newMetrics, baselineMetrics) {
        return {
            accuracy: ((newMetrics.accuracy / baselineMetrics.accuracy) - 1) * 100,
            creativity: ((newMetrics.creativity_score / baselineMetrics.creativity_score) - 1) * 100,
            time: ((newMetrics.time_taken / baselineMetrics.time_taken) - 1) * 100,
            cost: ((newMetrics.cost / baselineMetrics.cost) - 1) * 100
        };
    }

    formatChanges(changes) {
        return `
            <div class="metric-change ${changes.accuracy >= 0 ? 'positive' : 'negative'}">
                Accuracy: ${changes.accuracy.toFixed(1)}%
            </div>
            <div class="metric-change ${changes.creativity >= 0 ? 'positive' : 'negative'}">
                Creativity: ${changes.creativity.toFixed(1)}%
            </div>
            <div class="metric-change ${changes.time <= 0 ? 'positive' : 'negative'}">
                Time: ${changes.time.toFixed(1)}%
            </div>
            <div class="metric-change ${changes.cost <= 0 ? 'positive' : 'negative'}">
                Cost: ${changes.cost.toFixed(1)}%
            </div>
        `;
    }

    updateParameterSummary(data, parameters) {
        const temperatureSummary = document.getElementById('temperature-summary');
        const tokensSummary = document.getElementById('tokens-summary');
        const costSummary = document.getElementById('cost-summary');

        // Analyze temperature impact
        const tempResults = data.results.filter(r => r.parameter === 'temperature');
        if (tempResults.length > 0) {
            const tempAnalysis = this.analyzeParameterImpact(tempResults, this.baselineResults.responses[data.model]);
            temperatureSummary.innerHTML = `
                <p>Optimal temperature: ${tempAnalysis.optimal}</p>
                <p>Impact on:</p>
                <ul>
                    <li>Accuracy: ${tempAnalysis.impacts.accuracy}</li>
                    <li>Creativity: ${tempAnalysis.impacts.creativity}</li>
                    <li>Response Time: ${tempAnalysis.impacts.time}</li>
                </ul>
            `;
        }

        // Analyze token impact
        const tokenResults = data.results.filter(r => r.parameter === 'max_tokens');
        if (tokenResults.length > 0) {
            const tokenAnalysis = this.analyzeParameterImpact(tokenResults, this.baselineResults.responses[data.model]);
            tokensSummary.innerHTML = `
                <p>Optimal token length: ${tokenAnalysis.optimal}</p>
                <p>Impact on:</p>
                <ul>
                    <li>Response Quality: ${tokenAnalysis.impacts.quality}</li>
                    <li>Processing Time: ${tokenAnalysis.impacts.time}</li>
                    <li>Cost Efficiency: ${tokenAnalysis.impacts.cost}</li>
                </ul>
            `;
        }

        // Analyze cost-performance ratio
        costSummary.innerHTML = this.analyzeCostPerformance(data.results, this.baselineResults.responses[data.model]);
    }

    analyzeParameterImpact(results, baseline) {
        // Find the result with the best overall performance
        const optimal = results.reduce((best, current) => {
            const score = (
                (current.metrics.accuracy / baseline.metrics.accuracy) +
                (current.metrics.creativity / baseline.metrics.creativity_score) +
                (baseline.metrics.time_taken / current.metrics.time_taken)
            ) / 3;
            
            return score > best.score ? { result: current, score } : best;
        }, { result: results[0], score: 0 }).result;

        return {
            optimal: optimal.value,
            impacts: {
                accuracy: this.formatImpact(optimal.metrics.accuracy, baseline.metrics.accuracy),
                creativity: this.formatImpact(optimal.metrics.creativity, baseline.metrics.creativity_score),
                time: this.formatImpact(baseline.metrics.time_taken, optimal.metrics.time_taken),
                quality: this.formatImpact(optimal.metrics.accuracy * optimal.metrics.creativity, 
                                         baseline.metrics.accuracy * baseline.metrics.creativity_score),
                cost: this.formatImpact(baseline.metrics.cost, optimal.metrics.cost)
            }
        };
    }

    formatImpact(new_value, baseline_value) {
        const change = ((new_value / baseline_value) - 1) * 100;
        return `${change > 0 ? '+' : ''}${change.toFixed(1)}%`;
    }

    analyzeCostPerformance(results, baseline) {
        const bestValue = results.reduce((best, current) => {
            const performanceScore = (current.metrics.accuracy + current.metrics.creativity) / 2;
            const costEfficiency = performanceScore / current.metrics.cost;
            return costEfficiency > best.efficiency ? 
                { result: current, efficiency: costEfficiency } : best;
        }, { result: results[0], efficiency: 0 }).result;

        return `
            <p>Most cost-effective settings:</p>
            <ul>
                <li>${bestValue.parameter}: ${bestValue.value}</li>
                <li>Performance/Cost Ratio: ${(bestValue.metrics.accuracy / bestValue.metrics.cost).toFixed(2)}</li>
                <li>Cost Savings: ${this.formatImpact(baseline.metrics.cost, bestValue.metrics.cost)}</li>
            </ul>
            <p>Recommendation: ${this.generateRecommendation(bestValue, baseline)}</p>
        `;
    }

    generateRecommendation(bestResult, baseline) {
        const performanceChange = ((bestResult.metrics.accuracy / baseline.metrics.accuracy) - 1) * 100;
        const costChange = ((bestResult.metrics.cost / baseline.metrics.cost) - 1) * 100;

        if (performanceChange > 0 && costChange < 0) {
            return `Strongly recommended - Better performance at lower cost`;
        } else if (performanceChange > 0) {
            return `Consider if accuracy is priority - Better performance but higher cost`;
        } else if (costChange < 0) {
            return `Consider if cost is priority - Lower cost but slightly reduced performance`;
        } else {
            return `Stick with baseline settings for this use case`;
        }
    }

    async updateCharts(data) {
        if (!data || !data.results) return;

        const labels = data.results.map(r => r.parameter_value);
        
        // Update performance chart
        if (this.charts.performance) {
            this.charts.performance.data = {
                labels: labels,
                datasets: [{
                    label: 'Accuracy',
                    data: data.results.map(r => r.metrics.accuracy * 100),
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }, {
                    label: 'Creativity',
                    data: data.results.map(r => r.metrics.creativity_score * 100),
                    borderColor: 'rgb(255, 99, 132)',
                    tension: 0.1
                }, {
                    label: 'Logical Score',
                    data: data.results.map(r => r.metrics.logical_score * 100),
                    borderColor: 'rgb(153, 102, 255)',
                    tension: 0.1
                }]
            };
            this.charts.performance.update();
        }

        // Update response characteristics chart
        if (this.charts.response) {
            this.charts.response.data = {
                labels: labels,
                datasets: [{
                    label: 'Response Time (s)',
                    data: data.results.map(r => r.metrics.time_taken),
                    borderColor: 'rgb(75, 192, 192)',
                    yAxisID: 'y'
                }, {
                    label: 'Cost ($)',
                    data: data.results.map(r => r.metrics.cost),
                    borderColor: 'rgb(255, 99, 132)',
                    yAxisID: 'y1'
                }]
            };
            this.charts.response.update();
        }
    }

    saveToHistory(data) {
        this.testHistory.push({
            timestamp: new Date(),
            ...data
        });
        this.updateHistoricalChart();
    }

    updateHistoricalChart() {
        const ctx = document.getElementById('historical-chart').getContext('2d');
        
        if (this.charts.historical) {
            this.charts.historical.destroy();
        }

        const historicalData = this.testHistory.map(test => ({
            timestamp: test.timestamp,
            accuracy: test.results.reduce((acc, r) => acc + r.metrics.accuracy, 0) / test.results.length,
            cost: test.results.reduce((acc, r) => acc + r.metrics.cost, 0)
        }));

        this.charts.historical = new Chart(ctx, {
            type: 'line',
            data: {
                labels: historicalData.map(d => d.timestamp.toLocaleTimeString()),
                datasets: [{
                    label: 'Average Accuracy',
                    data: historicalData.map(d => d.accuracy * 100),
                    borderColor: 'rgb(75, 192, 192)',
                    yAxisID: 'y'
                }, {
                    label: 'Total Cost',
                    data: historicalData.map(d => d.cost),
                    borderColor: 'rgb(255, 99, 132)',
                    yAxisID: 'y1'
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Historical Performance'
                    }
                },
                scales: {
                    y: {
                        type: 'linear',
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Accuracy (%)'
                        }
                    },
                    y1: {
                        type: 'linear',
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Cost ($)'
                        }
                    }
                }
            }
        });
    }

    initializeCharts() {
        // Performance comparison chart
        const perfCtx = document.getElementById('performance-comparison-chart');
        if (perfCtx) {
            this.charts.performance = new Chart(perfCtx, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: []
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Performance Metrics Comparison'
                        },
                        legend: {
                            position: 'top'
                        }
                    }
                }
            });
        }

        // Response characteristics chart
        const respCtx = document.getElementById('response-comparison-chart');
        if (respCtx) {
            this.charts.response = new Chart(respCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: []
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Response Characteristics'
                        },
                        legend: {
                            position: 'top'
                        }
                    }
                }
            });
        }
    }
}

// Initialize after DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    if (!window.modelComparison) {
        window.modelComparison = new ModelComparison();
    }
    window.parameterTesting = new ParameterTesting();
}); 