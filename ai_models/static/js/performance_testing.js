class PerformanceTesting {
    constructor() {
        console.log("Initializing PerformanceTesting");  // Debug log
        this.initializeEventListeners();
        // Show batch processing by default
        this.showTestSection('batch');
    }

    initializeEventListeners() {
        console.log("Setting up event listeners");
        
        const testTypeSelect = document.getElementById('performance-test-type');
        const runLatencyTestBtn = document.getElementById('run-latency-test');
        const runBatchTestBtn = document.getElementById('run-batch-test');
        const runStreamingTestBtn = document.getElementById('run-streaming-test');
        
        if (testTypeSelect) {
            testTypeSelect.addEventListener('change', (e) => {
                console.log("Test type changed to:", e.target.value);
                this.showTestSection(e.target.value);
            });
        } else {
            console.error("Could not find performance-test-type select element");
        }

        if (runLatencyTestBtn) {
            runLatencyTestBtn.addEventListener('click', () => {
                console.log("Running latency test");
                this.runLatencyTest();
            });
        }

        if (runBatchTestBtn) {
            runBatchTestBtn.addEventListener('click', () => {
                console.log("Running batch test");
                this.runBatchTest();
            });
        }

        if (runStreamingTestBtn) {
            runStreamingTestBtn.addEventListener('click', () => {
                console.log("Running streaming test");
                this.runStreamingTest();
            });
        }

        // Parameter testing event listeners
        const runParameterTestBtn = document.getElementById('run-parameter-test');
        const temperatureSlider = document.getElementById('temperature');
        const maxTokensSlider = document.getElementById('max-tokens');
        const topPSlider = document.getElementById('top-p');

        if (runParameterTestBtn) {
            runParameterTestBtn.addEventListener('click', () => {
                console.log("Running parameter test");
                this.runParameterTest();
            });
        }

        // Update parameter values display
        const updateParamValue = (slider) => {
            if (slider) {
                const valueDisplay = slider.nextElementSibling;
                if (valueDisplay) {
                    valueDisplay.textContent = slider.value;
                }
                slider.addEventListener('input', (e) => {
                    if (valueDisplay) {
                        valueDisplay.textContent = e.target.value;
                    }
                });
            }
        };

        updateParamValue(temperatureSlider);
        updateParamValue(maxTokensSlider);
        updateParamValue(topPSlider);
    }

    showTestSection(testType) {
        console.log("Showing test section:", testType);
        
        const sections = document.querySelectorAll('.test-section');
        console.log("Found sections:", sections.length);
        
        sections.forEach(section => {
            console.log("Section ID:", section.id, "Current display:", section.style.display);
        });
        
        // Map test types to section IDs
        const sectionMap = {
            'batch': 'batch-test',
            'streaming': 'streaming-test',
            'simple-vs-complex': 'simple-vs-complex-test'
        };

        // Hide all sections first
        const testSections = document.querySelectorAll('.test-section');
        testSections.forEach(section => {
            if (section) {
                section.style.display = 'none';
            }
        });
        
        // Show the selected section
        const sectionId = sectionMap[testType];
        if (sectionId) {
            const selectedSection = document.getElementById(sectionId);
            if (selectedSection) {
                selectedSection.style.display = 'block';
            } else {
                console.error(`Could not find section with id: ${sectionId}`);
            }
        }
        
        // Clear results
        const resultsContent = document.getElementById('results-content');
        const streamingOutput = document.getElementById('streaming-output');
        
        if (resultsContent) resultsContent.innerHTML = '';
        if (streamingOutput) streamingOutput.innerHTML = '';
    }

    async runBatchTest() {
        const prompt = document.getElementById('batch-prompt').value;
        const batchSize = parseInt(document.getElementById('batch-size').value);
        const model = document.getElementById('batch-model').value;
        const loadingIndicator = document.getElementById('batch-loading');
        const progressText = document.getElementById('batch-progress');
        const resultsDiv = document.getElementById('results-content');
        const runBatchButton = document.getElementById('run-batch-test');

        if (!prompt) {
            alert('Please enter a prompt');
            return;
        }

        if (isNaN(batchSize) || batchSize < 1 || batchSize > 100) {
            alert('Please enter a valid batch size (1-100)');
            return;
        }

        try {
            // Show loading indicator
            if (loadingIndicator) loadingIndicator.style.display = 'block';
            if (resultsDiv) resultsDiv.innerHTML = '';
            if (runBatchButton) runBatchButton.disabled = true;

            // Update progress text
            if (progressText) progressText.textContent = 'Starting batch processing...';

            const response = await fetch('/test-performance', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    test_type: 'batch',
                    prompt: prompt,
                    batch_size: batchSize,
                    model: model
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            // Read the response as JSON directly
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }

            // Update progress text with completion
            if (progressText) {
                progressText.textContent = 'Batch processing completed!';
            }

            this.displayBatchResults(data);

        } catch (error) {
            console.error('Error in batch test:', error);
            if (resultsDiv) {
                resultsDiv.innerHTML = `
                    <div class="error">
                        Error running batch test: ${error.message}
                    </div>
                `;
            }
        } finally {
            // Hide loading indicator and re-enable button
            if (loadingIndicator) loadingIndicator.style.display = 'none';
            if (runBatchButton) runBatchButton.disabled = false;
        }
    }

    async runStreamingTest() {
        console.log("Starting streaming test");
        const resultsDiv = document.getElementById('streaming-output');
        const prompt = document.getElementById('streaming-prompt').value;
        const model = document.getElementById('streaming-model').value;

        if (!prompt) {
            alert('Please enter a prompt');
            return;
        }

        // Clear previous results and add containers for each model
        resultsDiv.innerHTML = `
            <div class="streaming-responses">
                <div class="model-stream" id="${model}-stream">
                    <h4>${model.charAt(0).toUpperCase() + model.slice(1)} Response</h4>
                    <div class="stream-content"></div>
                </div>
            </div>
        `;
        
        try {
            const response = await fetch('/test-performance', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    test_type: 'streaming',
                    prompt: prompt,
                    model: model
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const {value, done} = await reader.read();
                if (done) break;
                
                const text = decoder.decode(value);
                const lines = text.split('\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            const streamDiv = document.querySelector(`#${data.model}-stream .stream-content`);
                            
                            if (streamDiv) {
                                if (data.error) {
                                    streamDiv.innerHTML += `<p class="error">${data.content}</p>`;
                                } else if (!data.done) {
                                    streamDiv.innerHTML += data.content;
                                    streamDiv.scrollTop = streamDiv.scrollHeight;
                                }
                            }
                        } catch (e) {
                            console.error("Error parsing stream data:", e);
                        }
                    }
                }
            }
        } catch (error) {
            console.error("Error in streaming test:", error);
            resultsDiv.innerHTML += `<p class="error">Error: ${error.message}</p>`;
        }
    }

    async runLatencyTest() {
        const simpleQuery = document.getElementById('simple-query').value;
        const complexQuery = document.getElementById('complex-query').value;
        // Find the results div within the simple-vs-complex test section
        const testSection = document.getElementById('simple-vs-complex-test');
        const resultsDiv = testSection.querySelector('#results-content');
        const loadingIndicator = document.getElementById('latency-loading');
        const runLatencyButton = document.getElementById('run-latency-test');

        if (!simpleQuery || !complexQuery) {
            alert('Please enter both simple and complex queries');
            return;
        }

        try {
            // Show loading indicator if it exists
            if (loadingIndicator) {
                loadingIndicator.style.display = 'block';
            }
            if (runLatencyButton) {
                runLatencyButton.disabled = true;
            }
            if (resultsDiv) {
                resultsDiv.innerHTML = '<p>Running latency tests...</p>';
            }

            const response = await fetch('/test-performance', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    test_type: 'latency',
                    simple_prompt: simpleQuery,
                    complex_prompt: complexQuery
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            if (data.error) {
                throw new Error(data.error);
            }

            this.displayLatencyResults(data);
        } catch (error) {
            console.error('Error in latency test:', error);
            if (resultsDiv) {
                resultsDiv.innerHTML = `
                    <div class="error">Error running latency test: ${error.message}</div>
                `;
            }
        } finally {
            // Hide loading indicator and re-enable button
            if (loadingIndicator) {
                loadingIndicator.style.display = 'none';
            }
            if (runLatencyButton) {
                runLatencyButton.disabled = false;
            }
        }
    }

    async runParameterTest() {
        const prompt = document.getElementById('parameter-prompt').value;
        const model = document.getElementById('parameter-model').value;
        const temperature = parseFloat(document.getElementById('temperature').value);
        const maxTokens = parseInt(document.getElementById('max-tokens').value);
        const topP = parseFloat(document.getElementById('top-p').value);
        const resultsDiv = document.getElementById('results-content');

        if (!prompt) {
            alert('Please enter a prompt');
            return;
        }

        try {
            const response = await fetch('/parameter-test', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    prompt: prompt,
                    model: model,
                    params: {
                        temperature: temperature,
                        max_tokens: maxTokens,
                        top_p: topP
                    }
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.displayParameterResults(data);
        } catch (error) {
            console.error('Error in parameter test:', error);
            resultsDiv.innerHTML = `
                <div class="error">Error running parameter test: ${error.message}</div>
            `;
        }
    }

    displayBatchResults(data) {
        // Find the results div within the batch test section
        const testSection = document.getElementById('batch-test');
        const resultsDiv = testSection.querySelector('#results-content');
        
        if (!resultsDiv) {
            console.error('Could not find results div in batch test section');
            return;
        }

        let html = '<h4>Batch Processing Results</h4>';
        
        try {
            if (data.error) {
                throw new Error(data.error);
            }

            html += `
                <div class="batch-results">
                    <div class="metric-card">
                        <h5>Individual Requests</h5>
                        <p>Total Time: ${data.individual.total_time.toFixed(2)}s</p>
                        <p>Average Time: ${(data.individual.total_time / data.individual.requests_completed).toFixed(2)}s</p>
                        <p>Total Cost: $${data.individual.cost.toFixed(4)}</p>
                        <p>Average Cost: $${(data.individual.cost / data.individual.requests_completed).toFixed(4)}</p>
                        <p>Requests Completed: ${data.individual.requests_completed}</p>
                    </div>
                    
                    <div class="metric-card">
                        <h5>Batch Processing</h5>
                        <p>Total Time: ${data.batch.total_time.toFixed(2)}s</p>
                        <p>Average Time: ${(data.batch.total_time / data.batch.requests_completed).toFixed(2)}s</p>
                        <p>Total Cost: $${data.batch.cost.toFixed(4)}</p>
                        <p>Average Cost: $${(data.batch.cost / data.batch.requests_completed).toFixed(4)}</p>
                        <p>Requests Completed: ${data.batch.requests_completed}</p>
                    </div>
                    
                    <div class="metric-card">
                        <h5>Cache Performance</h5>
                        <p>Cache Hits: ${data.batch.cache_stats.hits}</p>
                        <p>Cache Misses: ${data.batch.cache_stats.misses}</p>
                        <p>Hit Rate: ${((data.batch.cache_stats.hits / (data.batch.cache_stats.hits + data.batch.cache_stats.misses)) * 100).toFixed(1)}%</p>
                    </div>
                    
                    <div class="metric-card">
                        <h5>Performance Improvement</h5>
                        <p>Time Saved: ${data.improvements.time_saved_percent.toFixed(1)}%</p>
                        <p>Cost Saved: ${data.improvements.cost_saved_percent.toFixed(1)}%</p>
                    </div>
                </div>
            `;
        } catch (error) {
            console.error('Error displaying batch results:', error);
            html = `
                <div class="error">
                    Error displaying results: ${error.message}<br>
                    ${data.error ? '' : `Raw data: <pre>${JSON.stringify(data, null, 2)}</pre>`}
                </div>
            `;
        }
        
        resultsDiv.innerHTML = html;
    }

    displayLatencyResults(data) {
        // Find the results div within the simple-vs-complex test section
        const testSection = document.getElementById('simple-vs-complex-test');
        const resultsDiv = testSection.querySelector('#results-content');
        let html = '<div class="latency-test-results">';

        for (const [model, results] of Object.entries(data)) {
            html += `
                <div class="model-result">
                    <h4>${model}</h4>
                    <div class="latency-results">
                        <div class="simple-query">
                            <h5>Simple Query</h5>
                            <p>Time: ${results.simple_latency.toFixed(2)}s</p>
                            <p>Cost: $${results.simple_cost.toFixed(4)}</p>
                            <p>Response: ${results.simple_response}</p>
                        </div>
                        <div class="complex-query">
                            <h5>Complex Query</h5>
                            <p>Time: ${results.complex_latency.toFixed(2)}s</p>
                            <p>Cost: $${results.complex_cost.toFixed(4)}</p>
                            <p>Response: ${results.complex_response}</p>
                        </div>
                    </div>
                </div>
            `;
        }

        html += '</div>';
        if (resultsDiv) {
            resultsDiv.innerHTML = html;
        } else {
            console.error('Could not find results div in simple-vs-complex test section');
        }
    }

    displayParameterResults(data) {
        const resultsDiv = document.getElementById('results-content');
        let html = `
            <div class="parameter-results">
                <h4>Parameter Test Results</h4>
                <div class="response-card">
                    <h5>Response</h5>
                    <p>${data.response}</p>
                    <div class="metrics">
                        <p>Time: ${data.metrics.time_taken.toFixed(2)}s</p>
                        <p>Length: ${data.metrics.length} tokens</p>
                        <p>Cost: $${data.metrics.cost.toFixed(4)}</p>
                    </div>
                </div>
            </div>
        `;
        resultsDiv.innerHTML = html;
    }

    // Add other test methods as needed...
}

// Initialize performance testing
document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM loaded, initializing PerformanceTesting");  // Debug log
    window.performanceTesting = new PerformanceTesting();
}); 