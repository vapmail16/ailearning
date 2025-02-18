class FunctionTesting {
    constructor() {
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Test type selection
        const testType = document.getElementById('test-type');
        const functionSection = document.getElementById('function-options');
        const parameterOptions = document.getElementById('parameter-options');
        const tierOptions = document.getElementById('tier-options');
        
        if (testType) {
            testType.addEventListener('change', (e) => {
                if (e.target.value === 'function') {
                    functionSection.style.display = 'block';
                    parameterOptions.style.display = 'none';
                    tierOptions.style.display = 'none';
                } else if (e.target.value === 'parameter') {
                    functionSection.style.display = 'none';
                    parameterOptions.style.display = 'block';
                    tierOptions.style.display = 'none';
                } else if (e.target.value === 'tier') {
                    functionSection.style.display = 'none';
                    parameterOptions.style.display = 'none';
                    tierOptions.style.display = 'block';
                }
            });
        }

        // Run function test button
        const runTestBtn = document.getElementById('run-function-test');
        if (runTestBtn) {
            runTestBtn.addEventListener('click', () => this.runFunctionTest());
        }

        // Initialize the correct section based on current selection
        if (testType && testType.value === 'function') {
            functionSection.style.display = 'block';
            parameterOptions.style.display = 'none';
            tierOptions.style.display = 'none';
        }

        // Function selection handler
        const functionSelect = document.getElementById('function-select');
        if (functionSelect) {
            functionSelect.addEventListener('change', () => {
                // Hide all parameter sections
                document.querySelectorAll('.function-params').forEach(el => {
                    el.style.display = 'none';
                });
                
                // Show the selected function's parameters
                const selectedFunction = functionSelect.value;
                const paramSection = document.getElementById(`${selectedFunction}-params`);
                if (paramSection) {
                    paramSection.style.display = 'block';
                } else {
                    console.error(`Parameter section not found for ${selectedFunction}`);
                }
            });

            // Initialize with the currently selected function
            const initialFunction = functionSelect.value;
            const initialParamSection = document.getElementById(`${initialFunction}-params`);
            if (initialParamSection) {
                initialParamSection.style.display = 'block';
            }
        }
    }

    async runFunctionTest() {
        const functionType = document.getElementById('function-select').value;
        const selectedModels = Array.from(document.querySelectorAll('.model-checkboxes input:checked'))
            .map(checkbox => checkbox.value);

        if (selectedModels.length === 0) {
            alert('Please select at least one model');
            return;
        }

        const params = this.getFunctionParameters(functionType);
        if (!params) return;

        const loadingSection = document.querySelector('.parameter-loading');
        const loadingText = loadingSection.querySelector('p');
        loadingText.textContent = 'Testing function calls...';
        loadingSection.style.display = 'block';

        try {
            const response = await fetch('/test-function', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    function: functionType,
                    parameters: params,
                    models: selectedModels
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.displayFunctionResults(data);

        } catch (error) {
            console.error('Error:', error);
            alert('Error testing function: ' + error.message);
        } finally {
            loadingSection.style.display = 'none';
            loadingText.textContent = 'Running parameter tests...';
        }
    }

    getFunctionParameters(functionType) {
        switch (functionType) {
            case 'get_weather':
                const location = document.getElementById('location').value;
                if (!location) {
                    alert('Please enter a location');
                    return null;
                }
                return {
                    location: location,
                    units: document.getElementById('units').value
                };
            
            case 'calculator':
                const numbersStr = document.getElementById('numbers').value;
                if (!numbersStr) {
                    alert('Please enter numbers');
                    return null;
                }
                return {
                    operation: document.getElementById('operation').value,
                    numbers: numbersStr.split(',').map(n => parseFloat(n.trim()))
                };
            
            case 'wikipedia_search':
                const query = document.getElementById('query').value;
                if (!query) {
                    alert('Please enter a search query');
                    return null;
                }
                return {
                    query: query,
                    limit: parseInt(document.getElementById('limit').value)
                };
            
            case 'currency_convert':
                const amount = document.getElementById('amount').value;
                const fromCurrency = document.getElementById('from-currency').value;
                const toCurrency = document.getElementById('to-currency').value;
                
                if (!amount || !fromCurrency || !toCurrency) {
                    alert('Please fill in all currency conversion fields');
                    return null;
                }
                return {
                    amount: parseFloat(amount),
                    from_currency: fromCurrency,
                    to_currency: toCurrency
                };
            
            case 'amazon_order':
                const product = document.getElementById('product').value;
                const quantity = document.getElementById('quantity').value;
                const maxPrice = document.getElementById('max-price').value;
                const priorityShipping = document.getElementById('priority-shipping').checked;
                
                if (!product) {
                    alert('Please enter a product description');
                    return null;
                }
                if (!maxPrice || maxPrice <= 0) {
                    alert('Please enter a valid maximum price');
                    return null;
                }
                
                return {
                    product: product,
                    quantity: parseInt(quantity),
                    max_price: parseFloat(maxPrice),
                    priority_shipping: priorityShipping
                };
            
            default:
                return null;
        }
    }

    displayFunctionResults(data) {
        const tbody = document.getElementById('function-results-body');
        if (!tbody) {
            console.error('Results table body not found');
            return;
        }

        let rows = '';
        Object.entries(data.results).forEach(([model, result]) => {
            // Format the response text nicely
            let responseText = '';
            if (result.success && data.actual_data) {
                if (data.actual_data.result) {
                    // For Amazon orders and other structured responses
                    responseText = JSON.stringify(data.actual_data.result, null, 2);
                } else {
                    // For simple responses
                    responseText = JSON.stringify(data.actual_data, null, 2);
                }
            } else {
                // For error responses, show the full error message
                responseText = typeof result.response === 'string' ? 
                    result.response : 
                    JSON.stringify(result.response, null, 2);
            }

            rows += `
                <tr>
                    <td>${model}</td>
                    <td>${result.function_called}</td>
                    <td><pre style="max-width: 200px; overflow-x: auto;">${JSON.stringify(result.parameters, null, 2)}</pre></td>
                    <td><pre style="max-width: 300px; overflow-x: auto;">${this.truncateText(responseText, 500)}</pre></td>
                    <td>${result.metrics.time_taken.toFixed(2)}s</td>
                    <td>
                        <span class="success-indicator ${result.success ? 'success' : 'failure'}"></span>
                        ${result.success ? 'Success' : 'Failed'}
                    </td>
                    <td>$${result.metrics.cost.toFixed(4)}</td>
                </tr>
            `;
        });

        tbody.innerHTML = rows;
    }

    truncateText(text, maxLength) {
        if (!text) return '';
        return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
    }
}

// Initialize after DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.functionTesting = new FunctionTesting();
}); 