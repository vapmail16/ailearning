class PerformanceMonitoring {
    constructor() {
        this.charts = {};
        this.initializeCharts();
        this.startPerformanceMonitoring();
    }

    initializeCharts() {
        // Response Time Chart
        const rtCtx = document.getElementById('response-time-chart').getContext('2d');
        this.charts.responseTime = new Chart(rtCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Response Time (s)',
                    data: [],
                    borderColor: '#4CAF50',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Response Time Trend'
                    }
                }
            }
        });

        // Cost Distribution Chart
        const cdCtx = document.getElementById('cost-distribution-chart').getContext('2d');
        this.charts.costDistribution = new Chart(cdCtx, {
            type: 'doughnut',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    backgroundColor: [
                        '#4CAF50',
                        '#2196F3',
                        '#FFC107',
                        '#9C27B0'
                    ]
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Cost Distribution by Model'
                    }
                }
            }
        });
    }

    async startPerformanceMonitoring() {
        try {
            const response = await fetch('/performance');
            const data = await response.json();
            this.updateDashboard(data);
        } catch (error) {
            console.error('Error fetching performance metrics:', error);
        }
    }

    updateDashboard(data) {
        // Update summary cards
        document.getElementById('avg-response-time').textContent = 
            `${data.average_response_time.toFixed(2)}s`;
        document.getElementById('total-cost').textContent = 
            `$${data.total_cost.toFixed(4)}`;
        document.getElementById('cache-efficiency').textContent = 
            `${(data.cache_efficiency * 100).toFixed(1)}%`;

        // Update model performance table
        const tbody = document.getElementById('model-performance-body');
        tbody.innerHTML = '';
        
        Object.entries(data.model_performance).forEach(([model, metrics]) => {
            const row = tbody.insertRow();
            row.innerHTML = `
                <td>${model}</td>
                <td>${metrics.avg_response_time.toFixed(2)}s</td>
                <td>$${metrics.total_cost.toFixed(4)}</td>
                <td>${metrics.calls}</td>
                <td>${((metrics.success_rate || 0) * 100).toFixed(1)}%</td>
            `;
        });

        // Update charts
        this.updateCharts(data);
    }

    updateCharts(data) {
        // Update Response Time Chart
        const timeData = data.response_times.slice(-20); // Last 20 responses
        this.charts.responseTime.data.labels = timeData.map(d => 
            new Date(d.timestamp).toLocaleTimeString());
        this.charts.responseTime.data.datasets[0].data = 
            timeData.map(d => d.time);
        this.charts.responseTime.update();

        // Update Cost Distribution Chart
        const costByModel = Object.entries(data.model_performance)
            .map(([model, metrics]) => ({
                model,
                cost: metrics.total_cost
            }));
        
        this.charts.costDistribution.data.labels = 
            costByModel.map(d => d.model);
        this.charts.costDistribution.data.datasets[0].data = 
            costByModel.map(d => d.cost);
        this.charts.costDistribution.update();
    }
}

// Initialize performance monitoring when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.performanceMonitoring = new PerformanceMonitoring();
}); 