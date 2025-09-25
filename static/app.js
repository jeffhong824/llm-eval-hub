// LLM Evaluation Hub Frontend JavaScript

const API_BASE_URL = 'http://localhost:8000';

// Global state
let currentSection = 'dashboard';
let availableMetrics = [];
let availableJudgeModels = {};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    loadDashboardData();
    loadAvailableModels();
    loadPersonas();
    
    // Load evaluation metrics and judge models
    loadEvaluationMetrics();
    loadJudgeModels();
});

function initializeApp() {
    console.log('Initializing LLM Evaluation Hub...');
    
    // Load available metrics and judge models
    loadAvailableMetrics();
    loadAvailableJudgeModels();
    
    // Setup navigation
    setupNavigation();
    
    // Handle initial hash if present
    handleInitialHash();
}

function setupEventListeners() {
    console.log('Setting up event listeners...');
    
    // Scenario to docs form
    const scenarioForm = document.getElementById('scenario-to-docs-form');
    if (scenarioForm) {
        scenarioForm.addEventListener('submit', handleScenarioToDocs);
        console.log('Scenario form listener added');
    } else {
        console.error('Scenario form not found');
    }
    
    // RAG testset form
    const ragForm = document.getElementById('rag-testset-form');
    if (ragForm) {
        ragForm.addEventListener('submit', handleRAGTestset);
        console.log('RAG form listener added');
        
        // Setup RAG model selection
        const ragProviderSelect = document.getElementById('rag-model-provider');
        const ragModelSelect = document.getElementById('rag-model-name');
        if (ragProviderSelect && ragModelSelect) {
            setupModelSelection(ragProviderSelect, ragModelSelect);
        }
    } else {
        console.error('RAG form not found');
    }
    
    // Agent testset form
    const agentForm = document.getElementById('agent-testset-form');
    if (agentForm) {
        agentForm.addEventListener('submit', handleAgentTestset);
        console.log('Agent form listener added');
        
        // Setup Agent model selection
        const agentProviderSelect = document.getElementById('agent-model-provider');
        const agentModelSelect = document.getElementById('agent-model-name');
        if (agentProviderSelect && agentModelSelect) {
            setupModelSelection(agentProviderSelect, agentModelSelect);
        }
    } else {
        console.error('Agent form not found');
    }
    
    // RAG Evaluation form
    const ragEvalForm = document.getElementById('rag-evaluation-form');
    if (ragEvalForm) {
        ragEvalForm.addEventListener('submit', handleRAGEvaluation);
        console.log('RAG Evaluation form listener added');
    } else {
        console.error('RAG Evaluation form not found');
    }
    
    // Agent Evaluation form
    const agentEvalForm = document.getElementById('agent-evaluation-form');
    if (agentEvalForm) {
        agentEvalForm.addEventListener('submit', handleAgentEvaluation);
        console.log('Agent Evaluation form listener added');
    } else {
        console.error('Agent Evaluation form not found');
    }
    
    // API Test form
    const apiTestForm = document.getElementById('api-test-form');
    if (apiTestForm) {
        apiTestForm.addEventListener('submit', handleAPITest);
        console.log('API Test form listener added');
    } else {
        console.error('API Test form not found');
    }
    
    // Judge model type change
    const judgeModelType = document.getElementById('judge-model-type');
    if (judgeModelType) {
        judgeModelType.addEventListener('change', updateJudgeModelOptions);
        console.log('Judge model type listener added');
    } else {
        console.error('Judge model type not found');
    }
}

function setupNavigation() {
    console.log('Setting up navigation...');
    const navLinks = document.querySelectorAll('.nav-link');
    console.log(`Found ${navLinks.length} navigation links`);
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const section = this.getAttribute('data-section');
            console.log(`Navigating to section: ${section}`);
            showSection(section);
            
            // Update URL hash
            window.location.hash = section;
            
            // Update active state
            navLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');
        });
    });
    
    // Listen for hash changes
    window.addEventListener('hashchange', function() {
        const hash = window.location.hash.substring(1); // Remove the #
        if (hash) {
            console.log(`Hash changed to: ${hash}`);
            showSection(hash);
            updateActiveNavLink(hash);
        }
    });
}

function handleInitialHash() {
    const hash = window.location.hash.substring(1); // Remove the #
    if (hash) {
        console.log(`Initial hash detected: ${hash}`);
        showSection(hash);
        updateActiveNavLink(hash);
    }
}

function updateActiveNavLink(sectionName) {
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('data-section') === sectionName) {
            link.classList.add('active');
        }
    });
}

function showSection(sectionName) {
    console.log(`Showing section: ${sectionName}`);
    
    // Hide all sections
    const sections = document.querySelectorAll('.content-section');
    console.log(`Found ${sections.length} content sections`);
    sections.forEach(section => {
        section.style.display = 'none';
        console.log(`Hiding section: ${section.id}`);
    });
    
    // Show selected section
    const targetSection = document.getElementById(`${sectionName}-section`);
    if (targetSection) {
        targetSection.style.display = 'block';
        currentSection = sectionName;
        console.log(`Showing section: ${targetSection.id}`);
        
        // Load section-specific data
        if (sectionName === 'results') {
            loadResults();
        }
    } else {
        console.error(`Section not found: ${sectionName}-section`);
    }
}

async function loadDashboardData() {
    try {
        // Load system status
        const healthResponse = await fetch(`${API_BASE_URL}/health`);
        const healthData = await healthResponse.json();
        
        updateSystemStatus(healthData);
        
        // Load recent activity (placeholder)
        updateRecentActivity([]);
        
    } catch (error) {
        console.error('Error loading dashboard data:', error);
        showError('Failed to load dashboard data');
    }
}

function updateSystemStatus(healthData) {
    const statusElements = {
        'api-status': healthData.status === 'healthy' ? 'Healthy' : 'Unhealthy',
        'db-status': healthData.dependencies?.database === 'healthy' ? 'Connected' : 'Disconnected',
        'langsmith-status': healthData.dependencies?.langsmith === 'healthy' ? 'Connected' : 'Disconnected'
    };
    
    Object.entries(statusElements).forEach(([id, status]) => {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = status;
            element.className = status.includes('Healthy') || status.includes('Connected') ? 'status-success' : 'status-error';
        }
    });
}

function updateRecentActivity(activities) {
    const activityContainer = document.getElementById('recent-activity');
    if (activities.length === 0) {
        activityContainer.innerHTML = '<p class="text-muted">No recent activity</p>';
    } else {
        // Implementation for displaying activities
    }
}

async function loadAvailableMetrics() {
    try {
        console.log('Loading available metrics...');
        const response = await fetch(`${API_BASE_URL}/api/v1/evaluation/metrics/available`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        availableMetrics = data.all_metrics || [];
        console.log('Available metrics loaded:', availableMetrics);
        
        // Update metrics count
        const metricsElement = document.getElementById('available-metrics');
        if (metricsElement) {
            metricsElement.textContent = availableMetrics.length;
        }
        
    } catch (error) {
        console.error('Error loading available metrics:', error);
        // Set default metrics if API fails
        availableMetrics = ['answer_relevancy', 'answer_correctness', 'faithfulness'];
    }
}

async function loadAvailableJudgeModels() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/v1/evaluation/judge-models/available`);
        const data = await response.json();
        availableJudgeModels = data.available_models || {};
        
        // Update judge models count
        const totalModels = Object.values(availableJudgeModels).flat().length;
        document.getElementById('available-judges').textContent = totalModels;
        
    } catch (error) {
        console.error('Error loading available judge models:', error);
    }
}

function updateJudgeModelOptions() {
    const judgeType = document.getElementById('judge-model-type').value;
    const judgeModelInput = document.getElementById('judge-model');
    
    // Set default model based on type
    const defaultModels = {
        'openai': 'gpt-4-turbo-preview',
        'gemini': 'gemini-pro',
        'ollama': 'llama2',
        'huggingface': 'microsoft/DialoGPT-medium'
    };
    
    judgeModelInput.value = defaultModels[judgeType] || '';
}

// Scenario to Docs Handler
async function handleScenarioToDocs(e) {
    e.preventDefault();
    console.log('Scenario to docs form submitted');
    
    const prompt = document.getElementById('scenario-prompt').value;
    const outputFolder = document.getElementById('output-folder').value;
    const numDocs = parseInt(document.getElementById('num-docs').value);
    
    console.log('Form data:', { prompt, outputFolder, numDocs });
    
    if (!prompt || !outputFolder) {
        showError('Please fill in all required fields');
        return;
    }
    
    showLoading('scenario-results');
    
    try {
        console.log('Sending request to API...');
        const response = await fetch(`${API_BASE_URL}/api/v1/testset/scenario-to-docs`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt: prompt,
                output_folder: outputFolder,
                num_docs: numDocs
            })
        });
        
        console.log('Response status:', response.status);
        const result = await response.json();
        console.log('Response data:', result);
        
        if (response.ok) {
            showSuccess('scenario-results', 'Documents generated successfully!', result);
            
            // Add download functionality if documents are available
            if (result.documents && result.documents.length > 0) {
                addDownloadButtons('scenario-results', result.documents);
            }
        } else {
            showError('scenario-results', result.detail || 'Failed to generate documents');
        }
        
    } catch (error) {
        console.error('Error generating documents:', error);
        showError('scenario-results', `Failed to generate documents: ${error.message}`);
    }
}

// RAG Testset Handler
async function handleRAGTestset(e) {
    e.preventDefault();
    
    const documentsFolder = document.getElementById('rag-folder-path').value;
    const outputFolder = document.getElementById('rag-output-folder').value;
    const modelProvider = document.getElementById('rag-model-provider').value;
    const modelName = document.getElementById('rag-model-name').value;
    const language = document.getElementById('rag-language').value;
    const chunkSize = parseInt(document.getElementById('chunk-size').value);
    const chunkOverlap = parseInt(document.getElementById('chunk-overlap').value);
    const qaPerChunk = parseInt(document.getElementById('rag-qa-per-chunk').value);
    
    if (!documentsFolder) {
        showError('Please provide a documents folder path');
        return;
    }
    
    if (!outputFolder) {
        showError('Please provide an output folder path');
        return;
    }
    
    if (!modelName) {
        showError('Please select a model');
        return;
    }
    
    showLoading('rag-results');
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/v1/testset/rag-testset-from-documents`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                documents_folder: documentsFolder,
                output_folder: outputFolder,
                model_provider: modelProvider,
                model_name: modelName,
                language: language,
                chunk_size: chunkSize,
                chunk_overlap: chunkOverlap,
                qa_per_chunk: qaPerChunk
            })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showSuccess('rag-results', 'RAG testset generated successfully!', result);
        } else {
            showError('rag-results', result.detail || 'Failed to generate RAG testset');
        }
        
    } catch (error) {
        console.error('Error generating RAG testset:', error);
        showError('rag-results', 'Failed to generate RAG testset');
    }
}

// Agent Testset Handler
async function handleAgentTestset(e) {
    e.preventDefault();
    
    const documentsFolder = document.getElementById('agent-documents-folder').value;
    const outputFolder = document.getElementById('agent-output-folder').value;
    const modelProvider = document.getElementById('agent-model-provider').value;
    const modelName = document.getElementById('agent-model-name').value;
    const language = document.getElementById('agent-language').value;
    const chunkSize = parseInt(document.getElementById('agent-chunk-size').value);
    const chunkOverlap = parseInt(document.getElementById('agent-chunk-overlap').value);
    const tasksPerChunk = parseInt(document.getElementById('agent-tasks-per-chunk').value);
    
    if (!documentsFolder) {
        showError('Please provide a documents folder path');
        return;
    }
    
    if (!outputFolder) {
        showError('Please provide an output folder path');
        return;
    }
    
    if (!modelName) {
        showError('Please select a model');
        return;
    }
    
    showLoading('agent-results');
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/v1/testset/agent-testset-from-documents`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                documents_folder: documentsFolder,
                output_folder: outputFolder,
                model_provider: modelProvider,
                model_name: modelName,
                language: language,
                chunk_size: chunkSize,
                chunk_overlap: chunkOverlap,
                tasks_per_chunk: tasksPerChunk
            })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showSuccess('agent-results', 'Agent testset generated successfully!', result);
        } else {
            showError('agent-results', result.detail || 'Failed to generate agent testset');
        }
        
    } catch (error) {
        console.error('Error generating agent testset:', error);
        showError('agent-results', 'Failed to generate agent testset');
    }
}

// Evaluation Handler
async function handleEvaluation(e) {
    e.preventDefault();
    
    const testsetIds = document.getElementById('testset-ids').value.split(',').map(id => id.trim());
    const llmEndpoint = document.getElementById('llm-endpoint').value;
    const systemType = document.getElementById('system-type').value;
    const judgeModelType = document.getElementById('judge-model-type').value;
    const judgeModel = document.getElementById('judge-model').value;
    
    if (!testsetIds.length || !llmEndpoint) {
        showError('Please fill in all required fields');
        return;
    }
    
    // Get selected metrics
    const selectedMetrics = [];
    const metricCheckboxes = document.querySelectorAll('input[type="checkbox"]:checked');
    metricCheckboxes.forEach(checkbox => {
        selectedMetrics.push(checkbox.id.replace('metric-', ''));
    });
    
    if (selectedMetrics.length === 0) {
        showError('Please select at least one metric');
        return;
    }
    
    showLoading('evaluation-results');
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/v1/evaluation/evaluate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                testset_ids: testsetIds,
                llm_endpoint: llmEndpoint,
                metrics: selectedMetrics,
                system_type: systemType,
                judge_model_type: judgeModelType,
                judge_model: judgeModel
            })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showSuccess('evaluation-results', 'Evaluation completed successfully!', result);
        } else {
            showError('evaluation-results', result.detail || 'Failed to run evaluation');
        }
        
    } catch (error) {
        console.error('Error running evaluation:', error);
        showError('evaluation-results', 'Failed to run evaluation');
    }
}

// Utility Functions
function showLoading(containerId) {
    const container = document.getElementById(containerId);
    container.innerHTML = `
        <div class="text-center">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-2">Processing...</p>
        </div>
    `;
}

function showSuccess(containerId, message, data) {
    const container = document.getElementById(containerId);
    container.innerHTML = `
        <div class="result-card">
            <div class="d-flex align-items-center mb-3">
                <i class="fas fa-check-circle text-success me-2"></i>
                <h5 class="mb-0">${message}</h5>
            </div>
            <div class="row">
                <div class="col-md-6">
                    <p><strong>Status:</strong> <span class="status-success">Success</span></p>
                    <p><strong>Timestamp:</strong> ${new Date().toLocaleString()}</p>
                </div>
                <div class="col-md-6">
                    <p><strong>Testset ID:</strong> ${data.testset_id || 'N/A'}</p>
                    <p><strong>Generation Time:</strong> ${data.generation_time || data.evaluation_time || 'N/A'}s</p>
                </div>
            </div>
            ${data.knowledge_graph_path ? `<p><strong>Output Path:</strong> ${data.knowledge_graph_path}</p>` : ''}
            ${data.metrics ? `
                <div class="mt-3">
                    <h6>Evaluation Metrics:</h6>
                    <div>
                        ${Object.entries(data.metrics).map(([metric, score]) => 
                            `<span class="metric-badge">${metric}: ${score.toFixed(3)}</span>`
                        ).join('')}
                    </div>
                </div>
            ` : ''}
        </div>
    `;
}

function showError(containerId, message) {
    const container = document.getElementById(containerId);
    container.innerHTML = `
        <div class="result-card">
            <div class="d-flex align-items-center">
                <i class="fas fa-exclamation-circle text-danger me-2"></i>
                <h5 class="mb-0 text-danger">Error</h5>
            </div>
            <p class="mt-2">${message}</p>
        </div>
    `;
}

function showError(message) {
    // Global error display
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-danger alert-dismissible fade show position-fixed';
    alertDiv.style.top = '20px';
    alertDiv.style.right = '20px';
    alertDiv.style.zIndex = '9999';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    document.body.appendChild(alertDiv);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.parentNode.removeChild(alertDiv);
        }
    }, 5000);
}

function addDownloadButtons(containerId, documents) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    // Create download section
    const downloadSection = document.createElement('div');
    downloadSection.className = 'mt-3';
    downloadSection.innerHTML = `
        <h6>下載文檔：</h6>
        <div class="btn-group-vertical w-100" role="group">
            ${documents.map((doc, index) => `
                <button type="button" class="btn btn-outline-primary mb-2" onclick="downloadDocument('${doc.filename}', \`${doc.content.replace(/`/g, '\\`')}\`)">
                    <i class="fas fa-download me-2"></i>${doc.filename}
                </button>
            `).join('')}
            <button type="button" class="btn btn-success mt-2" onclick="downloadAllDocuments(${JSON.stringify(documents)})">
                <i class="fas fa-download me-2"></i>下載所有文檔
            </button>
        </div>
    `;
    
    container.appendChild(downloadSection);
}

function downloadDocument(filename, content) {
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

function downloadAllDocuments(documents) {
    documents.forEach((doc, index) => {
        setTimeout(() => {
            downloadDocument(doc.filename, doc.content);
        }, index * 500); // Delay each download by 500ms
    });
}

async function loadResults() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/v1/evaluation/results`);
        const data = await response.json();
        
        const resultsContainer = document.getElementById('results-list');
        
        if (data.results && data.results.length > 0) {
            resultsContainer.innerHTML = data.results.map(result => `
                <div class="result-card">
                    <div class="d-flex justify-content-between align-items-center">
                        <h6>${result.evaluation_id || 'Unknown'}</h6>
                        <span class="badge bg-${result.status === 'success' ? 'success' : 'danger'}">${result.status}</span>
                    </div>
                    <p class="text-muted mb-0">${new Date(result.timestamp).toLocaleString()}</p>
                </div>
            `).join('');
        } else {
            resultsContainer.innerHTML = '<p class="text-muted">No results available</p>';
        }
        
    } catch (error) {
        console.error('Error loading results:', error);
        document.getElementById('results-list').innerHTML = '<p class="text-danger">Failed to load results</p>';
    }
}

// Load available models
async function loadAvailableModels() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/v1/testset/models/available`);
        const data = await response.json();
        
        console.log('Available models:', data);
        
        // Update model provider dropdown
        const providerSelect = document.getElementById('model-provider');
        const modelSelect = document.getElementById('model-name');
        
        if (providerSelect && modelSelect) {
            // Clear existing options
            modelSelect.innerHTML = '';
            
            // Add models for each provider
            Object.entries(data.summary).forEach(([provider, models]) => {
                console.log(`Processing provider: ${provider}, models:`, models);
                if (models && Array.isArray(models) && models.length > 0) {
                    models.forEach(model => {
                        const option = document.createElement('option');
                        option.value = model;
                        option.textContent = `${provider.toUpperCase()}: ${model}`;
                        option.dataset.provider = provider;
                        modelSelect.appendChild(option);
                        console.log(`Added model: ${provider} - ${model}`);
                    });
                } else {
                    console.log(`No models for provider: ${provider}`);
                }
            });
            
            // Update provider selection handler
            providerSelect.addEventListener('change', function() {
                const selectedProvider = this.value;
                const options = modelSelect.querySelectorAll('option');
                let hasAvailableModels = false;
                let firstAvailableModel = null;
                
                options.forEach(option => {
                    if (option.dataset.provider === selectedProvider) {
                        option.style.display = 'block';
                        if (!hasAvailableModels) {
                            firstAvailableModel = option.value;
                            hasAvailableModels = true;
                        }
                    } else {
                        option.style.display = 'none';
                    }
                });
                
                // Set the first available model as selected
                if (hasAvailableModels && firstAvailableModel) {
                    modelSelect.value = firstAvailableModel;
                } else {
                    // If no models available for selected provider, show placeholder
                    modelSelect.value = '';
                    // Add a placeholder option if it doesn't exist
                    if (!modelSelect.querySelector('option[value=""]')) {
                        const placeholderOption = document.createElement('option');
                        placeholderOption.value = '';
                        placeholderOption.textContent = 'No models available';
                        placeholderOption.style.display = 'block';
                        modelSelect.appendChild(placeholderOption);
                    }
                }
            });
            
            // Set initial selection
            providerSelect.dispatchEvent(new Event('change'));
        }
        
    } catch (error) {
        console.error('Error loading available models:', error);
    }
}

// Load personas
async function loadPersonas() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/v1/testset/personas?count=20`);
        const data = await response.json();
        
        console.log('Available personas:', data);
        
        // Update personas display
        const numDocsInput = document.getElementById('num-docs');
        if (numDocsInput) {
            numDocsInput.addEventListener('input', function() {
                const count = parseInt(this.value) || 1;
                const personas = data.personas.slice(0, count);
                
                // Show personas preview
                let personasPreview = document.getElementById('personas-preview');
                if (!personasPreview) {
                    personasPreview = document.createElement('div');
                    personasPreview.id = 'personas-preview';
                    personasPreview.className = 'mt-2';
                    numDocsInput.parentNode.appendChild(personasPreview);
                }
                
                if (count <= 5) {
                    personasPreview.innerHTML = `
                        <div class="form-text">
                            <strong>Personas:</strong> ${personas.join(', ')}
                        </div>
                    `;
                } else {
                    personasPreview.innerHTML = `
                        <div class="form-text">
                            <strong>Personas:</strong> ${personas.slice(0, 3).join(', ')} and ${count - 3} more...
                        </div>
                    `;
                }
            });
            
            // Trigger initial update
            numDocsInput.dispatchEvent(new Event('input'));
        }
        
    } catch (error) {
        console.error('Error loading personas:', error);
    }
}

// Setup model selection for RAG and Agent forms
function setupModelSelection(providerSelect, modelSelect) {
    fetch(`${API_BASE_URL}/api/v1/testset/models/available`)
        .then(response => response.json())
        .then(data => {
            console.log('Available models for form:', data);
            
            // Clear existing options
            modelSelect.innerHTML = '';
            
            // Add models for each provider
            Object.entries(data.summary).forEach(([provider, models]) => {
                console.log(`Processing provider: ${provider}, models:`, models);
                if (models && Array.isArray(models) && models.length > 0) {
                    models.forEach(model => {
                        const option = document.createElement('option');
                        option.value = model;
                        option.textContent = `${provider.toUpperCase()}: ${model}`;
                        option.dataset.provider = provider;
                        modelSelect.appendChild(option);
                        console.log(`Added model: ${provider} - ${model}`);
                    });
                } else {
                    console.log(`No models for provider: ${provider}`);
                }
            });
            
            // Update provider selection handler
            providerSelect.addEventListener('change', function() {
                const selectedProvider = this.value;
                const options = modelSelect.querySelectorAll('option');
                let hasAvailableModels = false;
                let firstAvailableModel = null;
                
                console.log(`Selected provider: ${selectedProvider}`);
                
                options.forEach(option => {
                    if (option.dataset.provider === selectedProvider) {
                        option.style.display = 'block';
                        console.log(`Showing option: ${option.value}`);
                        if (!hasAvailableModels) {
                            firstAvailableModel = option.value;
                            hasAvailableModels = true;
                        }
                    } else {
                        option.style.display = 'none';
                    }
                });
                
                console.log(`Has available models: ${hasAvailableModels}, First model: ${firstAvailableModel}`);
                
                // Set the first available model as selected
                if (hasAvailableModels && firstAvailableModel) {
                    modelSelect.value = firstAvailableModel;
                    console.log(`Set model to: ${firstAvailableModel}`);
                } else {
                    modelSelect.value = '';
                    console.log('No models available, clearing selection');
                }
            });
            
            // Set initial selection
            providerSelect.dispatchEvent(new Event('change'));
        })
        .catch(error => {
            console.error('Error loading available models:', error);
        });
}

// Evaluation functions
async function loadEvaluationMetrics() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/v1/evaluation/metrics/available`);
        const metrics = await response.json();
        
        // Load RAG metrics
        const ragContainer = document.getElementById('rag-metrics-container');
        if (ragContainer) {
            ragContainer.innerHTML = '';
            metrics.rag_metrics.forEach(metric => {
                const col = document.createElement('div');
                col.className = 'col-md-6';
                col.innerHTML = `
                    <div class="form-check">
                        <input class="form-check-input" type="checkbox" id="rag-metric-${metric}" checked>
                        <label class="form-check-label" for="rag-metric-${metric}">${metric.replace(/_/g, ' ').toUpperCase()}</label>
                    </div>
                `;
                ragContainer.appendChild(col);
            });
        }
        
        // Load Agent metrics
        const agentContainer = document.getElementById('agent-metrics-container');
        if (agentContainer) {
            agentContainer.innerHTML = '';
            metrics.agent_metrics.forEach(metric => {
                const col = document.createElement('div');
                col.className = 'col-md-6';
                col.innerHTML = `
                    <div class="form-check">
                        <input class="form-check-input" type="checkbox" id="agent-metric-${metric}" checked>
                        <label class="form-check-label" for="agent-metric-${metric}">${metric.replace(/_/g, ' ').toUpperCase()}</label>
                    </div>
                `;
                agentContainer.appendChild(col);
            });
        }
    } catch (error) {
        console.error('Error loading evaluation metrics:', error);
    }
}

async function loadJudgeModels() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/v1/evaluation/judge-models/available`);
        const models = await response.json();
        
        // Update RAG judge models
        updateJudgeModelSelect('rag-judge-model-type', 'rag-judge-model', models);
        
        // Update Agent judge models
        updateJudgeModelSelect('agent-judge-model-type', 'agent-judge-model', models);
        
    } catch (error) {
        console.error('Error loading judge models:', error);
    }
}

function updateJudgeModelSelect(providerSelectId, modelSelectId, models) {
    const providerSelect = document.getElementById(providerSelectId);
    const modelSelect = document.getElementById(modelSelectId);
    
    if (!providerSelect || !modelSelect) return;
    
    // Clear existing options
    modelSelect.innerHTML = '<option value="">Loading models...</option>';
    
    // Add models for each provider
    Object.entries(models).forEach(([provider, modelList]) => {
        if (modelList && modelList.length > 0) {
            modelList.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = `${provider.toUpperCase()}: ${model}`;
                option.dataset.provider = provider;
                modelSelect.appendChild(option);
            });
        }
    });
    
    // Update provider selection handler
    providerSelect.addEventListener('change', function() {
        const selectedProvider = this.value;
        const options = modelSelect.querySelectorAll('option');
        let hasAvailableModels = false;
        let firstAvailableModel = null;
        
        options.forEach(option => {
            if (option.dataset.provider === selectedProvider) {
                option.style.display = 'block';
                if (!hasAvailableModels) {
                    firstAvailableModel = option.value;
                    hasAvailableModels = true;
                }
            } else {
                option.style.display = 'none';
            }
        });
        
        if (hasAvailableModels && firstAvailableModel) {
            modelSelect.value = firstAvailableModel;
        } else {
            modelSelect.value = '';
        }
    });
    
    // Set initial selection
    providerSelect.dispatchEvent(new Event('change'));
}

async function handleRAGEvaluation(e) {
    e.preventDefault();
    
    const testsetPath = document.getElementById('rag-testset-path').value;
    const apiUrl = document.getElementById('rag-api-url').value;
    const apiHeaders = document.getElementById('rag-api-headers').value;
    const judgeModelType = document.getElementById('rag-judge-model-type').value;
    const judgeModel = document.getElementById('rag-judge-model').value;
    const outputPath = document.getElementById('rag-output-path').value;
    
    if (!testsetPath || !apiUrl || !judgeModel) {
        showError('Please fill in all required fields');
        return;
    }
    
    try {
        // Parse headers
        let headers = {};
        if (apiHeaders.trim()) {
            headers = JSON.parse(apiHeaders);
        }
        
        // Get selected metrics
        const selectedMetrics = [];
        document.querySelectorAll('#rag-metrics-container input[type="checkbox"]:checked').forEach(checkbox => {
            selectedMetrics.push(checkbox.id.replace('rag-metric-', ''));
        });
        
        const requestData = {
            testset_path: testsetPath,
            api_config: {
                url: apiUrl,
                headers: headers
            },
            judge_model_type: judgeModelType,
            judge_model: judgeModel,
            output_path: outputPath,
            selected_metrics: selectedMetrics
        };
        
        showLoading('evaluation-results');
        
        const response = await fetch(`${API_BASE_URL}/api/v1/evaluation/rag`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showSuccess('evaluation-results', 'RAG evaluation completed successfully!', result);
        } else {
            showError('evaluation-results', `Error: ${result.detail || 'Unknown error'}`);
        }
    } catch (error) {
        console.error('Error in RAG evaluation:', error);
        showError('evaluation-results', 'Failed to start RAG evaluation');
    }
}

async function handleAgentEvaluation(e) {
    e.preventDefault();
    
    const testsetPath = document.getElementById('agent-testset-path').value;
    const apiUrl = document.getElementById('agent-api-url').value;
    const apiHeaders = document.getElementById('agent-api-headers').value;
    const maxTurns = parseInt(document.getElementById('agent-max-turns').value);
    const judgeModelType = document.getElementById('agent-judge-model-type').value;
    const judgeModel = document.getElementById('agent-judge-model').value;
    const outputPath = document.getElementById('agent-output-path').value;
    
    if (!testsetPath || !apiUrl || !judgeModel) {
        showError('Please fill in all required fields');
        return;
    }
    
    try {
        // Parse headers
        let headers = {};
        if (apiHeaders.trim()) {
            headers = JSON.parse(apiHeaders);
        }
        
        // Get selected metrics
        const selectedMetrics = [];
        document.querySelectorAll('#agent-metrics-container input[type="checkbox"]:checked').forEach(checkbox => {
            selectedMetrics.push(checkbox.id.replace('agent-metric-', ''));
        });
        
        const requestData = {
            testset_path: testsetPath,
            api_config: {
                url: apiUrl,
                headers: headers
            },
            judge_model_type: judgeModelType,
            judge_model: judgeModel,
            output_path: outputPath,
            max_turns: maxTurns,
            selected_metrics: selectedMetrics
        };
        
        showLoading('evaluation-results');
        
        const response = await fetch(`${API_BASE_URL}/api/v1/evaluation/agent`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showSuccess('evaluation-results', 'Agent evaluation completed successfully!', result);
        } else {
            showError('evaluation-results', `Error: ${result.detail || 'Unknown error'}`);
        }
    } catch (error) {
        console.error('Error in Agent evaluation:', error);
        showError('evaluation-results', 'Failed to start Agent evaluation');
    }
}

async function handleAPITest(e) {
    e.preventDefault();
    
    const apiUrl = document.getElementById('test-api-url').value;
    const apiHeaders = document.getElementById('test-api-headers').value;
    
    if (!apiUrl) {
        showError('api-test-results', 'Please provide API URL');
        return;
    }
    
    try {
        // Parse headers
        let headers = {};
        if (apiHeaders.trim()) {
            headers = JSON.parse(apiHeaders);
        }
        
        const requestData = {
            url: apiUrl,
            headers: headers
        };
        
        showLoading('api-test-results');
        
        const response = await fetch(`${API_BASE_URL}/api/v1/evaluation/test-api-connection`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        const result = await response.json();
        
        if (result.success) {
            showSuccess('api-test-results', 'API connection successful!', result);
        } else {
            showError('api-test-results', `Connection failed: ${result.error || result.message}`);
        }
    } catch (error) {
        console.error('Error testing API connection:', error);
        showError('api-test-results', 'Failed to test API connection');
    }
}
