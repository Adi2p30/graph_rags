// Graph RAG Dashboard JavaScript

let currentGraph = null;
let graphData = { entities: [], relationships: [] };

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    loadGraphs();
});

// Load all graphs
async function loadGraphs() {
    try {
        const response = await fetch('/api/graphs');
        const data = await response.json();

        if (data.success) {
            displayGraphsList(data.graphs);
        } else {
            showNotification('Error loading graphs: ' + data.error, 'error');
        }
    } catch (error) {
        showNotification('Error: ' + error.message, 'error');
    }
}

// Display graphs in sidebar
function displayGraphsList(graphs) {
    const container = document.getElementById('graphsList');

    if (graphs.length === 0) {
        container.innerHTML = '<p class="text-muted">No graphs yet. Create one!</p>';
        return;
    }

    container.innerHTML = '';
    graphs.forEach(graph => {
        const item = document.createElement('div');
        item.className = 'graph-item';
        if (currentGraph === graph.name) {
            item.classList.add('active');
        }

        item.innerHTML = `
            <div class="graph-item-header">
                <span class="graph-item-name">${graph.name}</span>
                <span class="graph-item-delete" onclick="deleteGraph('${graph.name}', event)">🗑️</span>
            </div>
            <div style="font-size: 0.85rem; color: #718096; margin-top: 5px;">
                ${graph.statistics.entity_count} entities, ${graph.statistics.relationship_count} relationships
            </div>
        `;

        item.onclick = (e) => {
            if (!e.target.classList.contains('graph-item-delete')) {
                selectGraph(graph.name);
            }
        };

        container.appendChild(item);
    });
}

// Select a graph
async function selectGraph(graphName) {
    currentGraph = graphName;

    // Update UI
    document.querySelectorAll('.graph-item').forEach(item => {
        item.classList.remove('active');
    });
    event.target.closest('.graph-item').classList.add('active');

    // Load graph data
    await loadGraphData(graphName);
    updateStatistics();
    visualizeGraph();
    displayDataExplorer();
}

// Load graph data
async function loadGraphData(graphName) {
    try {
        const response = await fetch(`/api/graphs/${graphName}`);
        const data = await response.json();

        if (data.success) {
            graphData = {
                entities: data.graph.entities,
                relationships: data.graph.relationships
            };
        } else {
            showNotification('Error loading graph: ' + data.error, 'error');
        }
    } catch (error) {
        showNotification('Error: ' + error.message, 'error');
    }
}

// Update statistics display
function updateStatistics() {
    if (!currentGraph) return;

    const graph = graphData;
    const statsContent = document.getElementById('statsContent');

    const entityTypes = {};
    graph.entities.forEach(e => {
        const type = e.entity_type || 'UNKNOWN';
        entityTypes[type] = (entityTypes[type] || 0) + 1;
    });

    const relationshipTypes = {};
    graph.relationships.forEach(r => {
        const type = r.type || 'UNKNOWN';
        relationshipTypes[type] = (relationshipTypes[type] || 0) + 1;
    });

    let html = `
        <div class="stat-item">
            <span class="stat-label">Entities:</span>
            <span class="stat-value">${graph.entities.length}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Relationships:</span>
            <span class="stat-value">${graph.relationships.length}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Entity Types:</span>
            <div style="margin-top: 5px; font-size: 0.85rem;">
    `;

    for (const [type, count] of Object.entries(entityTypes)) {
        html += `<div>${type}: ${count}</div>`;
    }

    html += `
            </div>
        </div>
        <div class="stat-item">
            <span class="stat-label">Relationship Types:</span>
            <div style="margin-top: 5px; font-size: 0.85rem;">
    `;

    for (const [type, count] of Object.entries(relationshipTypes)) {
        html += `<div>${type}: ${count}</div>`;
    }

    html += `</div></div>`;
    statsContent.innerHTML = html;
}

// Visualize graph using Plotly
function visualizeGraph() {
    if (!currentGraph || graphData.entities.length === 0) {
        document.getElementById('noGraphMessage').style.display = 'block';
        return;
    }

    document.getElementById('noGraphMessage').style.display = 'none';

    // Create node positions using a simple force-directed layout simulation
    const positions = calculateNodePositions(graphData);

    // Prepare edge traces
    const edgeTraces = [];
    graphData.relationships.forEach(rel => {
        const sourcePos = positions[rel.source];
        const targetPos = positions[rel.target];

        if (sourcePos && targetPos) {
            edgeTraces.push({
                x: [sourcePos.x, targetPos.x],
                y: [sourcePos.y, targetPos.y],
                mode: 'lines',
                line: {
                    color: '#cbd5e0',
                    width: 2
                },
                hoverinfo: 'text',
                text: rel.type,
                showlegend: false
            });
        }
    });

    // Prepare node trace
    const nodeX = [];
    const nodeY = [];
    const nodeText = [];
    const nodeColors = [];

    const colorMap = {
        'PERSON': '#48bb78',
        'ORGANIZATION': '#4299e1',
        'LOCATION': '#ed8936',
        'ENTITY': '#9f7aea'
    };

    graphData.entities.forEach(entity => {
        const pos = positions[entity.id];
        if (pos) {
            nodeX.push(pos.x);
            nodeY.push(pos.y);
            nodeText.push(entity.id);
            nodeColors.push(colorMap[entity.entity_type] || '#9f7aea');
        }
    });

    const nodeTrace = {
        x: nodeX,
        y: nodeY,
        mode: 'markers+text',
        text: nodeText,
        textposition: 'top center',
        marker: {
            size: 20,
            color: nodeColors,
            line: {
                color: 'white',
                width: 2
            }
        },
        hoverinfo: 'text',
        showlegend: false
    };

    // Combine traces
    const traces = [...edgeTraces, nodeTrace];

    // Layout
    const layout = {
        title: `Graph: ${currentGraph}`,
        showlegend: false,
        hovermode: 'closest',
        xaxis: { showgrid: false, zeroline: false, showticklabels: false },
        yaxis: { showgrid: false, zeroline: false, showticklabels: false },
        plot_bgcolor: '#f7fafc',
        paper_bgcolor: '#f7fafc'
    };

    Plotly.newPlot('graphVisualization', traces, layout, { responsive: true });
}

// Simple force-directed layout for node positions
function calculateNodePositions(graph) {
    const positions = {};
    const entities = graph.entities;
    const relationships = graph.relationships;

    // Initialize random positions
    entities.forEach((entity, i) => {
        const angle = (i / entities.length) * 2 * Math.PI;
        const radius = 5;
        positions[entity.id] = {
            x: radius * Math.cos(angle),
            y: radius * Math.sin(angle)
        };
    });

    // Simple force simulation (very basic)
    for (let iter = 0; iter < 50; iter++) {
        // Repulsion between all nodes
        entities.forEach(e1 => {
            entities.forEach(e2 => {
                if (e1.id !== e2.id) {
                    const dx = positions[e1.id].x - positions[e2.id].x;
                    const dy = positions[e1.id].y - positions[e2.id].y;
                    const dist = Math.sqrt(dx * dx + dy * dy) || 0.1;
                    const force = 0.1 / (dist * dist);

                    positions[e1.id].x += (dx / dist) * force;
                    positions[e1.id].y += (dy / dist) * force;
                }
            });
        });

        // Attraction along edges
        relationships.forEach(rel => {
            if (positions[rel.source] && positions[rel.target]) {
                const dx = positions[rel.target].x - positions[rel.source].x;
                const dy = positions[rel.target].y - positions[rel.source].y;
                const dist = Math.sqrt(dx * dx + dy * dy) || 0.1;
                const force = dist * 0.01;

                positions[rel.source].x += (dx / dist) * force;
                positions[rel.source].y += (dy / dist) * force;
                positions[rel.target].x -= (dx / dist) * force;
                positions[rel.target].y -= (dy / dist) * force;
            }
        });
    }

    return positions;
}

// Display data in explorer
function displayDataExplorer() {
    if (!currentGraph) return;

    // Display entities
    const entitiesContainer = document.getElementById('entitiesList');
    if (graphData.entities.length === 0) {
        entitiesContainer.innerHTML = '<p class="text-muted">No entities yet</p>';
    } else {
        entitiesContainer.innerHTML = '';
        graphData.entities.forEach(entity => {
            const card = document.createElement('div');
            card.className = 'entity-card';
            card.innerHTML = `
                <div class="card-header">${entity.id}</div>
                <div class="card-details">
                    Type: ${entity.entity_type || 'ENTITY'}<br>
                    ${Object.entries(entity).filter(([k]) => k !== 'id' && k !== 'entity_type').map(([k, v]) => `${k}: ${v}`).join('<br>')}
                </div>
            `;
            entitiesContainer.appendChild(card);
        });
    }

    // Display relationships
    const relationshipsContainer = document.getElementById('relationshipsList');
    if (graphData.relationships.length === 0) {
        relationshipsContainer.innerHTML = '<p class="text-muted">No relationships yet</p>';
    } else {
        relationshipsContainer.innerHTML = '';
        graphData.relationships.forEach(rel => {
            const card = document.createElement('div');
            card.className = 'relationship-card';
            card.innerHTML = `
                <div class="card-header">${rel.source} → ${rel.target}</div>
                <div class="card-details">
                    Type: ${rel.type}
                </div>
            `;
            relationshipsContainer.appendChild(card);
        });
    }
}

// Create new graph
async function createGraph(event) {
    event.preventDefault();

    const graphName = document.getElementById('graphName').value;

    try {
        const response = await fetch('/api/graphs', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: graphName })
        });

        const data = await response.json();

        if (data.success) {
            showNotification('Graph created successfully!', 'success');
            closeModal('createGraphModal');
            document.getElementById('createGraphForm').reset();
            await loadGraphs();
        } else {
            showNotification('Error: ' + data.error, 'error');
        }
    } catch (error) {
        showNotification('Error: ' + error.message, 'error');
    }
}

// Delete graph
async function deleteGraph(graphName, event) {
    event.stopPropagation();

    if (!confirm(`Are you sure you want to delete graph "${graphName}"?`)) {
        return;
    }

    try {
        const response = await fetch(`/api/graphs/${graphName}`, {
            method: 'DELETE'
        });

        const data = await response.json();

        if (data.success) {
            showNotification('Graph deleted successfully!', 'success');
            if (currentGraph === graphName) {
                currentGraph = null;
                graphData = { entities: [], relationships: [] };
            }
            await loadGraphs();
        } else {
            showNotification('Error: ' + data.error, 'error');
        }
    } catch (error) {
        showNotification('Error: ' + error.message, 'error');
    }
}

// Add data to graph
async function addData(event) {
    event.preventDefault();

    if (!currentGraph) {
        showNotification('Please select a graph first', 'error');
        return;
    }

    const dataType = document.getElementById('dataType').value;

    try {
        let response;

        if (dataType === 'text') {
            const text = document.getElementById('documentText').value;
            response = await fetch(`/api/graphs/${currentGraph}/documents`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });
        } else {
            const jsonText = document.getElementById('jsonData').value;
            const jsonData = JSON.parse(jsonText);
            response = await fetch(`/api/graphs/${currentGraph}/structured-data`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ data: jsonData })
            });
        }

        const data = await response.json();

        if (data.success) {
            showNotification(`Added ${data.result.entities_added} entities and ${data.result.relationships_added} relationships!`, 'success');
            closeModal('addDataModal');
            document.getElementById('addDataForm').reset();
            await loadGraphData(currentGraph);
            updateStatistics();
            visualizeGraph();
            displayDataExplorer();
        } else {
            showNotification('Error: ' + data.error, 'error');
        }
    } catch (error) {
        showNotification('Error: ' + error.message, 'error');
    }
}

// Query graph
async function queryGraph(event) {
    event.preventDefault();

    if (!currentGraph) {
        showNotification('Please select a graph first', 'error');
        return;
    }

    const query = document.getElementById('queryText').value;
    const maxResults = document.getElementById('maxResults').value;

    try {
        const response = await fetch(`/api/graphs/${currentGraph}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query, max_results: parseInt(maxResults) })
        });

        const data = await response.json();

        if (data.success) {
            displayQueryResults(data.result);
            closeModal('queryModal');
            switchTab('query');
        } else {
            showNotification('Error: ' + data.error, 'error');
        }
    } catch (error) {
        showNotification('Error: ' + error.message, 'error');
    }
}

// Display query results
function displayQueryResults(result) {
    const container = document.getElementById('queryResults');

    if (result.entities.length === 0) {
        container.innerHTML = '<p class="text-muted">No results found</p>';
        return;
    }

    container.innerHTML = `<h4>Query: "${result.query}"</h4><p>Found ${result.total_results} results:</p>`;

    result.entities.forEach(entity => {
        const item = document.createElement('div');
        item.className = 'result-item';

        const entityRels = result.relationships.filter(r =>
            r.source === entity.id || r.target === entity.id
        );

        let relsHtml = '';
        if (entityRels.length > 0) {
            relsHtml = '<div class="result-relationships"><strong>Relationships:</strong>';
            entityRels.forEach(rel => {
                relsHtml += `<div class="relationship-item">${rel.source} -[${rel.type}]-> ${rel.target}</div>`;
            });
            relsHtml += '</div>';
        }

        item.innerHTML = `
            <div class="result-entity">
                ${entity.id} (${entity.entity_type || 'ENTITY'})
            </div>
            <div style="font-size: 0.9rem; color: #4a5568;">
                ${Object.entries(entity).filter(([k]) => k !== 'id' && k !== 'entity_type').map(([k, v]) => `${k}: ${v}`).join(', ')}
            </div>
            ${relsHtml}
        `;

        container.appendChild(item);
    });
}

// Add entity
async function addEntity(event) {
    event.preventDefault();

    if (!currentGraph) {
        showNotification('Please select a graph first', 'error');
        return;
    }

    const entityId = document.getElementById('entityId').value;
    const entityType = document.getElementById('entityType').value || 'ENTITY';
    const propertiesText = document.getElementById('entityProperties').value;

    let properties = {};
    if (propertiesText) {
        try {
            properties = JSON.parse(propertiesText);
        } catch (e) {
            showNotification('Invalid JSON in properties', 'error');
            return;
        }
    }

    try {
        const response = await fetch(`/api/graphs/${currentGraph}/entities`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                id: entityId,
                type: entityType,
                properties: properties
            })
        });

        const data = await response.json();

        if (data.success) {
            showNotification('Entity added successfully!', 'success');
            closeModal('addEntityModal');
            document.getElementById('addEntityForm').reset();
            await loadGraphData(currentGraph);
            updateStatistics();
            visualizeGraph();
            displayDataExplorer();
        } else {
            showNotification('Error: ' + data.error, 'error');
        }
    } catch (error) {
        showNotification('Error: ' + error.message, 'error');
    }
}

// Add relationship
async function addRelationship(event) {
    event.preventDefault();

    if (!currentGraph) {
        showNotification('Please select a graph first', 'error');
        return;
    }

    const source = document.getElementById('relSource').value;
    const target = document.getElementById('relTarget').value;
    const type = document.getElementById('relType').value;
    const propertiesText = document.getElementById('relProperties').value;

    let properties = {};
    if (propertiesText) {
        try {
            properties = JSON.parse(propertiesText);
        } catch (e) {
            showNotification('Invalid JSON in properties', 'error');
            return;
        }
    }

    try {
        const response = await fetch(`/api/graphs/${currentGraph}/relationships`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                source: source,
                target: target,
                type: type,
                properties: properties
            })
        });

        const data = await response.json();

        if (data.success) {
            showNotification('Relationship added successfully!', 'success');
            closeModal('addRelationshipModal');
            document.getElementById('addRelationshipForm').reset();
            await loadGraphData(currentGraph);
            updateStatistics();
            visualizeGraph();
            displayDataExplorer();
        } else {
            showNotification('Error: ' + data.error, 'error');
        }
    } catch (error) {
        showNotification('Error: ' + error.message, 'error');
    }
}

// UI Helper Functions
function switchTab(tabName) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

    event.target.classList.add('active');
    document.getElementById(tabName).classList.add('active');
}

function showCreateGraphModal() {
    document.getElementById('createGraphModal').style.display = 'block';
}

function showAddDataModal() {
    if (!currentGraph) {
        showNotification('Please select a graph first', 'error');
        return;
    }
    document.getElementById('addDataModal').style.display = 'block';
}

function showQueryModal() {
    if (!currentGraph) {
        showNotification('Please select a graph first', 'error');
        return;
    }
    document.getElementById('queryModal').style.display = 'block';
}

function showAddEntityModal() {
    if (!currentGraph) {
        showNotification('Please select a graph first', 'error');
        return;
    }
    document.getElementById('addEntityModal').style.display = 'block';
}

function showAddRelationshipModal() {
    if (!currentGraph) {
        showNotification('Please select a graph first', 'error');
        return;
    }
    document.getElementById('addRelationshipModal').style.display = 'block';
}

function closeModal(modalId) {
    document.getElementById(modalId).style.display = 'none';
}

function toggleDataInput() {
    const dataType = document.getElementById('dataType').value;
    const textInput = document.getElementById('textInput');
    const jsonInput = document.getElementById('jsonInput');

    if (dataType === 'text') {
        textInput.style.display = 'block';
        jsonInput.style.display = 'none';
    } else {
        textInput.style.display = 'none';
        jsonInput.style.display = 'block';
    }
}

function showNotification(message, type) {
    const notification = document.getElementById('notification');
    notification.textContent = message;
    notification.className = 'notification ' + type;
    notification.style.display = 'block';

    setTimeout(() => {
        notification.style.display = 'none';
    }, 3000);
}

// Close modals when clicking outside
window.onclick = function(event) {
    if (event.target.classList.contains('modal')) {
        event.target.style.display = 'none';
    }
}
