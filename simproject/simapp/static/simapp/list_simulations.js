// Event listener setup for all "Plot" buttons to fetch and display simulation data.
document.querySelectorAll('.plot-btn').forEach(button => {
    button.addEventListener('click', () => {
        const simulationName = button.getAttribute('data-simulation');
        

        fetchSimulationData(simulationName);
    });
});

// Fetches and processes simulation data for plotting.
async function fetchSimulationData(simulationName) {
    try {
        const response = await fetch(`/list-simulations/?plot_simulation=${encodeURIComponent(simulationName)}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let jsonText = '';
        let accumulatedData = [];

        // Reading and parsing NDJSON stream.
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            jsonText += decoder.decode(value, { stream: true });
            const lines = jsonText.split('\n');
            jsonText = lines.pop();  // Handling the last incomplete line.

            lines.forEach(line => {
                if (line.trim()) {
                    try {
                        const parsedItem = JSON.parse(line);
                        accumulatedData.push(parsedItem);  // Accumulating parsed data.
                    } catch (e) {
                        console.warn("Error parsing NDJSON line:", e);
                    }
                }
            });
        }
        setupCarousel(accumulatedData);  // Plotting data using a carousel.
    } catch (error) {
        console.error('Failed to fetch NDJSON data:', error);
    }
}

// Event listener setup for all "Delete" buttons to remove simulations.
document.querySelectorAll('.delete-btn').forEach(button => {
    button.addEventListener('click', () => {
        const csrfToken = document.querySelector('meta[name="csrf-token"]').getAttribute('content');
        const simulationName = button.getAttribute('data-simulation');
        if (confirm(`Are you sure you want to delete simulation "${simulationName}"?`)) {
            deleteSimulation(simulationName, csrfToken);
        }
    });
});

// Deletes a simulation from the server and updates the UI.
function deleteSimulation(simulationName, csrfToken) {
    fetch('/list-simulations/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
            'X-CSRFToken': csrfToken // CSRF token for secure requests.
        },
        body: new URLSearchParams({
            'delete_simulation': simulationName
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            // Remove the simulation entry from the DOM.
            document.querySelector(`li[data-simulation="${simulationName}"]`).remove();
        } else {
            console.error('Failed to delete simulation', data.message);
        }
    })
    .catch(error => console.error('Error:', error));
}