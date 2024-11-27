let rowCount = 0;
let isIterationModeEnabled = false;
const iterationModeCheckbox = document.getElementById('iterationMode');
let currentlyCheckedBox = null;
let existingSimulationNames=""
let singleRowAdded = false;
document.addEventListener("DOMContentLoaded", function () {
    const csrfToken = document.querySelector('meta[name="csrf-token"]').getAttribute('content');
    const runSimulationButton = document.getElementById('runSimulationButton');
    const addRowButton = document.getElementById('id_add_row');
    //set simulations to the value of the simulations variable in the template
    
    existingSimulationNames= previousSimulations;
    

    registerSimulationButtonListener(runSimulationButton, csrfToken);
    registerAddRowButtonListener(addRowButton, iterationModeCheckbox);
    setupIterationModeToggle(iterationModeCheckbox);
});


function registerSimulationButtonListener(button, csrfToken) {
    button.addEventListener('click', function (event) {
        event.preventDefault();
        handleSubmitSimulationForm(csrfToken);
    });
}

async function handleSubmitSimulationForm(csrfToken) {
    const requestData = buildRequestData();
    if (!validateRequestData(requestData)) return;

    try {
        const response = await fetch('/run-simulation/', {
            method: 'POST',
            headers: {'Content-Type': 'application/json', 'X-CSRFToken': csrfToken},
            body: JSON.stringify(requestData)
        });
        const result = await response.json();
        fetchSimulationData(result.name,requestData);
    } catch (error) {
        console.error('Error:', error);
        alert("An error occurred during simulation submission.");
    }
}

function handelRowData() {
    let rowData = [];
    let iterationData = {};
    const rows = document.querySelectorAll('.row-container'); 
    rows.forEach((row,index) => {
        const plantType = document.getElementById(`plant-type-${index+1}`).value;
        const plantingType = document.getElementById(`planting-type-${index+1}`).value;
        const rowWidthInput = document.getElementById(`row-width-${index+1}`);
        const rowSpacingInput = document.getElementById(`row-spacing-${index+1}`);
        const numIterationsInput = document.getElementById(`num-iterations-${index+1}`);



        const rowDetails = {
            plantType: plantType,
            plantingType: plantingType,
            stripWidth: parseFloat(rowWidthInput.value),
            rowSpacing: parseFloat(rowSpacingInput.value),
            numSets: parseInt(numIterationsInput.value)
        };
     
        rowData.push(rowDetails);
    });
    return rowData;
}
function handelIterationData() {
    let iterationData = {};
    const rows = document.querySelectorAll('.row-container'); 
    rows.forEach((row,index) => {
        if (isIterationModeEnabled) {
            const rowWidthClone = document.getElementById(`row-width-1-clone`);
            const rowSpacingClone = document.getElementById(`row-spacing-1-clone`);
            const numSetsClone = document.getElementById(`num-iterations-1-clone`);
            
            if (rowWidthClone) {
                iterationData.stripWidth = parseFloat(rowWidthClone.value);
            }
            if (rowSpacingClone) {
                iterationData.rowSpacing = parseFloat(rowSpacingClone.value);
            }
            if (numSetsClone) {
                iterationData.numSets = parseInt(numSetsClone.value);
            }
        }
    });

    
    if (isIterationModeEnabled && rowCount>1) {
        iterationData.rows = -99;  
    }
    if (isIterationModeEnabled) {
        const startDateClone = document.getElementById(`id_sim-startDate-clone`);
        const stepSizeClone = document.getElementById(`id_sim-stepSize-clone`);
        const rowLengthClone = document.getElementById(`id_sim-length-clone`);

        if (startDateClone) {
            iterationData.startDate = startDateClone.value;
        }
        if (stepSizeClone) {
            iterationData.stepSize = parseInt(stepSizeClone.value);
        }
        if (rowLengthClone) {
            iterationData.rowLength = parseInt(rowLengthClone.value);
        }
    }
    return iterationData;
}


function buildRequestData() {
    // Build the request data object by collecting values from form inputs
    const requestData = {
        simName: document.getElementById('unique_name_id').value,
        startDate: document.getElementById('id_sim-startDate').value,
        stepSize: parseInt(document.getElementById('id_sim-stepSize').value),
        rowLength: parseInt(document.getElementById('id_sim-length').value),
        harvestType: document.getElementById('id_sim-harvestType').value,
        rows: handelRowData(),
        iterationMode: isIterationModeEnabled,
        iterationData: handelIterationData(),
        useTemperature: document.getElementById('useTemperature').checked,
        useWater: document.getElementById('useWaterlevel').checked,
        allowWeedgrowth: document.getElementById('allowWeedgrowth').checked
    };
    console.log("inputData",requestData);
    return requestData;
}


function validateRequestData(data) {
    // Add validation logic for request data
    if (!data.simName) {
        alert("Please enter a simulation name.");
        return false;
    }

    if (existingSimulationNames.includes(data.simName)) {
        alert("Simulation name already exists. Please choose a different name.");
        return false;
    }

    if (!data.startDate) {
        alert("Please enter a start date between 2022-09-30 and 2024-04-01.");
        return false;
    }

    // Validate date range
    const minDate = new Date("2022-09-30");
    const maxDate = new Date("2024-04-01");

    // Assuming data.startDate is in "YYYY-MM-DD" format
    const inputDate = new Date(data.startDate);

    // Check if the inputDate is a valid date object
    if (isNaN(inputDate.getTime())) {
        alert("Invalid date format. Please enter a valid date.");
        return false;
    }

    if (inputDate < minDate || inputDate > maxDate) {
        alert("Start date must be between 2022-09-30 and 2024-04-01.");
        return false;
    }


    // Validate other inputs
    
    if (isNaN(data.stepSize) || data.stepSize <= 0) {
        alert("Please enter a valid step size greater than zero.");
        return false;
    }

    if (isNaN(data.rowLength) || data.rowLength <= 0) {
        alert("Please enter a valid row length greater than zero.");
        return false;
    }

    if (!data.harvestType) {
        alert("Please select a harvest type.");
        return false;
    }

    if (data.rows.length === 0) {
        alert("Please add at least one row to the simulation.");
        return false;
    }


    if (data.IterationMode && data.iterationData.keys(data.iterationData).length === 0) {
        alert("Please add a value for the iteration mode.");
        return false;
        
    }
    return true;
}


async function fetchSimulationData(simulationName,requestData) {
    try {
        const response = await fetch(`/api/get_simulation_data/?name=${encodeURIComponent(simulationName)}`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let jsonText = '';
        let accumulatedData = [];

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            jsonText += decoder.decode(value, { stream: true });

            // Process each line to extract JSON objects
            const lines = jsonText.split('\n');
            jsonText = lines.pop(); // Keep the last incomplete line

            for (let line of lines) {
                if (line.trim()) {  // Ignore empty lines
                    try {
                        const parsedItem = JSON.parse(line);
                        accumulatedData.push(parsedItem); // Add each parsed item to the list
                    } catch (e) {
                        console.warn("Error parsing NDJSON line:", e);
                    }
                }
            }
        }

        // All data is now collected; pass it to the plotting functions
        console.log('Simulation data:', accumulatedData);
        setupCarousel(accumulatedData,requestData);
    } catch (error) {
        console.error('Failed to fetch NDJSON data:', error);
    }
}

function registerAddRowButtonListener(button, iterationModeCheckbox) {
    button.addEventListener('click', function () {
        addNewRow(iterationModeCheckbox.checked);
    });
}

function addNewRow(isIterationModeEnabled) {
    const allowMultipleRowsChecked = document.getElementById('multiRowCheckbox').checked;

    if (isIterationModeEnabled) {
        if (!allowMultipleRowsChecked && singleRowAdded) {
            alert("You can only add one row when 'Allow Multiple Rows' is unchecked in iteration mode.");
            return;
        }
    } else {
        if (!allowMultipleRowsChecked && singleRowAdded) {
            singleRowAdded = false; 
        }
    }

    rowCount++;
    const rowList = document.getElementById('row-list');
    const newRow = document.createElement('div');
    newRow.className = 'row-container';
    newRow.id =` row-${rowCount}`;
    let plantOptionsHtml = '';
    plants.forEach(plant => {
        plantOptionsHtml += `<option value="${plant.fields.name}">${plant.fields.name}</option>`;
    });
    newRow.innerHTML = `
    <!-- Plant Type -->
    <div class="row mb-2">
        <div class="col-md-8 mx-auto">
            <label for="plant-type-${rowCount}" class="form-label">Plant Type:</label>
                                <select id="plant-type-${rowCount}" class="form-control plant-type">
                ${plantOptionsHtml}
            </select>
        </div>
    </div>

    <!-- Row Width -->
    <div class="row mb-2 justify-content-center">
        <div class="col-md-8">
            <div class="form-check">
                <input type="checkbox" class="form-check-input testing-checkbox d-none" id="splitRowWidth-${rowCount}">   
            </div>
            <label for="row-width-${rowCount}" class="form-label">Width of the row (cm):</label>
            <input type="number" id="row-width-${rowCount}" class="form-control row-width" placeholder="Width in cm" value="80" min="1" max="200">
        </div>
    </div>
    
    <!-- Planting Type -->
    <div class="row mb-2">
        <div class="col-md-8 mx-auto">
            <label for="planting-type-${rowCount}" class="form-label">Planting Type:</label>
                <select id="planting-type-${rowCount}" class="form-control planting-type">
                <option value="grid">Grid</option>
                <option value="alternating">Alternating</option>
                <option value="random">Random</option>
                <option value="empty">Empty</option>
            </select>
        </div>
    </div>

    <!-- Row Spacing -->
    <div class="row mb-2 justify-content-center">
        <div class="col-md-8">
            <div class="form-check">
                <input type="checkbox" class="form-check-input testing-checkbox d-none" id="splitRowSpacing-${rowCount}">         
           </div>
            <label for="row-spacing-${rowCount}" class="form-label">Space between plants (cm):</label>
            <input type="number" id="row-spacing-${rowCount}" class="form-control row-spacing" placeholder="Row spacing in cm" value="30" min="1" max="100">
        </div>
    </div>


    <!-- Number of Iterations -->
    <div class="row mb-2 justify-content-center">
        <div class="col-md-8">
            <div class="form-check">
                <input type="checkbox" class="form-check-input testing-checkbox d-none" id="splitNumIterations-${rowCount}">
            </div>
            <label for="num-iterations-${rowCount}" class="form-label">Number of Iterations:</label>
            <input type="number" id="num-iterations-${rowCount}" class="form-control numIterations" placeholder="Number of Iterations" value="1" min="1" max="10">
        </div>
    </div>

    <!-- Delete Row Button -->
    <div class="row mb-2">
        <div class="col-md-12">
            <button class="btn btn-danger remove-btn">Delete Row</button>
        </div>
    </div>
    `;


    rowList.appendChild(newRow);

    // Ensure the newly added testing checkboxes are initially hidden
    const newTestingCheckboxes = document.querySelectorAll('.testing-checkbox');
    // Toggle visibility based on conditions
    if (isIterationModeEnabled) {
        newTestingCheckboxes.forEach(checkbox => {
            checkbox.classList.remove('d-none');
        });
    }
    
    // Remove button event listener
const removeButton = newRow.querySelector('.remove-btn');
removeButton.addEventListener('click', function () {
    // Remove the current row
    newRow.remove();

    // If not allowing multiple rows, reset the singleRowAdded flag
    if (!allowMultipleRowsChecked) {
        singleRowAdded = false;
    }

    // Decrement the rowCount
    rowCount--;

    // Update IDs and names for remaining rows to ensure proper sequence
    const rows = document.querySelectorAll('.row-container');
    rows.forEach((row, index) => {
        // Set new ID for the row
        row.id = `row-${index + 1}`;

        // Update IDs, names, and labels for each input and select element in the row
        row.querySelectorAll('input, select').forEach(input => {
            const oldId = input.id;
            const newId = oldId.replace(/-\d+/, `-${index + 1}`);

            input.id = newId;
            input.name = input.name.replace(/-\d+/, `-${index + 1}`);

            // Update the label 'for' attribute to match the new input ID
            const label = row.querySelector(`label[for="${oldId}"]`);
            if (label) {
                label.setAttribute('for', newId);
            }
        });
    });
});

    if (iterationModeCheckbox.checked) {
        if (!document.getElementById('multiRowCheckbox').checked) {
            newTestingCheckboxes.forEach(checkbox => checkbox.classList.remove('d-none'));
            singleRowAdded = true;
        }
    }
    setupIterationModeToggle(iterationModeCheckbox);
};
    


function handleSplitForms(event) {
    const targetCheckbox = event.target;

    // Uncheck the previously checked checkbox
    if (currentlyCheckedBox && currentlyCheckedBox !== targetCheckbox) {
        currentlyCheckedBox.checked = false;
        if (currentlyCheckedBox.id === 'multiRowCheckbox') {
            // Reset to allow only one row
            singleRowAdded = false;
        } else {
            resetForm(currentlyCheckedBox.closest('.row'));
        }
    }

    if (targetCheckbox.checked) {
        currentlyCheckedBox = targetCheckbox;
        if (targetCheckbox.id === 'multiRowCheckbox') {
            // Allow multiple rows
            singleRowAdded = false;
        } else {
            // Split the form
            const formContainer = targetCheckbox.closest('.row');
            formContainer.classList.add('justify-content-center');
            const clonedForm = formContainer.querySelector('.cloned-form');

            if (!clonedForm) {
                splitForm(formContainer);
            }
        }
    } else {
        if (targetCheckbox.id === 'multiRowCheckbox') {
            // Reset to allow only one row
            singleRowAdded = false;
        } else {
            // Reset the form
            const formContainer = targetCheckbox.closest('.row');
            resetForm(formContainer);
        }
        currentlyCheckedBox = null;
    }
}


function splitForm(formContainer) {
    const formCol = formContainer.querySelector('.col-md-8');
    formCol.classList.remove('col-md-8');
    formCol.classList.add('col-md-4'); // Shrink to half width

    const clone = formCol.cloneNode(true);

    // Update the IDs of the cloned form inputs and handle checkboxes
    const inputFields = clone.querySelectorAll('input, select, .form-check-input');
    inputFields.forEach((inputField) => {
        inputField.id += '-clone'; // Append '-clone' to each ID
        // Additionally, check if the input field is a checkbox and remove it with its label
        if (inputField.type === 'checkbox') {
            const label = inputField.nextElementSibling; // assuming label immediately follows the checkbox
            const formGroup = inputField.parentElement; // assuming the checkbox and label are wrapped in a parent container like a div
            inputField.remove(); // remove the checkbox
            if (label && label.tagName === 'LABEL') {
                label.remove(); // remove the label
            }

            // Add margin to the next form group to maintain alignment
            if (!formGroup.querySelector(`label[for='row-width-${rowCount}']`) &&
                !formGroup.querySelector(`label[for='num-iterations-${rowCount}']`) &&
                !formGroup.querySelector(`label[for='row-spacing-${rowCount}']`)) {
                // Add margin to the next form group to maintain alignment
                const nextFormGroup = formGroup.nextElementSibling;
                if (nextFormGroup) {
                    nextFormGroup.style.marginTop = '1.5rem'; // Adjust the margin as needed
                }
            }
        }
        
        // If the element is a checkbox, adjust as necessary
        if (inputField.type === 'checkbox') {
            // The label directly following the checkbox needs to be updated to refer to the new checkbox id
            const checkboxLabel = inputField.nextElementSibling;
            if (checkboxLabel && checkboxLabel.tagName === 'LABEL') {
                checkboxLabel.htmlFor += '-clone';
            }
        }
    });

    clone.classList.add('cloned-form');
    const rowContainer = formContainer.closest('.row');
    rowContainer.appendChild(clone);

    // Adding justify-content-center to the row to align cloned forms in the center
    rowContainer.classList.add('justify-content-center');
}


function resetForm(formContainer) {
    const formCol = formContainer.querySelector('.col-md-4');
    if (formCol) {
        formCol.classList.remove('col-md-4');
        formCol.classList.add('col-md-8'); // Reset to full width
    }

    const clonedForm = formContainer.querySelector('.cloned-form');
    if (clonedForm) {
        clonedForm.remove(); // Remove cloned form
    }
}


function resetAllForms() {
    const rows = document.querySelectorAll('.row');
    rows.forEach(row => resetForm(row));
}
// Add new row when button is clicked

function setupIterationModeToggle(checkbox) {
    const iterationCheckboxes = document.querySelectorAll('.testing-checkbox');
    checkbox.addEventListener('change', function () {
        if(checkbox.checked){
            isIterationModeEnabled = true;
            iterationCheckboxes.forEach(checkbox => {
                checkbox.classList.remove('d-none');
                checkbox.addEventListener('change', handleSplitForms);
            });
        } else {
            isIterationModeEnabled = false;
            iterationCheckboxes.forEach(checkbox => {
                checkbox.classList.add('d-none');
                checkbox.checked = false;
            });
            resetAllForms();
            currentlyCheckedBox = null;
        }
    });
}