function setupCarousel(allData, inputData) {
    // Elements for each carousel
    const carouselElement1 = document.getElementById('carouselCaption1');
    const carouselElement2 = document.getElementById('carouselCaption2');
    const carouselElement3 = document.getElementById('carouselCaption3');


    const carousel1 = new bootstrap.Carousel(carouselElement1, { wrap: true, interval: false });
    const carousel2 = new bootstrap.Carousel(carouselElement2, { wrap: true, interval: false });
    const carousel3 = new bootstrap.Carousel(carouselElement3, { wrap: true, interval: false });
    let totalSlides = allData.length;
    
    // Create carousel slides and plots once
    createAndPopulateSlides(totalSlides, 'carouselCaption1', 'plot1_', allData, displayFirstPlot);
    createAndPopulateSlides(totalSlides, 'carouselCaption2', 'plot2_', allData, displaySecondPlot);
    createAndPopulateSlides(totalSlides, 'carouselCaption3', 'heatmap_', allData, displayHeatmap, inputData);  // Pass inputData only for the heatmap

    let currentSlideIndex = 0; // Track current slide index globally

    // Function to synchronize all carousels
    function syncCarousels(newIndex) {
        // Update the global index
        currentSlideIndex = newIndex;

        // Move each carousel to the new index
        carousel1.to(currentSlideIndex);
        carousel2.to(currentSlideIndex);
        carousel3.to(currentSlideIndex);
    }
    // Sync carousel slides
    [carouselElement1, carouselElement2, carouselElement3].forEach(element => {
        element.addEventListener('slid.bs.carousel', function (event) {
            let newIndex = event.to; // Get the new slide index from the event
            if (newIndex !== currentSlideIndex) { // Check if the new index matches the global current index
                syncCarousels(newIndex); // Sync if different
            }
            document.getElementById('carousel1Index').innerText = event.to + 1;
            document.getElementById('carousel2Index').innerText = event.to + 1;
            document.getElementById('carousel3Index').innerText = event.to + 1;
        });
    });

    function addControlCooldown(duration) {
        const controls = document.querySelectorAll('.carousel-control-prev, .carousel-control-next');
        controls.forEach(control => {
            control.disabled = true; // Disable control
            setTimeout(() => {
                control.disabled = false; // Re-enable control after the duration
            }, duration);
        });
    }
    document.querySelectorAll('.carousel-control-prev, .carousel-control-next').forEach(control => {
        control.addEventListener('click', () => addControlCooldown(1500)); // Cooldown of 500 milliseconds
    });

    // Update the second carousel plots based on the selected plot type
    const selectPlotType = document.getElementById('selectPlotType');
    selectPlotType.addEventListener('change', function() {
        const selectedPlotType = selectPlotType.value;
        updateAllSecondCarouselPlots(selectedPlotType, allData);
    });

    function updateAllSecondCarouselPlots(plotType, allData) {
        const allPlot2Elements = document.querySelectorAll('[id^="plot2_"]');

        allPlot2Elements.forEach((plotContainer, index) => {
            const plotId = plotContainer.id; // Ensure IDs are unique and correctly assigned
            if (allData[index]) { // Check if data at index exists
                const data = allData[index];
                displaySecondPlot(data, plotId, plotType);
            } else {
                console.error(`No data available for plot index ${index}`);
            }
        });
    }
  
    setupComparisonPlot(allData);
    
}


function createAndPopulateSlides(numSlides, carouselId, plotPrefix, allData, plotFunction, inputData = null) {
    const carouselInner = document.querySelector(`#${carouselId} .carousel-inner`);
    const carouselIndicators = document.querySelector(`#${carouselId} .carousel-indicators`);

    carouselInner.innerHTML = '';  // Clear existing slides
    carouselIndicators.innerHTML = '';  // Clear indicators

    for (let i = 0; i < numSlides; i++) {
        const slide = document.createElement('div');
        slide.classList.add('carousel-item');
        if (i === 0) slide.classList.add('active');
        const content = document.createElement('div');
        const plotId = `${plotPrefix}${i}`;
        content.innerHTML += `<div id="${plotId}" class="plot-container"></div>`;
        slide.appendChild(content);
        carouselInner.appendChild(slide);
        const indicator = document.createElement('li');
        indicator.setAttribute('data-bs-target', `#${carouselId}`);
        indicator.setAttribute('data-bs-slide-to', i);
        if (i === 0) indicator.classList.add('active');
        carouselIndicators.appendChild(indicator);
    }

    // Initialize the plot for each slide
    setTimeout(() => {
        for (let i = 0; i < numSlides; i++) {
            const plotId = `${plotPrefix}${i}`;
            if (plotPrefix === 'heatmap_') {
                plotFunction(allData[i], plotId, inputData);  // Pass inputData to heatmap
            } else {
                plotFunction(allData[i], plotId, 'growth');  // No inputData for other plots
            }
        }
    }, 0);
}


// Function to setup the comparison plot
function setupComparisonPlot(result) {
const yAxisSelect = document.getElementById('y-axis-select');

// Initial plot with default value (growth)
plotComparison(result, 'growth');

// Listen for changes in the Y-Axis dropdown and update the plot
yAxisSelect.addEventListener('change', function () {
    const selectedValue = this.value;
    plotComparison(result, selectedValue);
});
}

function plotComparison(allData, yAxisKey) {
if (!allData || allData.length === 0) {
    console.error("No data available for plotting.");
    return;
}

const xValues = allData.map(entry => entry.param_value); // x-axis based on the parameter values


// Define full titles with units for each yAxisKey
const titles = {
    "profit_per_plant": "Profit per Plant (€ / Plant)",
    "profit_per_area": "Profit per Area (€ / m²)",
    "yield_per_plant": "Yield per Plant (g / Plant)",
    "growth_per_plant": "Average Growth-rate per Plant (g/TU per Plant)",
    "yield_per_area": "Yield per Area (g per m²)",
    "growth_per_area": "Growth per Area (g/TU per m²)",
    "number_of_plants": "Number of Plants (n)",
    "growth": "Average Growth-rate (g/TU)",
    "yield": "Total yield (g)",
    "profit": " Total profit (€)"
};

const yValues = allData.map(entry => {
    const lastOutput = entry.outputs[entry.outputs.length - 1]; // Get the last output of each iteration
    const area = lastOutput.map[0].length * lastOutput.map.length;
    switch (yAxisKey) {
        case "profit_per_plant":
            return lastOutput.profit / lastOutput.num_plants;
        case "profit_per_area":
            return lastOutput.profit / area*10000;
        case "yield_per_plant":
            return lastOutput.yield / lastOutput.num_plants;
        case "growth_per_plant":
            const meanGrowthPerPlant = entry.outputs.reduce((sum, output) => sum + output.growth, 0) / entry.outputs.length;
            return meanGrowthPerPlant / lastOutput.num_plants;
        case "yield_per_area":
            return lastOutput.yield / area*10000;
        case "growth_per_area":
            const meanGrowthPerArea = entry.outputs.reduce((sum, output) => sum + output.growth, 0) / entry.outputs.length;
            return meanGrowthPerArea / area*10000;
        case "number_of_plants":
            return lastOutput.num_plants;
        case "growth":
            return entry.outputs.reduce((sum, output) => sum + output.growth, 0) / entry.outputs.length;
        default:
            return lastOutput[yAxisKey];
    }
});

const trace = {
    x: xValues,
    y: yValues,
    type: 'scatter',
    mode: 'lines+markers',
    name: `Comparison of ${yAxisKey.replace('_', ' ')}`,
    line: {
        color: 'rgb(126,185,48)',
        width: 3
    },
    marker: {
        color: 'rgb(126,185,48)',
        size: 8
    }
};

const layout = {
    title: `Comparison of ${yAxisKey.replace('_', ' ')} Across Iterations`,
    xaxis: {
        title: "Iteration Index",
        showline: true,
        showgrid: true,
        showticklabels: true,
        linecolor: 'black',
        linewidth: 2,
        mirror: true
    },
    yaxis: {
        title: titles[yAxisKey] || yAxisKey.charAt(0).toUpperCase() + yAxisKey.slice(1).replace('_', ' '),
        showline: true,
        showgrid: true,
        showticklabels: true,
        linecolor: 'black',
        linewidth: 2,
        mirror: true
    },
    plot_bgcolor: 'white',
    paper_bgcolor: 'white',
    margin: { t: 30, b: 30, l: 80, r: 30 }
};

// Plot the comparison chart using Plotly
Plotly.newPlot('comparisonPlot', [trace], layout);
}



function displayFirstPlot(data, plotId) {
    const dates = data.outputs.map(output => output.date);
    const yields = data.outputs.map(output => output.yield);
    const growths = data.outputs.map(output => output.growth);
    const growthTrace = {
        x: dates,
        y: growths,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Growthrate',
        line: {
            color: 'rgb(3,98,76)',
            width: 2
        },
        marker: {
            color: 'rgb(3,98,76)',
            size: 8
        }
    };
    const yieldTrace = {
        x: dates,
        y: yields,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Yield',
        yaxis: 'y2',
        line: {
            color: 'rgb(126,185,48)',
            width: 2
        },
        marker: {
            color: 'rgb(126,185,48)',
            size: 8
        }
    };
    const layout1 = {
        title: 'Growthrate and Yield Over Time',
        xaxis: {
            title: 'Date',
            showgrid: true, // Show grid lines
            zeroline: false, // Remove the zero line
            showline: true, // Show the line at axis base
            mirror: 'allticks', // Mirror the tick lines on all sides
            linewidth: 2, // Width of the axis line
            linecolor: 'black' // Color of the axis line
        },
        yaxis: {
            title: 'Growthrate (g/TU)',
            showgrid: true,
            zeroline: false,
            showline: true,
            mirror: 'allticks',
            linewidth: 2,
            linecolor: 'black'
        },
        yaxis2: {
            title: 'Yield (g)',
            overlaying: 'y',
            side: 'right',
            showgrid: true,
            zeroline: false,
            showline: true,
            mirror: 'allticks',
            linewidth: 2,
            linecolor: 'black'
        },
        plot_bgcolor: 'white', // Set the background color to white
        margin: {t: 40, r: 40, b: 40, l: 40}, // Adjust margin to ensure all elements fit
        paper_bgcolor: 'white' // Set the paper background color to white
    };
    Plotly.newPlot(plotId, [growthTrace, yieldTrace], layout1);
}

function displaySecondPlot(data, plotId, plotType) {
    const dates = data.outputs.map(output => output.date);
    const map = data.outputs.map(output => output.map);
    const area = map[0].length * map[0][0].length;
    let traces = [];
    let layout;
    const titles = {
        "profit_per_plant": "Profit per Plant (€/Plant)",
        "profit_per_area": "Profit per Area (€/m²)",
        "yield_per_plant": "Yield per Plant (g/Plant)",
        "growth_per_plant": "Growth per Plant (g/TU per Plant)",
        "yield_per_area": "Yield per Area (g/m²)",
        "growth_per_area": "Growth per Area (g/TU per m²)",
        "number_of_plants": "Number of Plants",
        "growth": "Growth (g/TU)",
        "yield": "Yield (g)",
        "profit": "Profit (€)",
        "temperature": "Temperature (°C)",
        "water": "Water (ml)",
        "overlap": "Overlaping (cm²)",
        "rain": "Rain (mm)",
        "time_needed": "Time Needed per plant and TU (s/plant per TU)"
    };

    // Determine plot title and y-axis title
    let plotTitle = titles[plotType] || `${plotType.charAt(0).toUpperCase() + plotType.slice(1).replace('_', ' ')}`;
    let yAxisTitle = titles[plotType] || `${plotType.charAt(0).toUpperCase() + plotType.slice(1).replace('_', ' ')}`;

    // Define the trace based on plot type
    let yData = data.outputs.map(output => {
        switch (plotType) {
            case "profit_per_plant":
                return output.profit / output.num_plants;
            case "profit_per_area":
                return output.profit / area*10000;
            case "yield_per_plant":
                return output.yield / output.num_plants;
            case "growth_per_plant":
                return output.growth / output.num_plants;
            case "yield_per_area":
                return output.yield / area*10000;
            case "growth_per_area":
                return output.growth / area*10000;
            case "number_of_plants":
                return output.num_plants;
            case "growth":
                return output.growth;
            default:
                return output[plotType];
        }
    });

    traces.push({
        x: dates,
        y: yData,
        type: 'scatter',
        mode: 'lines+markers',
        name: plotTitle,
        line: { color: 'rgb(126,185,48)' }
    });

    layout = {
        title: plotTitle + " Over Time",
        xaxis: {
            title: 'Date',
            showgrid: true,
            zeroline: false,
            showline: true,
            mirror: 'allticks',
            linewidth: 2,
            linecolor: 'black'
        },
        yaxis: {
            title: yAxisTitle,
            showgrid: true,
            zeroline: false,
            showline: true,
            mirror: 'allticks',
            linewidth: 2,
            linecolor: 'black'
        },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        margin: { t: 40, r: 40, b: 40, l: 40 }
    };

    // Clear previous plot
    Plotly.react(plotId, traces, layout);
}
function displayHeatmap(data, plotId, inputData) {
    const heatmapData = data.outputs.map(output => output.map);
    const weed = data.outputs.map(output => output.weed);
    const dates = data.outputs.map(output => output.date);
    const slider = document.getElementById('dateSlider');
    const sliderValueDisplay = document.getElementById('sliderValue');

    // Calculate strip data with columns as the primary index
    const fieldLength = heatmapData[0].length; // Assuming each map row has the same length
    const fieldWidth = heatmapData[0][0].length; // Assuming each map column has the same width
    const stripData = Array(fieldLength).fill(null).map(() => Array(fieldWidth).fill(0)); // Initialize with zeros

    let rowStart = 0;
    if (inputData)  {
        inputData.rows.forEach((row, stripIndex) => {
            const rowWidth = row.stripWidth;
            for (let x = rowStart; x < rowStart + rowWidth; x++) {
                for (let y = 0; y < fieldLength; y++) {
                    stripData[y][x] = stripIndex; // Assign stripIndex for each column within this row width
                }
            }
            rowStart += rowWidth;
        });
    }

    // Initialize slider
    slider.max = heatmapData.length - 1;

    // Event Listener for slider movement
    slider.addEventListener('input', function() {
        const sliderValue = slider.value;
        //show the current date on the slider
        sliderValueDisplay.textContent = dates[sliderValue];
        Heatmap(sliderValue, heatmapData, weed, dates);
    });

    const showStrips = document.getElementById('showStrips');
    const selectedOption = document.getElementById('heatmapOption');
    showStrips.addEventListener('change', function() {
        Heatmap(slider.value, heatmapData, weed, dates);
    });
    selectedOption.addEventListener('change', function() {
        Heatmap(slider.value, heatmapData, weed, dates);
    });

    // Function to display the corresponding Heatmap data
    function Heatmap(index, map, weed, dates) {
        const mapData = map[index];
        const weedData = weed[index];
        const showStrips = document.getElementById('showStrips').checked;
        const selectedOption = document.getElementById('heatmapOption').value;

        if (mapData) {
            // Heatmap data trace
            const heatmapTrace = {
                z: mapData,
                type: 'heatmap',
                colorscale: [
                    [0, 'rgb(100,50,0)'],  
                    [0.01, 'rgb(126,185,48)'],
                    [1, 'rgb(0,100,0)']
                ],
                colorbar: {
                    title: 'Plant Size',
                    titleside: 'right'
                }
            };


            const weedTrace = {
                z: weedData,
                type: 'heatmap',
                colorscale: [
                    [0, 'rgba(100,50,0,0)'],
                    [0.01, 'rgba(255,165,0,0.5)'],
                    [1, 'rgba(255,255,0,1)']
                ],
                colorbar: {
                    title: 'Weed Size',
                    titleside: 'left',
                    x: -0.15
                },
                opacity: 0.5
            };


            const stripTrace = {
                z: stripData,
                type: 'heatmap',
                colorscale: [[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0.2)']],
                showscale: false
            };

            const layout = {
                title: `Heatmap on ${dates[index]}`,
                xaxis: {
                    title: 'Width',
                    showgrid: false,
                    zeroline: false,
                    showline: true,
                    mirror: 'allticks',
                    linewidth: 2,
                    linecolor: 'black'
                },
                yaxis: {
                    title: 'Length',
                    showgrid: false,
                    zeroline: false,
                    showline: true,
                    mirror: 'allticks',
                    linewidth: 2,
                    linecolor: 'black'
                },
                margin: { t: 40, r: 20, b: 40, l: 50 },
                plot_bgcolor: 'white',
                paper_bgcolor: 'white'
            };

            let traces = [];
            if (selectedOption === 'plants') {
                traces.push(heatmapTrace);
            } else if (selectedOption === 'weeds') {
                traces.push(weedTrace);
            } else if (selectedOption === 'plantsweeds') {
                traces.push(heatmapTrace, weedTrace);
            }

            if (showStrips) {
                traces.push(stripTrace);
            }

            Plotly.newPlot(plotId, traces, layout);
        } else {
            
        }
    }

    // Display the initial heatmap
    Heatmap(0, heatmapData, weed, dates);
}