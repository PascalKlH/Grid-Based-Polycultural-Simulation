# Project Title

**Grid-Based Polycultural Simulation**

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)



## Introduction

This project simulates the growth of crops and the management of weeds in agricultural fields. It allows users to input parameters related to planting and manage the growth process through a structured simulation. The results can be analyzed and visualized using various data analysis techniques.

## Features

- Dynamic simulation of crop growth based on user-defined parameters.
- Support for multiple planting types (grid, alternating, random).
- Weed growth simulation and management.
- Data storage in SQLite for easy retrieval and analysis.
- Visualization of simulation results using Plotly.
- Integration with Django for web-based interface.

## Technologies Used

- Python
- Django
- SQLite
- NumPy
- Pandas
- Plotly
- JSON
- Bootstrap (for front-end styling)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/PascalKlH/BA-Pascal.git
   cd yourproject
2. **Create a virtual environment
   ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    
3. **Install dependencies:**
   ```bash
    pip install -r requirements.txt
4. **Navigate to the simproject folder :**
   
   drag the simulation folder into the indlude folder
5. **Navigate to the simproject folder:**
   ```bash
   cd simproject
6. **Run the development server:**
   ```bash
    python manage.py runserver


## Usage
Open your web browser and navigate to http://127.0.0.1:8000/.
Use the web interface to input the simulation parameters (e.g., start date, number of sets, step size).
Choose the planting configuration (e.g., plant types, strip widths) and run the simulation.
View the results in the plots below.
Configure Plots as needed.


## Project Structure

```csharp
NRSsimulation/
│
├── simproject/
│   ├── simapp/
│   │   ├── __pycache__/
│   │   ├── migrations/         # Django migrations folder
│   │   ├── management/
│   │   │   └── commands/
│   │   │       └── export_simulation_data.py  # Command to export simulation data
│   │   ├── scripts/
│   │   │   ├── __pycache__/
│   │   │   ├── calculate.py     # Calculation scripts for the simulation
│   │   │   ├── add_initial_weatherdata.py  # Script to add initial weather data
│   │   │   └── data/
│   │   │       └── transformed_weather_data.csv  # Weather data for simulation
│   │   ├── static/
│   │   │   ├── documents/
│   │   │   │   └── tutorial.pdf  # Tutorial document for users
│   │   │   ├── images/
│   │   │   │   └── favicon.ico   # Favicon for the web application
│   │   │   ├── simapp/
│   │   │   │   ├── list_simulations.js  # JavaScript for simulation listing
│   │   │   │   ├── manage_plants.js     # JavaScript for managing plants
│   │   │   │   ├── plotting.js         # JavaScript for data plotting
│   │   │   │   ├── run_simulation.js   # JavaScript to handle simulation runs
│   │   │   │   └── style.css           # Main stylesheet for the application
│   │   ├── templates/
│   │   │   ├── simapp/
│   │   │   │   ├── accordion_view.html  # View for accordion UI component
│   │   │   │   ├── comparison.html      # Template for comparison views
│   │   │   │   ├── first_plot.html      # First plot display template
│   │   │   │   ├── heatmap.html         # Heatmap display template
│   │   │   │   ├── list_simulations.html  # Template for listing simulations
│   │   │   │   ├── manage_plants.html     # Template for managing plants
│   │   │   │   ├── run_simulation.html    # Template to run simulations
│   │   │   │   └── second_plot.html       # Second plot display template
│   │   ├── __init__.py
│   │   ├── admin.py
│   │   ├── apps.py
│   │   ├── forms.py
│   │   ├── models.py
│   │   ├── tests.py
│   │   ├── urls.py
│   │   └── views.py
│   ├── __pycache__/
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── db.sqlite3                 # SQLite database file
├── manage.py                  # Django management script
├── README.md                  # Project README file
└── requirements.txt           # File with listed dependencies to install

