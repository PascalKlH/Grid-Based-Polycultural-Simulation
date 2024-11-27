from django.shortcuts import render
from django.http import JsonResponse, HttpResponseBadRequest
from .scripts.calculate import main  
from .scripts.add_initial_data_to_db import add_initial_weather_data_to_db, add_initial_plant_data_to_db
import json
from .models import  DataModelOutput, SimulationIteration
from .models import Weather, Plant,DataModelInput
from django.shortcuts import get_object_or_404
from .forms import PlantForm
from django.http import StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from .forms import SimulationForm
from django.core.serializers import serialize
from django.template.loader import render_to_string
from django.core.serializers.json import DjangoJSONEncoder


def index(request):
    return accordion_view(request)

def get_simulation_data(request):
    """
    Streams simulation data for a specified simulation in NDJSON format.

    Retrieves and streams detailed simulation data for each iteration of a given simulation name,
    which is specified via a 'name' query parameter. The data includes metrics such as yield, growth,
    and weather conditions for each iteration.

    Parameters:
        request (HttpRequest): The request object containing GET data including the 'name'
                               of the simulation to retrieve data for.

    Returns:
        StreamingHttpResponse: Streams the data in NDJSON format if the simulation name is provided.
        JsonResponse: Returns an error as JSON if the simulation name is not provided or is invalid.

    Raises:
        JsonResponse: If the 'name' query parameter is missing or empty, returns a 400 error.
    """
    simulation_name = request.GET.get('name')
    if not simulation_name:
        return JsonResponse({'error': 'No simulation name provided'}, status=400)


    def data_generator():
        """
        Generator function to yield simulation data incrementally.

        Fetches data from the database and yields one simulation iteration at a time
        in JSON format. This approach helps manage memory usage and response times
        effectively when streaming large datasets.
        """
        iterations = SimulationIteration.objects.filter(input__simName=simulation_name).order_by('iteration_index')
        for iteration in iterations:
            iteration_data = {
                "iteration_index": iteration.iteration_index,
                "param_value": iteration.param_value,
                "outputs": [
                    {
                        "date": output.date,
                        "yield": output.yield_value,
                        "growth": output.growth,
                        "water": output.water,
                        "overlap": output.overlap,
                        "map": output.map,
                        "weed": output.weed,
                        "time_needed": output.time_needed,
                        "profit": output.profit,
                        "rain": output.rain,
                        "temperature": output.temperature,
                        "num_plants": output.num_plants,
                    }
                    for output in DataModelOutput.objects.filter(iteration=iteration).order_by('date')
                ]
            }
            yield json.dumps(iteration_data) + '\n'

    response = StreamingHttpResponse(data_generator(), content_type="application/x-ndjson")
    return response



def manage_plants(request):
    """
    Manages CRUD operations for plants. Allows users to create, update, and delete plants,
    excluding any plant named 'weed'. Responds to both GET and POST requests.

    GET:
        Renders a form for creating or updating plants.
    POST:
        Handles the submission of the plant form and deletion requests.
    """
    # Filter out plants named 'weed' from the beginning
    plants = Plant.objects.exclude(name="weed")
    selected_plant = None

    # Retrieve plant_id from POST or GET request
    plant_id = request.POST.get('plant_id') or request.GET.get('plant_id')
    if plant_id:
        selected_plant = get_object_or_404(Plant, id=plant_id)

    if request.method == 'POST':
        form = PlantForm(request.POST, instance=selected_plant)

        if 'delete' in request.POST and selected_plant:
            plant_id = selected_plant.id
            selected_plant.delete()
            return JsonResponse({
                'success': True, 'message': 'Plant deleted successfully.',
                'deletedPlantId': plant_id
            })

        if form.is_valid():
            plant = form.save()
            print(f"Plant saved: {plant.name}")
            return JsonResponse({
                'success': True, 'message': 'Plant saved successfully.',
                'plant': {'id': plant.id, 'name': plant.name}
            })
        else:
            print(form.errors.get_json_data())
            return JsonResponse({'success': False, 'errors': form.errors.get_json_data()}, status=400)

    else:
        # Prepare form for GET request
        form = PlantForm(instance=selected_plant)

    return render(request, 'manage_plants.html', {
        'plants': plants,
        'form': form,
        'selected_plant': selected_plant
    })


def get_plant_details(request, plant_id):
    plant = get_object_or_404(Plant, id=plant_id)
    plant_data = {
        'id': plant.id,
        'name': plant.name,
        'W_max': plant.W_max,
        'k': plant.k,
        'n': plant.n,
        'b': plant.b,
        'max_moves': plant.max_moves,
        'Yield': plant.Yield,
        'planting_cost': plant.planting_cost,
        'revenue': plant.revenue
    }
    return JsonResponse(plant_data)

@csrf_exempt
def accordion_view(request):
    """
    Serves as a central view for rendering various parts of the simulation control panel using an accordion UI.
    
    This view dynamically loads different parts of the UI such as managing plants, running simulations,
    viewing simulation results, and generating plots. The content is loaded into an accordion structure
    in the frontend for user interaction.

    Args:
        request: HttpRequest object containing metadata about the request.

    Returns:
        HttpResponse: Renders the accordion template populated with dynamically generated content.
    """
    # Check if the plant table is empty and populate it with initial data if necessary
    if Plant.objects.count() == 0:
        print("Adding initial plant data to the database.")
        add_initial_plant_data_to_db()

    # Retrieve all plants and the selected plant if a plant_id is provided
    plants = Plant.objects.all()
    plant_id = request.GET.get('plant_id')
    selected_plant = get_object_or_404(Plant, id=plant_id) if plant_id else None

    # Initialize forms with unique prefixes to avoid name collisions
    form = PlantForm(instance=selected_plant)
    simulation_form = SimulationForm(prefix="sim")
    plants_json = serialize('json', plants)  # Serialize plants data for use in the frontend
    previous_simulations = json.dumps(list(DataModelInput.objects.all().values_list('simName', flat=True)), cls=DjangoJSONEncoder)

    # Prepare content for each section of the accordion
    content_list = [
        {
            'name': 'Manage Plants',
            'content': render_to_string('manage_plants.html', {
                'plants': plants,
                'form': form,
                'selected_plant': selected_plant
            }, request=request)
        },
        {
            'name': 'Previous Simulations',
            'content': render_to_string('list_simulations.html', {
                'simulations': DataModelInput.objects.all()
            }, request=request)
        },
        {
            'name': 'Run Simulations',
            'content': render_to_string('run_simulation.html', {
                'simulation_form': simulation_form,
                'plants': plants_json,
                "previous_simulations": previous_simulations
            }, request=request)
        },
        {
            'name': 'Yield and Growth Plot',
            'content': render_to_string('first_plot.html', request=request)
        },
        {
            'name': 'Detailed Plots',
            'content': render_to_string('second_plot.html', request=request)
        },
        {
            'name': 'Map of the Field',
            'content': render_to_string('heatmap.html', request=request)
        },
        {
            'name': 'Iteration Comparison',
            'content': render_to_string('comparison.html', request=request)
        }
    ]

    # Render and return the accordion template with the prepared content list
    return render(request, 'accordion_template.html', {'content_list': content_list})


def list_simulations(request):
    simulations = DataModelInput.objects.all()  # Get all simulations

 # Check if a simulation should be deleted via AJAX
    if request.method == 'POST' and 'delete_simulation' in request.POST:
        sim_name = request.POST.get('delete_simulation')
        simulation = get_object_or_404(DataModelInput, simName=sim_name)
        simulation.delete()  # Delete simulation and related data via CASCADE
        return JsonResponse({'status': 'success', 'message': f'Simulation "{sim_name}" deleted successfully'})

    # Handle GET request to fetch data for plotting as NDJSON
    if request.method == 'GET' and 'plot_simulation' in request.GET:
        sim_name = request.GET.get('plot_simulation')
        simulation = get_object_or_404(DataModelInput, simName=sim_name)
        iterations = SimulationIteration.objects.filter(input=simulation)
        outputs = DataModelOutput.objects.filter(iteration__in=iterations)

        # Streaming response to return NDJSON
        def ndjson_stream():
            for iteration in iterations:
                iteration_data = {
                    'index': iteration.iteration_index,
                    'param_value': iteration.param_value,
                    'outputs': [output.get_data() for output in outputs.filter(iteration=iteration)]
                }
                yield json.dumps(iteration_data) + '\n'  # Each JSON object on a new line
        
        return StreamingHttpResponse(ndjson_stream(), content_type='application/x-ndjson')

    return render(request, 'list_simulations.html', {'simulations': simulations})

def run_simulation(request):
    if request.method == 'GET':
        # Return all simulation names for checking duplicates in the frontend
        simulation_names = list(DataModelInput.objects.values_list('simName', flat=True))
        return JsonResponse({'existing_simulation_names': simulation_names})

    if request.method == 'POST':
        simulation_name = request.POST.get('simName')
        if DataModelInput.objects.filter(simName=simulation_name).exists():
            return JsonResponse({'error': 'Simulation name already exists. Please choose a different name.'}, status=400)

        try:
            data = json.loads(request.body.decode('utf-8'))
            # Add initial data if needed
            if Weather.objects.count() == 0:
                add_initial_weather_data_to_db()

            main(data)  # Run the simulation with the provided data
            return JsonResponse({'name': data["simName"]})

        except json.JSONDecodeError:
            return HttpResponseBadRequest("Invalid JSON format")
    return JsonResponse({'error': 'POST request required'}, status=405)


