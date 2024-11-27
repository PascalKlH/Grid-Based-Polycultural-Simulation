import numpy as np
from threading import Lock
from datetime import datetime, timedelta
from scipy.ndimage import convolve
import random
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from simapp.models import Plant
from ..models import DataModelInput, DataModelOutput, SimulationIteration, RowDetail, Weather
import time
from dateutil import parser
from django.core.exceptions import ObjectDoesNotExist

class Crop:
    """
    A class to represent a crop.

    Attributes
    ----------
    name : str
        The name of the crop.
    center : tuple
        The (x, y) coordinates of the center of the crop.
    radius : int
        The radius of the crop.
    parameters : dict
        A dictionary of parameters related to the crop.
    sim : Simulation
        The simulation object that manages the growth and interactions of the crop.
    cells : np.ndarray
        An array representing the cells occupied by the crop.
    boundary : np.ndarray
        An array defining the boundary of the crop's area.
    moves : int
        The number of moves the crop has made during the simulation.
    overlap : float
        The amount of overlap with other plants.
    previous_growth : float
        The growth rate of the crop during the previous time step.

    Methods
    -------
    grow(size_layer, obj_layer, pos_layer)
        Grows the crop based on the provided size layer, object layer, and position layer.
    update_cells_and_boundary()
        Updates the cells and boundary of the crop based on its current center and radius.
    generate_circular_mask(radius)
        Generates a circular mask for the crop area with the specified radius.
    """


    def __init__(self, name, center, parameters, sim):
        """
        Initializes the Crop instance.

        Parameters
        ----------
        name : str
            The name of the crop.
        center : tuple
            The (x, y) coordinates of the center of the crop.
        radius : int
            The radius of the crop.
        parameters : dict
            A dictionary of parameters related to the crop.
        sim : Simulation
            The simulation object managing the crop.
        """
        self.name = name
        self.center = center
        self.radius = 0  # Initial radius is 0
        self.parameters = parameters
        self.sim = sim
        self.size =0
        self.cells = np.zeros(
            (
                self.parameters["W_max"] + self.parameters["max_moves"] + 2,
                self.parameters["W_max"] + self.parameters["max_moves"] + 2,
            ),
            dtype=bool,
        )
        self.boundary = np.zeros(
            (
                self.parameters["W_max"] + self.parameters["max_moves"] + 2,
                self.parameters["W_max"] + self.parameters["max_moves"] + 2,
            ),
            dtype=bool,
        )
        self.moves = 0  # Number of moves
        self.overlap = 1  # Overlap with other plants
        self.previous_growth = 0  # Growth rate of the previous hour


    def grow(self, size_layer, obj_layer, pos_layer, strip):
        """
        Grow the crop based on the size layer, object layer, and position layer.

        Parameters
        ----------
        size_layer : np.ndarray
            An array representing the size information for the growth process.
        obj_layer : np.ndarray
            An array containing information about objects that may affect growth.
        pos_layer : np.ndarray
            An array representing the position information for growth.
        """
        growthrate = self.calculate_growthrate(strip)
        self.add_growthrate_tp_plant(growthrate,size_layer)
        prevous_growth = growthrate.copy()
        self.radius = self.radius + growthrate
        if self.radius > self.parameters["W_max"]:
            self.radius = self.parameters["W_max"]
        rounded_radius = int(np.round(self.radius / 2))
        if rounded_radius != int(np.round(prevous_growth / 2)):
            self.check_overlap(rounded_radius,size_layer,obj_layer,pos_layer)
            self.update_cells()
            self.update_boundary()
        return growthrate, self.overlap
    

    def calculate_waterfactor(self):
        """
        Calculate the impact of water on plant growth.

        Returns
        -------
        float
            The water factor influencing growth.
        """
        if self.sim.input_data['useWater']:
            current_time = self.sim.current_date
            #get the current paticipitation if no data is available the paticipitation is set to 0
            current_paticipitation = self.sim.weather_data[current_time.hour][1] if current_time.hour in self.sim.weather_data else 0.5
            self.sim.water_layer += current_paticipitation* 0.0001
            self.sim.water_layer[self.sim.water_layer > 2] = 2

            optimal_water = 0.5
            return 1 - abs(current_paticipitation - optimal_water) / optimal_water
        else:
            return 1
        

    def calculate_tempfactor(self):
        """
        Calculate the impact of temperature on plant growth, considering temperature data
        from the current hour back to the number of hours specified by `stepsize`.

        Returns
        -------
        float
            The temperature factor influencing growth, where 1 indicates optimal conditions,
            and decreases as temperature deviates from optimal.
        """
        if not self.sim.input_data['useTemperature']:
            return 1  # If temperature usage is not enabled, return a neutral factor.

        current_time = self.sim.current_date
        optimal_temp = 20
        temperatures = []

        # Collect temperature data for the range from current hour back to the step size limit.
        for i in range(self.sim.stepsize):
            hour = (current_time.hour - i) % 24  # Adjust for hour roll-over (0-23)
            if hour in self.sim.weather_data:
                temperatures.append(self.sim.weather_data[hour][2])
            else:
                temperatures.append(optimal_temp)  # Default to optimal if no data is available.

        # Calculate the average temperature.
        average_temperature = sum(temperatures) / len(temperatures)

        # Calculate the temperature factor based on deviation from the optimal temperature.
        temp_factor = 1 - abs(average_temperature - optimal_temp) / optimal_temp

        return max(0, temp_factor)  # Ensure the factor does not go below 0.
    

    def calculate_growthrate(self, strip):
        """
        Calculate the growth rate of the plant based on various environmental and internal factors.

        Parameters
        ----------
        strip : Strip
            The plant strip containing sowing date and other relevant data.

        Returns
        -------
        float
            The calculated growth rate for the plant.
        """
        # Calculate the difference in hours since the sowing date
        current_time = self.sim.current_date

        t_diff_hours = ((current_time - strip.sowing_date).total_seconds() / 3600)
        t_diff_hours = t_diff_hours
        # Calculate environmental impact factors on growth
        water_factor = self.calculate_waterfactor()

        temp_factor = self.calculate_tempfactor()

        # Random factor to introduce natural variation in growth
        random_factor = random.uniform(1, 1.001)

        #diese werte fur den Versuch
        h = self.parameters["W_max"]
        r = self.parameters["k"]
        m = self.parameters["n"]
        x = t_diff_hours
        b= self.parameters["b"]/self.sim.stepsize
        growth_rate=(h*b*r)/(m-1)*np.exp(-r*x)*(1+b*np.exp(-r*x))**(m/(1-m))

        #get the betrag of the growth rate
        growth_rate = abs(growth_rate*self.sim.stepsize*self.overlap*water_factor*temp_factor*random_factor)
        # Update previous growth and modify the water layer based on the new growth
        self.previous_growth = growth_rate
        self.sim.water_layer[self.center] -= 0.1 * growth_rate

        return growth_rate
        # If the radius is the same as before, we can simply add the growth rate to the circular mask


    def add_growthrate_tp_plant(self,growth_rate,size_layer):
        """
        Adds the given growth rate to the plant area within the size layer.

        The function calculates a circular area (mask) based on the radius of the plant and adds the growth rate value to the corresponding cells within the provided `size_layer`. The circular area is centered on `self.center`, and any growth rate value that falls within the mask boundary is added to the `size_layer` using NumPy's `add.at` function.

        Parameters:
        -----------
        growth_rate : float
            The growth rate to be added to the plant area.
        size_layer : numpy.ndarray
            The 2D array representing the field, where the growth rate will be added.

        Returns:
        --------
        None
        The function modifies the `size_layer` in place.
        """
        rounded_radius = int(np.round(self.radius / 2))
        mask = self.generate_circular_mask(rounded_radius)
        crop_mask = np.zeros_like(size_layer, dtype=bool)

        # Check if the mask is within the boundaries of the field
        r_start = int(max(self.center[0] - rounded_radius, 0))
        r_end = int(min(self.center[0] + rounded_radius + 1, size_layer.shape[0]))
        c_start = int(max(self.center[1] - rounded_radius, 0))
        c_end = int(min(self.center[1] + rounded_radius + 1, size_layer.shape[1]))

        # Check if the mask is within the boundaries of the field
        mask_r_start = int(r_start - (self.center[0] - rounded_radius))
        mask_r_end = int(mask_r_start + (r_end - r_start))
        mask_c_start = int(c_start - (self.center[1] - rounded_radius))
        mask_c_end = int(mask_c_start + (c_end - c_start))

        # Add the mask to the crop mask
        crop_mask[r_start:r_end, c_start:c_end] = mask[
            mask_r_start:mask_r_end, mask_c_start:mask_c_end
        ]
        np.add.at(size_layer, np.where(crop_mask), growth_rate)

    def check_overlap(self, rounded_radius, size_layer, obj_layer, pos_layer):
        """
        Checks for overlap of a plant within a given area based on its size and position.

        Parameters:
            rounded_radius (int): The rounded radius of the plant's influence area.
            size_layer (np.array): The layer representing the sizes of objects in the grid.
            obj_layer (np.array): The layer containing object identifiers.
            pos_layer (np.array): The layer containing position information.

        Returns:
            None: Modifies the overlap attribute of the object based on detected overlap.
        """
        # Calculate the boundary indices for the area around the crop
        r_min = max(self.center[0] - rounded_radius - 1, 0)
        r_max = min(self.center[0] + rounded_radius + 2, size_layer.shape[0])
        c_min = max(self.center[1] - rounded_radius - 1, 0)
        c_max = min(self.center[1] + rounded_radius + 2, size_layer.shape[1])

        # Create a slice of the size_layer to check for overlap
        snipped_size_layer = size_layer[r_min:r_max, c_min:c_max]
        # Convert all non-zero cells to 1 to create a mask
        mask = np.where(snipped_size_layer > 0, 1, 0)

        # Calculate the center of the plant's own cell matrix
        center_row = self.cells.shape[0] // 2
        center_col = self.cells.shape[1] // 2

        # Define the boundary around the center based on the radius
        r_min = max(center_row - rounded_radius - 1, 0)
        r_max = min(center_row + rounded_radius + 2, self.cells.shape[0])
        c_min = max(center_col - rounded_radius - 1, 0)
        c_max = min(center_col + rounded_radius + 2, self.cells.shape[1])

        # Extract the relevant section of the plant's cell matrix
        cells_slice = self.cells[r_min:r_max, c_min:c_max]
        boundary_slice = self.boundary[r_min:r_max, c_min:c_max]

        # Adjust the mask size to fit the cells_slice if necessary
        if mask.shape != cells_slice.shape:
            # Ensure mask and cells_slice are the same size to avoid index errors
            if mask.shape[0] > cells_slice.shape[0] or mask.shape[1] > cells_slice.shape[1]:
                mask = mask[:cells_slice.shape[0], :cells_slice.shape[1]]
            else:
                cells_slice = cells_slice[:mask.shape[0], :mask.shape[1]]
                boundary_slice = boundary_slice[:mask.shape[0], :mask.shape[1]]

        # Subtract the current plant's cells from the mask to avoid counting its own cells as overlap
        mask -= cells_slice
        mask[mask < 0] = 0

        # Add boundary values to the mask for overlap calculation
        mask += boundary_slice

        # Check for any overlap with other plants by checking values greater than 1
        if np.any(mask > 1):
            # Calculate total and relative overlap
            total_overlap = np.sum(mask > 1)
            relative_overlap = total_overlap / np.sum(self.boundary) if np.sum(self.boundary) > 0 else 0
            # Define a maximum acceptable overlap percentage
            maxoverlap = 0.07
            # Update the overlap attribute, inversely proportional to the relative overlap
            self.overlap = max(0, min(1, 1 - (relative_overlap / maxoverlap)))
            # Move plant if possible and if needed (logic to be implemented in move_plant)
            if self.moves < self.parameters["max_moves"]:
                self.move_plant(mask, size_layer, obj_layer, pos_layer)
        else:
            # Reset overlap if there's no interference with other plants
            self.overlap = 1

    def move_plant(self, mask, size_layer, obj_layer, pos_layer):
        
        """
        Moves the plant to reduce overlap based on the calculated interference mask.

        Parameters:
            mask (np.array): A mask indicating the areas of overlap.
            size_layer (np.array): The layer representing the sizes of objects in the grid.
            obj_layer (np.array): The layer containing object identifiers.
            pos_layer (np.array): The layer containing position information.

        Returns:
            None: Modifies the center, obj_layer, and pos_layer to reflect the new position.
        """
        # Find indices where the interference is greater than 1
        interference_indices = np.where(mask > 1)
        # Stack these indices into x, y coordinates
        interference_points = np.column_stack(interference_indices)
        # Get the current center of the plant
        center_point = np.array(self.center)

        # Return if there is no interference
        if interference_points.size == 0:
            return

        # Initialize a zero vector for the movement direction
        total_vector = np.zeros(2)

        # Accumulate all repulsion vectors from points of interference
        for point in interference_points:
            direction_vector = center_point - point
            distance = np.linalg.norm(direction_vector)
            if distance > 0:  # Avoid division by zero
                normalized_vector = direction_vector / distance
                # Weight the vector inversely by distance
                weighted_vector = normalized_vector / distance
                total_vector += weighted_vector

        # Normalize the total movement vector to ensure a uniform step size
        norm = np.linalg.norm(total_vector)
        if norm == 0:
            return

        # Calculate movement vector and ensure it is integer for grid movement
        movement_vector = np.round(total_vector / norm).astype(int)
        new_center_x, new_center_y = center_point + movement_vector

        # Ensure the new position is within grid bounds before moving
        if (
            0 <= new_center_x < size_layer.shape[0]
            and 0 <= new_center_y < size_layer.shape[1] 
        ):
            new_center_x, new_center_y = self.center
            # Update the plant's center
            center_x, center_y = self.center
            self.center = (new_center_x, new_center_y)

            # Clear the old position and update the new position in pos_layer
            pos_layer[center_x, center_y] = False
            pos_layer[new_center_x, new_center_y] = True

            # Move the plant object in the object layer
            obj_layer[center_x, center_y] = None
            obj_layer[new_center_x, new_center_y] = self

            # Increment the move counter for this plant
            self.moves += 1

        # If movement is not possible, simply return without updating
        else:
            return



    def update_cells(self):
        """
        Updates the cells array for the crop based on its current center and radius. This method 
        ensures the plant's influence area is accurately represented in the grid.
        """
        # Calculate the rounded radius and generate the corresponding circular mask
        rounded_radius = int(np.round(self.radius / 2))
        mask = self.generate_circular_mask(rounded_radius)
        
        # Reset the cells within the calculated boundary based on the center
        center_r = self.cells.shape[0] // 2
        center_c = self.cells.shape[1] // 2
        r_start = max(center_r - mask.shape[0] // 2, 0)
        r_end = min(r_start + mask.shape[0], self.cells.shape[0])
        c_start = max(center_c - mask.shape[1] // 2, 0)
        c_end = min(c_start + mask.shape[1], self.cells.shape[1])
        
        # Adjust the mask if the computed bounds are out of the array range
        mask_r_start = 0 if r_start >= 0 else -r_start
        mask_r_end = mask.shape[0] if r_end <= self.cells.shape[0] else mask.shape[0] - (r_end - self.cells.shape[0])
        mask_c_start = 0 if c_start >= 0 else -c_start
        mask_c_end = mask.shape[1] if c_end <= self.cells.shape[1] else mask.shape[1] - (c_end - self.cells.shape[1])
        
        # Apply the mask to the cells array
        self.cells[r_start:r_end, c_start:c_end] = mask[mask_r_start:mask_r_end, mask_c_start:mask_c_end]


    def update_boundary(self):
        """
        Updates the boundary array for the crop based on its current cells state. This method uses
        convolution to determine the boundary area where the plant influences its neighbors.
        """
        # Determine the rounded radius for boundary calculations
        rounded_radius = int(np.round(self.radius / 2))
        
        # Perform convolution to find the boundary using a cross-shaped kernel
        self.boundary = convolve(
            self.cells,
            np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
            mode="constant",
            cval=0.0
        ) ^ self.cells  # XOR with cells to get boundary cells only
        
        # Calculate the bounds for the boundary slice
        r_min = int(max(self.center[0] - rounded_radius - 1, 0))
        r_max = int(min(self.center[0] + rounded_radius + 2, self.sim.boundary_layer.shape[0]))
        c_min = int(max(self.center[1] - rounded_radius - 1, 0))
        c_max = int(min(self.center[1] + rounded_radius + 2, self.sim.boundary_layer.shape[1]))
        
        # Ensure the indices are within the grid boundaries
        if r_min == 0 or r_max == self.sim.boundary_layer.shape[0] or c_min == 0 or c_max == self.sim.boundary_layer.shape[1]:
            return  # Exit if the boundary is at the edge of the grid

        # Define the slice indices for the boundary
        start_index = max(self.parameters["W_max"] // 2 - rounded_radius - 1, 0)
        end_index = min(self.parameters["W_max"] // 2 + rounded_radius + 2, self.parameters["W_max"])
        
        # Create a slice from the boundary array
        new_boundary_slice = self.boundary[start_index:end_index, start_index:end_index]
        boundary = self.sim.boundary_layer[r_min:r_max, c_min:c_max]
        
        # Resize if necessary and add the new boundary slice
        if boundary.shape > new_boundary_slice.shape:
            boundary = boundary[:new_boundary_slice.shape[0], :new_boundary_slice.shape[1]]
        boundary += new_boundary_slice.astype(int)
    


    @staticmethod
    def generate_circular_mask(radius):
        """
        Generate a circular mask with the given radius. The mask is a boolean
        array that is True within the circle defined by the given radius and
        False outside.

        Parameters
        ----------
        radius : int
            The radius of the circle for which to create the mask.

        Returns
        -------
        np.ndarray
            A boolean array with shape (2*radius+1, 2*radius+1) representing the circular mask.

        Example
        -------
        >>> radius = 3
        >>> mask = Strip.generate_circular_mask(radius)
        >>> print(mask.astype(int))
        [[0 0 0 1 0 0 0]
         [0 0 1 1 1 0 0]
         [0 1 1 1 1 1 0]
         [1 1 1 1 1 1 1]
         [0 1 1 1 1 1 0]
         [0 0 1 1 1 0 0]
         [0 0 0 1 0 0 0]]
        """
        y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        mask = x**2 + y**2 <= radius**2
        return mask



class Strip:
    """
    Represents a strip in a simulation environment where crops are planted, managed, and harvested.

    Attributes:
        num_sets (int): Number of planting-harvesting cycles this strip will support.
        current_set (int): Current cycle number.
        index (int): Position index of the strip within the simulation grid.
        width (int): The width of the strip.
        plantType (str): Type of plant being grown in the strip.
        plantingType (str): The planting methodology used.
        harvesttype (str): The type of harvest strategy employed.
        rowSpacing (int): The space between rows within the strip.
        start (int): The starting position of the strip in the simulation space.
        num_plants (int): Number of plants currently in the strip.
        sowing_date (date): The date on which planting was done.
        sim (Simulation): Reference to the simulation instance managing this strip.
        previous_sizes (list): Tracks the last 5 size measurements for stability checks.
        plant_parameters (dict): Parameters specific to the plant type grown in the strip.

    Methods:
        __init__: Initializes a new instance of the Strip class.
    """

    def __init__(
        self,
        strip_width,
        plantType,
        plantingType,
        rowSpacing,
        sim,
        index,
        num_sets,
        harvesttype,
        
    ):
        """
        Initializes a new Strip with specified attributes and planting details.

        Parameters:
            strip_width (int): The width of the strip in units.
            plantType (str): Type of plant to grow.
            plantingType (str): The methodology for planting (e.g., 'dense', 'sparse').
            rowSpacing (int): Distance between rows within the strip in units.
            sim (Simulation): The simulation environment this strip is part of.
            index (int): The index position of the strip within the simulation.
            harvesttype (str, optional): The harvesting strategy to apply. Defaults to 'max_Yield'.
            num_sets (int, optional): The number of sets or cycles the strip will be used for. Defaults to 1.
        """
        self.num_sets = num_sets
        self.current_set = 1
        self.index = index
        self.width = strip_width
        self.plantType = plantType
        self.plantingType = plantingType
        self.harvesttype = harvesttype
        self.rowSpacing = rowSpacing
        self.start = sum(
            [sim.input_data["rows"][i]["stripWidth"] for i in range(index)]
        )  # Calculate the starting position based on preceding strips
        self.num_plants = 0  # Will be calculated during planting
        self.sowing_date = sim.current_date
        self.sim = sim
        # Initialize previous sizes for growth stability checks
        self.previous_sizes = [None] * 5
        # Retrieve plant parameters from a class method
        self.plant_parameters = Strip.get_plant_parameters(self.plantType)

    def get_plant_parameters(plant_name):
        """
        Retrieves the parameters for a specified plant type from the database.

        This function queries the database for a Plant object by name and extracts its attributes
        to construct a dictionary of plant parameters. If the plant is not found, it handles the
        exception and returns None.

        Parameters:
            plant_name (str): The name of the plant whose parameters are to be retrieved.

        Returns:
            dict or None: A dictionary containing the plant's parameters if found, otherwise None.
        """
        
        try:
            # Attempt to retrieve the plant by name from the Plant model
            plant = Plant.objects.get(name=plant_name)

            # Return a dictionary of the plant's parameters if found
            return {
                "name": str(plant.name),
                "W_max": int(plant.W_max),
                "k": float(plant.k),
                "n": float(plant.n),
                "b": int(plant.b),
                "max_moves": int(plant.max_moves),
                "Yield": float(plant.Yield),
                "planting_cost": float(plant.planting_cost),
                "revenue": float(plant.revenue),
            }
        except Plant.DoesNotExist:
            # Handle the case where the plant does not exist in the database
            print(f"Plant {plant_name} not found in the database.")
            return None

    def planting(self, sim):
        """
        Executes the planting process for a strip based on the specified planting type.

        This method determines the planting strategy based on the `plantingType` attribute and
        delegates the planting operation to the respective private method handling each specific
        strategy. It updates the sowing date to the current date from the simulation context.

        Parameters:
            sim (Simulation): The simulation object containing global settings and current state,
                            including the current date and input data for planting configurations.

        Returns:
            None: This method updates the state of the strip but does not return any value.
        """

        # Define common parameters for all planting strategies
        strip_parameters = {
            "rowLength": sim.input_data["rowLength"],
            "rowDistance": self.rowSpacing,
            "columnDistance": self.width,
        }

        # Select and call the planting method based on the planting type
        if self.plantingType == "grid":
            self._grid_planting(
                strip_parameters,
                self.start,
                self.start + self.width,
                self.plant_parameters,
                sim,
            )
        elif self.plantingType == "alternating":
            self._alternating_planting(
                strip_parameters,
                self.start,
                self.start + self.width,
                self.plant_parameters,
                sim,
            )
        elif self.plantingType == "random":
            self._random_planting(
                strip_parameters,
                self.start,
                self.start + self.width,
                self.plant_parameters,
                sim,
            )
        else:
            # Handle undefined planting types with an empty/default method
            self._empty_planting()

        # Update the sowing date to the current date from the simulation
        self.sowing_date = sim.current_date

    def apply_planting(self, row_indices, col_indices, plant_parameters, sim):
        """
        Plants crops in specified rows and columns within the simulation's grid.

        This method uses numpy's meshgrid to generate matrices of row and column indices
        for planting, updates the simulation's layers for crop positions and objects, and initializes
        the crop sizes. It also calculates and updates the number of plants successfully planted.

        Parameters:
            row_indices (list or array): Indices of rows where crops will be planted.
            col_indices (list or array): Indices of columns where crops will be planted.
            plant_parameters (dict): A dictionary containing parameters specific to the plant type being planted.
            sim (Simulation): The simulation object which holds various layers (like crop position and size).

        Returns:
            tuple: A tuple containing two arrays (row_grid, col_grid) representing the grid positions where crops have been planted.
        """

        # Create grid indices for planting using numpy's meshgrid
        # meshgrid is used here to ensure that the planting grid coordinates align with simulation layers
        col_grid, row_grid = np.meshgrid(col_indices, row_indices, indexing="ij")

        # Update the crop position layer to reflect the newly planted crops
        sim.crops_pos_layer[row_grid, col_grid] = True

        # Vectorize the creation of crop objects for efficient batch processing across the grid
        create_crop = np.vectorize(lambda r, c: Crop(self.plantType, (r, c), plant_parameters, sim))
        crops = create_crop(row_grid, col_grid)

        # Place the newly created crop objects in the simulation's crop object layer
        sim.crops_obj_layer[row_grid, col_grid] = crops

        # Initialize the size of each crop in the crop size layer
        sim.crop_size_layer[row_grid, col_grid] = 0.001

        # Calculate and update the number of plants that have been planted
        self.num_plants = np.size(row_grid)  # np.size returns the total number of elements in the grid

        return row_grid, col_grid


    def _grid_planting(self, strip_parameters, start_col, end_col, plant_parameters, sim):
        """
        Implements a grid-based planting strategy within the specified column boundaries
        of a strip in the simulation.

        This method calculates the row and column indices where crops should be planted in a grid pattern
        based on the provided spacing parameters and applies the planting using the `apply_planting` method.

        Parameters:
            strip_parameters (dict): Dictionary containing parameters like row length and distance between rows.
            start_col (int): The starting column index for the strip where planting begins.
            end_col (int): The ending column index for the strip where planting ends.
            plant_parameters (dict): A dictionary of parameters specific to the plant being planted.
            sim (Simulation): The simulation object that manages the state and layers of the simulation.

        Returns:
            None: This method directly modifies the simulation's state and does not return any value.
        """
        # Retrieve the distance between plants and the length of the row from strip parameters
        plant_distance = strip_parameters["rowDistance"]  # Space between plants
        row_length = strip_parameters["rowLength"]  # Length of the row

        # Calculate half of the plant distance to create an offset for planting
        # This offset prevents planting directly on the edge of the planting area
        offset = plant_distance // 2

        # Adjust the row and column indices to ensure plants are centered and not planted on the edges
        row_start = offset
        row_end = row_length - offset
        col_start = start_col + offset
        col_end = end_col - offset

        # Generate grid indices within the adjusted bounds for rows and columns
        row_indices = np.arange(row_start, row_end, plant_distance)
        col_indices = np.arange(col_start, col_end, plant_distance)

        # Apply the planting using the calculated grid indices
        self.apply_planting(row_indices, col_indices, plant_parameters, sim)



    def _empty_planting(self):
        pass

    def _alternating_planting(self, strip_parameters, start_col, end_col, plant_parameters, sim):
        """
        Plants crops in an alternating row pattern within the specified column boundaries
        of a strip in the simulation. Alternating rows have a shifted planting position to
        ensure staggered growth and distribution.

        Parameters:
            strip_parameters (dict): Dictionary containing parameters like row length and distance between rows.
            start_col (int): The starting column index for the strip where planting begins.
            end_col (int): The ending column index for the strip where planting ends.
            plant_parameters (dict): A dictionary of parameters specific to the plant being planted.
            sim (Simulation): The simulation object that manages the state and layers of the simulation.

        Returns:
            None: This method directly modifies the simulation's state and does not return any value.
        """
        # Retrieve the distance between plants and the length of the row from strip parameters
        plant_distance = strip_parameters["rowDistance"]  # Space between plants vertically
        row_length = strip_parameters["rowLength"]  # Horizontal length of each row

        # Calculate half the plant distance to create an offset for planting
        # This offset prevents planting directly on the edge of the planting area
        offset = plant_distance // 2

        # Adjust the row and column indices to ensure plants are centered and not planted on the edges
        row_start = offset
        row_end = row_length - offset
        col_start = start_col + offset
        col_end = end_col - offset

        # Generate grid indices for rows
        row_indices = np.arange(row_start, row_end, plant_distance)

        # Alternating columns offset for even rows
        col_indices_odd = np.arange(col_start, col_end, plant_distance)
        col_indices_even = col_indices_odd + offset

        # Ensure the column indices do not exceed the end boundary
        col_indices_even = col_indices_even[col_indices_even < col_end]
        col_indices_odd = col_indices_odd[col_indices_odd < col_end]

        # Separate the row indices for odd and even rows to create an alternating pattern
        row_grid_odd = row_indices[::2]
        row_grid_even = row_indices[1::2]

        # Apply planting to odd rows with standard indices and even rows with offset indices
        self.apply_planting(row_grid_odd, col_indices_odd, plant_parameters, sim)
        self.apply_planting(row_grid_even, col_indices_even, plant_parameters, sim)




    def _random_planting(self, strip_parameters, start_col, end_col, plant_parameters, sim):
        """
        Plants crops in a random pattern within the specified column boundaries
        of a strip in the simulation.

        This method calculates random positions for planting within the defined grid, based on the provided
        parameters for distance and row length, ensuring that crops are not planted too close to the edges.

        Parameters:
            strip_parameters (dict): Dictionary containing parameters like row length and distance between rows.
            start_col (int): The starting column index for the strip where planting begins.
            end_col (int): The ending column index for the strip where planting ends.
            plant_parameters (dict): A dictionary of parameters specific to the plant being planted.
            sim (Simulation): The simulation object that manages the state and layers of the simulation.

        Returns:
            None: This method directly modifies the simulation's state and does not return any value.
        """
        # Calculate the distance between plants and the total number of plants to plant based on area and spacing
        plant_distance = strip_parameters["rowDistance"]
        num_plants = int(
            (strip_parameters["rowLength"] * (end_col - start_col)) / (plant_distance * 100)
        )

        # Calculate offsets to prevent planting at the very edges of the strip
        offset = plant_distance // 2
        row_start = offset
        row_end = strip_parameters["rowLength"] - offset
        col_start = start_col + offset
        col_end = end_col - offset

        # Calculate the adjusted grid dimensions based on offsets
        adjusted_row_length = row_end - row_start
        adjusted_col_length = col_end - col_start

        # Determine total possible planting positions and select random indices for planting
        total_positions = adjusted_row_length * adjusted_col_length
        plant_positions = np.random.choice(total_positions, num_plants, replace=False)
        
        # Convert linear indices to 2D indices
        row_indices, col_indices = np.unravel_index(
            plant_positions, (adjusted_row_length, adjusted_col_length)
        )

        # Adjust indices to fit within the actual grid coordinates
        row_indices += row_start
        col_indices += col_start

        # Apply the planting by setting the appropriate positions in the simulation's layers
        sim.crops_pos_layer[row_indices, col_indices] = True  # Mark positions in the crop position layer

        # Vectorize the creation of crop objects for efficient batch processing across the grid
        create_crop = np.vectorize(lambda r, c: Crop(self.plantType, (r, c), plant_parameters, sim))
        crops = create_crop(row_indices, col_indices)

        # Place the newly created crop objects in the simulation's crop object layer
        sim.crops_obj_layer[row_indices, col_indices] = crops

        # Initialize the size of each crop in the crop size layer
        sim.crop_size_layer[row_indices, col_indices] = 0.001

        # Update the number of plants that have been planted
        self.num_plants = np.size(row_indices)



    def harvesting(self, sim):
        """
        Performs the harvesting process based on the plant growth and specified harvesting strategy.

        Parameters:
            sim (Simulation): The simulation instance containing all necessary data and state configurations.

        This method checks if the required minimum duration since planting has been met, then performs
        harvesting based on the configured type (`max_Yield`, `max_quality`, or `earliest`). It utilizes
        a helper function `perform_harvest` to manage the harvesting mechanics.
        """
        # Check if 10 days have passed since planting to begin possible harvesting
        if (sim.current_date - self.sowing_date).days < 10:
            return  # Not enough time has elapsed to start harvesting

        harvesting_type = self.harvesttype
        strip = sim.crop_size_layer[:, self.start:self.start + self.width]
        pos = sim.crops_pos_layer[:, self.start:self.start + self.width]
        #size at pos
        size_at_pos = np.multiply(strip, pos)[np.multiply(strip, pos)>0]
        plant_sizes = strip[strip > 0]
        # Extract sizes of non-zero plants for consideration
        # Harvest based on maximum yield strategy
        if harvesting_type == "max_yield":
            current_average_size = np.mean(plant_sizes)
            rounded_current_size = round(current_average_size, 3)

            self.previous_sizes.pop(0)  # Remove the oldest recorded size
            self.previous_sizes.append(rounded_current_size)  # Add the latest size to records

            # Trigger harvest if the last 5 recorded sizes show stable growth
            if self.previous_sizes.count(rounded_current_size) == 5:
                self.perform_harvest(sim, strip)

        # Harvest based on maximum quality strategy
        elif harvesting_type == "max_quality":
            # Check if any plant has reached the maximum size threshold
            max_size_reached = np.any(size_at_pos >= self.plant_parameters["W_max"] - 5)
            if max_size_reached:
                self.perform_harvest(sim, strip)
        # Harvest based on the earliest possible strategy
        elif harvesting_type == "earliest":
            average_size = np.mean(size_at_pos)
            # Check if the average size meets 50% of the maximum size
            if average_size >= (0.5 * self.plant_parameters["W_max"]):
                self.perform_harvest(sim, strip)

        else:
            print("Invalid harvesting type. Simulation finished.")
            sim.finish = True  # Terminate the simulation if harvesting type is invalid


    def perform_harvest(self, sim, strip):
        """
        Manages the mechanics of harvesting by clearing harvested plants from various simulation layers
        and preparing for the next planting cycle if applicable.

        Parameters:
            sim (Simulation): The simulation instance to modify.
            strip (array): The slice of the crop size layer representing the current strip being harvested.
        """
        # Move the plants to the harvested plants layer and clear them from active layers
        sim.harvested_plants[:, self.start : self.start + self.width] = strip
        # Clear the harvested plants from the crop_size_layer
        sim.crop_size_layer[:, self.start : self.start + self.width+2] = 0
        # clear the crops from the crop_obj_layer
        sim.crops_obj_layer[:, self.start : self.start + self.width] = None
        # clear the crops from the crop_pos_layer
        sim.crops_pos_layer[:, self.start : self.start + self.width] = False
        # Update the current set index and check if replanting is necessary

        
        # replant the strip if num of sets is not reached
        if self.current_set < self.num_sets:
            self.num_plants = 0
            self.previous_sizes = [None] * 5
            self.planting(sim)
            self.current_set += 1
        else:
            if np.sum(sim.crop_size_layer) == 0:
                sim.finish = True



            
                
class Simulation:
    """
    A class designed to simulate crop growth and management across a specified area,
    integrating various environmental factors and growth behaviors.

    Attributes
    ----------
    input_data : dict
        Configuration parameters for the simulation, including planting details.
    weather_data : dict
        Weather data affecting crop and weed growth throughout the simulation.
    total_width : int
        Total width of the simulation area, derived from the sum of all strip widths.
    water_layer : np.ndarray
        2D array representing water levels across the entire simulation area.
    crop_size_layer : np.ndarray
        2D array tracking the size of crops at each grid position.
    crops_pos_layer : np.ndarray
        Boolean array indicating which positions are occupied by crops.
    crops_obj_layer : np.ndarray
        Array storing references to crop objects at each grid position.
    boundary_layer : np.ndarray
        Array indicating the boundaries of crop areas within the simulation grid.
    weeds_size_layer : np.ndarray
        Array tracking the size of weeds at each grid position.
    weeds_obj_layer : np.ndarray
        Array storing references to weed objects at each grid position.
    weeds_pos_layer : np.ndarray
        Boolean array indicating which positions are occupied by weeds.
    lock : Lock
        Thread lock for managing concurrent access to the simulation's shared resources.
    date : str
        The start date of the simulation.
    current_date : datetime
        Current simulation date and time.
    harvested_plants : np.ndarray
        Array storing the total amount of plants harvested per position.
    stepsize : int
        Time step size in simulation days, dictating the frequency of simulation updates.

    Methods
    -------
    planting()
        Initiates planting based on configured parameters.
    harvesting()
        Determines and executes harvesting operations based on crop maturity.
    grow_weeds()
        Simulates weed growth using environmental data and time progression.
    grow_plants()
        Manages crop growth using environmental conditions and current crop statuses.
    run_simulation()
        Executes the full simulation loop, updating states and managing events.
    record_data(date, size, growth_rate, water_level, overlap, size_layer, boundary, weed_size_layer)
        Logs simulation data for analysis and review.
    """

    def __init__(self, input_data, weather_data):
        """
        Initializes the Simulation with necessary data and configurations.

        Parameters
        ----------
        input_data : dict
            Contains all input settings including planting types and row configurations.
        weather_data : dict
            Historical or forecasted weather data used to influence the simulation outcomes.
        """
        self.input_data = input_data
        self.weather_data = weather_data
        self.total_width = sum(row["stripWidth"] for row in input_data["rows"])
        length = int(input_data["rowLength"])

        # Initialize simulation layers for water, crops, and weeds
        self.water_layer = np.full((length, self.total_width), 0.5, dtype=float)
        self.crop_size_layer = np.zeros((length, self.total_width+2), dtype=float)
        self.crops_pos_layer = np.zeros((length, self.total_width), dtype=bool)
        self.crops_obj_layer = np.full((length, self.total_width), None, dtype=object)
        self.boundary_layer = np.zeros((length, self.total_width), dtype=int)
        self.weeds_size_layer = np.zeros((length, self.total_width), dtype=float)
        self.weeds_obj_layer = np.full((length, self.total_width), None, dtype=object)
        self.weeds_pos_layer = np.zeros((length, self.total_width), dtype=bool)
        self.harvested_plants = np.zeros((length, self.total_width), dtype=float)

        # Configure thread lock for safe concurrent operations
        self.lock = Lock()

        #finish flag to indicate simulation completion
        self.finish = False

        # Initialize the date settings
        self.current_date = datetime.strptime(input_data["startDate"] + " 00:00:00", "%Y-%m-%d %H:%M:%S")
        self.stepsize = int(input_data["stepSize"])

        # Create strips based on input data
        self.strips = np.array([
            Strip(
                strip["stripWidth"],
                strip["plantType"],
                strip["plantingType"],
                strip["rowSpacing"],
                self,
                index,
                strip["numSets"],
                input_data["harvestType"],
            ) for index, strip in enumerate(input_data["rows"])
        ])
  
    def grow_weeds(self, strip):
        """
        Simulates the growth of weeds across the specified strip of the simulation area. This function
        manages weed growth in parallel to enhance performance, leveraging multiple threads to handle
        the computational load.

        Parameters:
            strip (Strip): The strip object where weeds need to be grown.

        This function grows existing weeds and potentially spawns new weeds randomly based on
        environmental conditions and weed growth patterns.
        """
        with self.lock:  # Ensure thread safety with a lock during weed growth calculations
            # Select only the positions where weeds are present
            weeds_present = self.weeds_pos_layer
            weeds = self.weeds_obj_layer[weeds_present]
            if len(weeds) > 0:
                weed_list = np.ravel(weeds)  # Flatten the array of weed objects for processing
                np.random.shuffle(weed_list)  # Shuffle for random growth order

                def grow_subset(subset):
                    """Function to grow a subset of weeds."""
                    for weed in subset:
                        weed.grow(
                            self.weeds_size_layer,
                            self.weeds_obj_layer,
                            self.weeds_pos_layer,
                            strip,
                        )

                # Determine the number of cores available for parallel processing
                num_cores = cpu_count()
                # Split the weed list into approximately equal subsets for each core
                weed_subsets = np.array_split(weed_list, num_cores)

                # Execute the growth in parallel across the subsets
                with ThreadPoolExecutor(max_workers=num_cores) as executor:
                    futures = [executor.submit(grow_subset, subset) for subset in weed_subsets]
                    for future in futures:
                        future.result()  # Ensure all processing completes

            # Randomly attempt to spawn a new weed based on existing crop sizes and a random chance
            weed_x = np.random.randint(0, self.crop_size_layer.shape[0], 1)
            weed_y = np.random.randint(0, self.crop_size_layer.shape[1] - 2, 1)
            size_at_spot = self.crop_size_layer[weed_x, weed_y]
            random_chance = np.random.uniform(0, (24 + size_at_spot) / self.stepsize, 1)
            if random_chance <= 0.2:  # Condition for new weed emergence
                weed_parameters = Strip.get_plant_parameters("weed")  # Get weed growth parameters
                self.weeds_pos_layer[weed_x, weed_y] = True  # Mark position as occupied by a weed
                # Create a new weed object at the determined position
                self.weeds_obj_layer[weed_x, weed_y] = Crop(
                    "weed", (int(weed_x), int(weed_y)), weed_parameters, self
                )

    def grow_plants(self, strip):
        """
        Simulates the growth of crops within a specified strip of the simulation area.
        This method manages the growth in parallel using multiple threads to enhance
        performance, efficiently handling computational load across the available cores.

        Parameters:
            strip (Strip): The specific strip where crops are being grown, which influences
                        local conditions such as spacing and type.

        Returns:
            tuple: Returns the total growth rate and overlap accumulated from all crop growth
                within the simulation step.

        This function processes crop growth based on their current size and position, adjusting
        growth rates and detecting overlaps.
        """
        with self.lock:  # Ensure thread safety with a lock during crop growth calculations
            # Select only the positions where crops are present
            crops_present = self.crops_pos_layer
            crops = self.crops_obj_layer[crops_present]
            crop_list = np.ravel(crops)  # Flatten the array of crops for processing
            np.random.shuffle(crop_list)  # Randomize the order to simulate natural variability

            def grow_subset(subset):
                """Function to grow a subset of crops and calculate growth metrics."""
                subset_growthrate = 0
                subset_overlap = 0
                for crop in subset:
                    # Each crop grows based on its current state and the environmental conditions
                    crop_growthrate, crop_overlap = crop.grow(
                        self.crop_size_layer,
                        self.crops_obj_layer,
                        self.crops_pos_layer,
                        strip,
                    )
                    subset_growthrate += crop_growthrate
                    subset_overlap += crop_overlap
                return subset_growthrate, subset_overlap

            # Determine the number of cores available and split the crop list accordingly
            num_cores = cpu_count()
            crop_subsets = np.array_split(crop_list, num_cores)

            # Use ThreadPoolExecutor to handle growth in parallel
            growthrate = 0
            overlap = 0
            with ThreadPoolExecutor(max_workers=num_cores) as executor:
                futures = [executor.submit(grow_subset, subset) for subset in crop_subsets]
                for future in futures:
                    # Collect growth rates and overlap from each thread
                    thread_growthrate, thread_overlap = future.result()
                    growthrate += thread_growthrate
                    overlap += thread_overlap

            return growthrate, overlap


    def run_simulation(self, iteration_instance):
        """
        Manages the execution of the entire simulation loop, updating crop and weed growth,
        and handles data recording until the simulation completion condition is met.

        Parameters:
            iteration_instance: An object representing the specific iteration of this simulation run,
                                typically used to track simulation runs in a larger experiment.

        Continuously updates the simulation's state based on growth functions and checks for harvesting,
        until a specified end date or other stopping condition (`self.finish`) is reached.
        Each state of the simulation is recorded for analysis.
        """
        # Initial planting for each strip
        for strip in self.strips:
            strip.planting(self)

        # Simulation main loop
        #while self.current_date < datetime.strptime(self.input_data["startDate"] + ":00:00:00", "%Y-%m-%d:%H:%M:%S") + timedelta(days=53):
        while not self.finish:
            total_growthrate, total_overlap = 0, 0
            start_time = time.time()

            # Update crop and weed growth for each strip
            for strip in self.strips:
                growthrate, overlap = self.grow_plants(strip)
                total_growthrate += growthrate
                total_overlap += overlap
                if self.input_data["allowWeedgrowth"]:
                    print("Weed growth enabled")
                    self.grow_weeds(strip)

            end_time = time.time()
            time_needed = end_time - start_time

            # Record the current state of the simulation
            self.record_data(time_needed, iteration_instance, total_growthrate, total_overlap)

            # Move simulation time forward by the specified step size
            self.current_date += timedelta(hours=self.stepsize)

            # Check for harvesting in each strip
            for strip in self.strips:
                strip.harvesting(self)

        print("Simulation finished.")

    def record_data(self, time_needed, iteration_instance, total_growthrate, total_overlap):
        """
        Records the current state of the simulation into a DataFrame for later analysis.

        Parameters:
            time_needed (float): The time taken to process the current simulation step.
            iteration_instance: The iteration object to which this data belongs.
            total_growthrate (float): The total growth rate of plants in this step.
            total_overlap (float): The calculated overlap of plant positions in this step.
        """
        # Calculate yields and other statistics
        index_where = np.where(self.crops_pos_layer > 0)
        yields = np.sum(self.crop_size_layer[index_where]) * self.strips[0].plant_parameters["Yield"]
        num_plants = np.sum(self.crops_pos_layer)
        profit = yields * self.strips[0].plant_parameters["revenue"] * 0.001 - 0.05 * self.strips[0].num_plants

        # Prepare the data dictionary for recording
        data = {
            "date": self.current_date,
            "yield": yields,
            "growth": total_growthrate,
            "water": np.sum(self.water_layer),
            "overlap": total_overlap,
            "map": self.crop_size_layer.tolist(),
            "boundary": self.boundary_layer.tolist(),
            "weed": self.weeds_size_layer.tolist(),
            "time_needed": time_needed,
            "profit": profit,
            #if the date exeeds the weather data, the value for rain and temperature gets set to 0
            "temperature": self.weather_data[self.current_date]["temperature"] if self.current_date in self.weather_data else 0,
            "rain": self.weather_data[self.current_date]["rain"] if self.current_date in self.weather_data else 0,
            "num_plants": num_plants,
        }

        # Save the data using the iteration instance's methods
        output_instance = DataModelOutput(iteration=iteration_instance)
        output_instance.set_data(data)
        output_instance.save()




def main(input_data):
    """
    The main entry point for running simulations with provided input data.
    This function handles different simulation modes based on the input configuration.

    Parameters:
        input_data (dict): Configuration parameters and data necessary for initializing the simulation.

    Returns:
        str: The name of the simulation, indicating which simulation instance was run.
    """

    # Save the initial setup data and obtain an instance containing the saved data
    input_instance = save_initial_data(input_data)

    # Fetch weather data which may impact simulation conditions
    weather_data = fetch_weather_data()

    # Check and handle if the simulation is running in iteration (testing) mode
    if input_data["iterationMode"]:
        handle_testing_mode(input_data, input_instance, weather_data)
    else:
        run_standard_simulation(input_data, input_instance, weather_data)

    # Return the name of the simulation instance, useful for tracking or referencing
    return input_instance.simName

def save_initial_data(input_data):
    """
    Saves the initial configuration data into a persistent storage model and creates detailed
    records for each row in the simulation setup.

    Parameters:
        input_data (dict): The simulation configuration data.

    Returns:
        DataModelInput: The database instance that represents the initial simulation configuration.
    """
    # Create an instance of DataModelInput using the provided input data
    input_instance = DataModelInput(
        startDate=input_data.get('startDate'),
        stepSize=input_data.get('stepSize'),
        rowLength=input_data.get('rowLength'),
        testingMode=input_data.get('testingMode'),
        simName=input_data.get('simName')
    )
    
    # Extract and handle specific testing data if in testing mode
    testing_data = input_data.get('iterationData', {})
    if testing_data and input_instance.testingMode:
        # Attempt to get a single key-value pair from testing data
        input_instance.testingKey, input_instance.testingValue = next(iter(testing_data.items()), (None, None))
        # If the value is a dictionary, handle or convert it appropriately
        if isinstance(input_instance.testingValue, dict):
            input_instance.testingValue = -99  # This is a placeholder for error or specific handling
    else:
        input_instance.testingKey = None
        input_instance.testingValue = None
    
    # Persist the initial configuration to the database
    input_instance.save()
    
    # Create and save detailed row configuration to the database
    for row in input_data.get('rows', []):
        row_instance = RowDetail(
            plantType=row.get('plantType'),
            plantingType=row.get('plantingType'),
            stripWidth=row.get('stripWidth'),
            rowSpacing=row.get('rowSpacing'),
            numSets=row.get('numSets'),
            input_data=input_instance  # Establish a relationship with the main input data
        )
        row_instance.save()

    return input_instance




def fetch_weather_data():
    """
    Retrieves weather data from the database and organizes it into a dictionary keyed by date.

    This function fetches weather data stored in a Django model `Weather`, which includes
    dates, rain, and temperature metrics. Each date is processed to ensure it is in a
    usable format, and the data is returned as a dictionary.

    Returns:
        dict or None: Returns a dictionary mapping dates to weather data if successful;
                      returns None if the data is not found or an error occurs.
    """
    try:
        # Query the database for weather data, retrieving it as a list of dictionaries
        weather_data = Weather.objects.values('date', 'rain', 'temperature')
        weather_data_dict = {}

        # Process each weather data entry
        for data in weather_data:
            try:
                # Parse the date using a flexible parser to handle different date formats robustly
                if data["date"] not in [None, 'NaT', '']:
                    data['date'] = parser.parse(data["date"])
                else:
                    continue  # Skip entries with no valid date, preventing errors in the dictionary
            except ValueError:
                # Handle cases where the date format is incorrect and unparseable
                print(f"Skipping invalid date format: {data['date']}")
                continue  # Skip this entry and proceed with the next

            # Store the parsed data in a dictionary keyed by the date
            weather_data_dict[data['date']] = data

        return weather_data_dict
    except ObjectDoesNotExist:
        # Handle the case where the Weather model does not exist in the database
        print("Weather data not found in the database.")
        return None


def handle_testing_mode(input_data, input_instance, weather_data):
    """
    Orchestrates the testing mode operations based on the provided input data configurations.

    This function differentiates between row variations and other parameter variations in the testing mode.
    Depending on the configuration, it delegates the task to appropriate handlers to execute simulations 
    with varied parameters or configurations.

    Parameters:
        input_data (dict): Contains all input settings including detailed configurations for each testing iteration.
        input_instance (DataModelInput): The database instance storing initial configuration details.
        weather_data (dict): Environmental data to be used during simulation.

    This function checks if the testing mode variations involve row-specific changes or broader parameter adjustments,
    directing the flow accordingly.
    """
    if "rows" in input_data["iterationData"]:
        # If row-specific variations are provided in the testing data, process each row variation
        handle_row_variations(input_data, input_instance, weather_data)
    else:
        # Otherwise, handle other parameter variations that affect the whole simulation
        handle_parameter_variations(input_data, input_instance, weather_data)


def handle_row_variations(input_data, input_instance, weather_data):
    """
    Executes simulations for each row variation provided in the input data during testing mode.

    This function iterates over each row configuration in the input data, creating a modified input data set
    for each row variation and executing the simulation accordingly.

    Parameters:
        input_data (dict): The complete set of input data provided for the simulation.
        input_instance (DataModelInput): The database instance related to the current simulation input.
        weather_data (dict): The weather data applicable to the simulations.

    Each iteration modifies the input data for a specific row before running the simulation,
    ensuring each row's unique conditions are tested.
    """
    # Process each row in the input data for specific variations
    for row_index, row in enumerate(input_data['rows']):
        # Create a modified version of the input data for this specific row variation
        modified_input_data = create_modified_input_data(input_data, row_index)
        # Create a new iteration instance in the database for this variation
        iteration_instance = create_iteration_instance(input_instance, row_index, -99)
        # Run the simulation with the modified input data and weather conditions
        start_simulation(modified_input_data, weather_data, iteration_instance)


def handle_parameter_variations(input_data, input_instance, weather_data):
    """
    Processes and runs simulations for different parameter variations specified in testing mode.

    This function identifies the parameter to vary from the `iterationData` provided in the input_data.
    It adjusts this parameter over a specified range or set of values and executes the simulation for each
    modified configuration.

    Parameters:
        input_data (dict): The simulation configuration data which includes testing variations.
        input_instance (DataModelInput): The database instance that tracks the simulation configuration.
        weather_data (dict): The environmental data applicable to the simulations.

    The function supports variations for both date and numerical parameters, handling each type appropriately.
    """
    # Extract the key and value for testing from the iterationData dictionary
    testing_key, testing_value = next(iter(input_data["iterationData"].items()))
    # Determine the starting and ending values for the parameter to vary
    if testing_key == "startDate":
        # Parsing the start and end dates from input and converting them into datetime objects
        start_date = datetime.strptime(input_data.get(testing_key, input_data["rows"][0].get(testing_key, "1970-01-01")), "%Y-%m-%d")
        end_date = datetime.strptime(testing_value, "%Y-%m-%d")

        # Generate a date range and run simulation for each date in the range
        current_date = start_date
        while current_date <= end_date:
            # Modify the input data for the current date parameter
            modified_input_data = modify_input_data_for_parameter(input_data, testing_key, current_date.strftime("%Y-%m-%d"))
            # Convert the date into a number format YYYYMMDD for unique identification
            date_number = int(current_date.strftime("%Y%m%d"))
            iteration_instance = create_iteration_instance(input_instance, date_number, date_number)
            # Run the simulation with the modified date
            start_simulation(modified_input_data, weather_data, iteration_instance)
            current_date += timedelta(days=1)  # Increment the day by one

    else:
        # Handle non-date parameters: assuming they are integers
        start_value = int(input_data["rows"][0].get(testing_key, input_data.get(testing_key, -99)))
        end_value = int(testing_value)

        # Running simulation for each integer value in the range
        for param_value in range(start_value, end_value + 1):
            # Modify the input data for the current parameter value
            modified_input_data = modify_input_data_for_parameter(input_data, testing_key, param_value)
            iteration_instance = create_iteration_instance(input_instance, param_value, param_value)
            # Run the simulation with the modified parameter
            start_simulation(modified_input_data, weather_data, iteration_instance)


def run_standard_simulation(input_data, input_instance, weather_data):
    """
    Executes the simulation with the default setup, typically used when not operating under testing conditions.

    Parameters:
        input_data (dict): Contains all input settings necessary for the simulation.
        input_instance (DataModelInput): The database instance storing initial configuration details.
        weather_data (dict): Environmental data that may influence the simulation outcomes.
    """
    # Create an iteration instance for tracking and logging purposes, assuming a default configuration
    iteration_instance = create_iteration_instance(input_instance, index=1, param_value=-99)
    # Initiate the simulation with the provided data
    start_simulation(input_data, weather_data, iteration_instance)

def modify_input_data_for_parameter(input_data, key, value):
    """
    Alters a specific parameter within the input data, creating a modified copy for simulation purposes.

    Parameters:
        input_data (dict): The original simulation input data.
        key (str): The parameter key to be modified.
        value (any): The new value to assign to the parameter key.

    Returns:
        dict: A new dictionary with the specified parameter modified.
    """
    # Create a shallow copy of the input data to avoid modifying the original
    modified_input_data = input_data.copy()
    # Check if the key is directly within the input data or within the rows' configuration
    if key in modified_input_data:
        modified_input_data[key] = value
    else:
        modified_input_data["rows"][0][key] = value
    return modified_input_data

def create_modified_input_data(original_data, row_index):
    """
    Isolates a specific row from the input data for focused simulation runs during testing mode.

    Parameters:
        original_data (dict): The complete input data from which a row will be isolated.
        row_index (int): The index of the row to isolate for the simulation.

    Returns:
        dict: A modified version of the original data containing only the selected row.
    """
    # Perform a deep copy of the original data to prevent alterations to the original structure
    modified_data = original_data.copy()
    # Select the specific row based on the provided index
    selected_row = original_data['rows'][row_index]
    # Replace the entire rows array with just the selected row
    modified_data['rows'] = [selected_row]
    return modified_data

def create_iteration_instance(input_instance, index, param_value):
    """
    Creates a new simulation iteration instance for recording and tracking purposes during the simulation.

    Parameters:
        input_instance (DataModelInput): The input configuration linked to the simulation.
        index (int): The index number identifying this particular iteration.
        param_value (int): A parameter value specific to this iteration, used for tracking and analysis.

    Returns:
        SimulationIteration: A new instance representing this specific iteration of the simulation.
    """
    # Create and return a new SimulationIteration object
    return SimulationIteration.objects.create(
        input=input_instance,
        iteration_index=index,
        param_value=param_value
    )

def start_simulation(input_data, weather_data, iteration_instance):
    """
    Initializes the simulation environment and starts the simulation process.

    Parameters:
        input_data (dict): The simulation's configuration and input data.
        weather_data (dict): Weather data affecting the simulation conditions.
        iteration_instance (SimulationIteration): An object tracking this specific run of the simulation.

    Initiates the simulation based on the provided data, configuring environmental factors and input settings.
    """
    # Initialize the Simulation class with the provided data
    sim = Simulation(input_data, weather_data)
    # Start the simulation process
    sim.run_simulation(iteration_instance)
