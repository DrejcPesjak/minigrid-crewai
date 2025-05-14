# COLOR_TO_IDX = ["red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5]
# OBJECT_TO_IDX = ["unseen": 0, "empty": 1, "wall": 2, "floor": 3, "door": 4, "key": 5, "ball": 6, "box": 7, "goal": 8, "lava": 9, "agent": 10]
# STATE_TO_IDX = ["open": 0, "closed": 1, "locked": 2]

import numpy as np

# Define mappings
COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}
IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

OBJECT_TO_IDX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10,
}
IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

STATE_TO_IDX = {"open": 0, "closed": 1, "locked": 2}
IDX_TO_STATE = dict(zip(STATE_TO_IDX.values(), STATE_TO_IDX.keys()))

# DIRECTION_MAP = {0: "North", 1: "East", 2: "South", 3: "West"}

def convert_observation(input_dict):
    # Extract components
    image = input_dict['image']
    direction_idx = input_dict['direction']
    mission = input_dict['mission']

    # Corrected direction mapping
    DIRECTION_MAP = {0: "East", 1: "South", 2: "West", 3: "North"}
    # DIRECTION_ARROW = {0: ">", 1: "v", 2: "<", 3: "^"}
    DIRECTION_ARROW = {0: "right", 1: "down", 2: "left", 3: "up"}
    direction = DIRECTION_MAP.get(direction_idx, "Unknown")

    # Convert image into object_color_state format
    grid = []
    for row in image:
        grid_row = []
        for cell in row:
            object_idx, color_idx, state_idx = cell
            object_name = IDX_TO_OBJECT.get(object_idx, "unknown")
            color_name = IDX_TO_COLOR.get(color_idx, "unknown")
            state_name = IDX_TO_STATE.get(state_idx, "unknown")
            c = ""
            if object_name in ["door","key","ball","box", "goal"]:
                c = f" {color_name}"
            s = ""
            if object_name in ["door"]:
                s = f" {state_name}"
            # if object_name == "agent":
            #     object_name += f" {DIRECTION_ARROW[direction_idx]}"
            grid_row.append(f"{object_name}{c}{s}")
        grid.append(grid_row)

    # Flip the grid 
    if direction == "East":
        grid = [row[::-1] for row in grid]
    elif direction == "West":
        pass
    elif direction == "North":
        grid = list(zip(*grid))
    elif direction == "South":
        grid = list(zip(*grid[::-1]))[::-1]
    
    # Add "Row i" to each row
    grid2 = [f"Row {i}: " + ", ".join(row) for i, row in enumerate(grid)]
    grid_string = " \n ".join(grid2)

    # # Convert grid to a string with newlines for better visualization
    # grid_string = " \n ".join(" ".join(cell for cell in row) for row in grid)

    # Create the formatted output
    formatted_observation = {
        "mission": mission,
        "direction": direction,
        "observation_grid_string": grid_string,  # String representation of the grid
    }

    return formatted_observation





# # Example input
# input_dict = {'image': np.array([[[ 0,  0,  0],
#         [ 0,  0,  0],
#         [ 0,  0,  0],
#         [ 0,  0,  0],
#         [ 0,  0,  0],
#         [ 0,  0,  0],
#         [ 0,  0,  0]],

#        [[ 0,  0,  0],
#         [ 0,  0,  0],
#         [ 0,  0,  0],
#         [ 0,  0,  0],
#         [ 0,  0,  0],
#         [ 0,  0,  0],
#         [ 0,  0,  0]],

#        [[ 0,  0,  0],
#         [ 0,  0,  0],
#         [ 0,  0,  0],
#         [ 2,  5,  0],
#         [ 2,  5,  0],
#         [ 2,  5,  0],
#         [ 2,  5,  0]],

#        [[ 0,  0,  0],
#         [ 0,  0,  0],
#         [ 0,  0,  0],
#         [ 2,  5,  0],
#         [ 1,  0,  0],
#         [ 9,  0,  0],
#         [10,  0,  0]],

#        [[ 0,  0,  0],
#         [ 0,  0,  0],
#         [ 0,  0,  0],
#         [ 2,  5,  0],
#         [ 1,  0,  0],
#         [ 1,  0,  0],
#         [ 1,  0,  0]],

#        [[ 0,  0,  0],
#         [ 0,  0,  0],
#         [ 0,  0,  0],
#         [ 2,  5,  0],
#         [ 8,  1,  0],
#         [ 9,  0,  0],
#         [ 1,  0,  0]],

#        [[ 0,  0,  0],
#         [ 0,  0,  0],
#         [ 0,  0,  0],
#         [ 2,  5,  0],
#         [ 2,  5,  0],
#         [ 2,  5,  0],
#         [ 2,  5,  0]]], dtype=np.uint8), 'direction': 0, 'mission': 'avoid the lava and get to the green goal square'}

# # Convert observation
# converted_observation = convert_observation(input_dict)

# # Print the result
# from pprint import pprint
# pprint(converted_observation)

# print(converted_observation['observation_grid_string'])

# print converted_observation row by row
# for row in converted_observation['observation_grid_string']:
#     print(row)
