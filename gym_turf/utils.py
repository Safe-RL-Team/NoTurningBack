import numpy as np


def generate_fixed_room():
    room_structure = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 2, 4],
                               [0, 0, 0, 0, 2, 2, 2, 2, 2, 0],
                               [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
                               [0, 2, 2, 2, 0, 0, 0, 0, 0, 0],
                               [0, 2, 2, 0, 0, 0, 0, 0, 0, 0],
                               [0, 2, 2, 0, 0, 0, 0, 0, 0, 0],
                               [2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                               [2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                               [2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                               [2, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    return room_structure, np.array([9, 0]), np.array([0, 9])


TYPE_LOOKUP = {
    0: 'grass',
    1: 'ruined grass',
    2: 'stone path',
    3: 'agent',
    4: 'goal',
}

ACTION_LOOKUP = {
    0: 'move up',
    1: 'move down',
    2: 'move left',
    3: 'move right',
}

CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}
