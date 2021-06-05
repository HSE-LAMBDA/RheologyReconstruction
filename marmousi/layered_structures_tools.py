import numpy as np

from scipy import interpolate


def get_mask(x_size, y_size, min_depth_step, max_depth_step, layers_sharpness, horizon_trend):
    """
    Creates mask for layered structure. Each mask values belongs to (0, 1].
    It is supposed that lower values denote softer media, and higher values are for harder layers.
    However, the caller is free to interpret the values in any manner.
    :param x_size: int, horizontal size of the mask, in points
    :param y_size: int, vertical size of the mask, in points
    :param min_depth_step: int, proposed minimal *average* vertical depth of each layer
    :param max_depth_step: int, proposed maximal *average* vertical depth of each layer
    :param layers_sharpness: int, how sharp is the difference between neighbour layers (probably, belongs to (3, 10))
    :param horizon_trend: int, proposed *typical* layer vertical displacement in this horizontal area in question
    :return: np.array, the shape is (x_size, y_size), the values are (0, 1]
    """

    # Initial experiments were carried out with the following values:
    #
    # X_SIZE = 170
    # Y_SIZE = 35
    # MIN_DEPTH_STEP = 2
    # MAX_DEPTH_STEP = 5
    # LAYERS_SHARPNESS = 5
    # HORIZON_TREND = 10

    # Magic numbers

    # How many points are used to create the form of the border of the layer
    BASE_POINTS_NUMBER = 5

    # Max horizontal displacement of the base point between adjacent layer borders
    X_DISTORT = 0.1

    # Min/max values for the initial vertical distribution
    INIT_FORM_MIN = 3
    INIT_FORM_MAX = 8

    # Min/max values for the horizontal distances between base points
    max_base_x_step = x_size // (BASE_POINTS_NUMBER + 1)
    min_base_x_step = int(max_base_x_step * 0.5)

    # We expect to start somewhere near the top left corner of the area (see the logic below)
    START_DEPTH = y_size + INIT_FORM_MAX

    # This is the number of borders required to cover the area in the worst case
    NUMBER_OF_BORDERS = (START_DEPTH + abs(horizon_trend)) // min_depth_step
    # However, we will limit this number, if we see the risk that the points will flip-flop horizontally
    NUMBER_OF_BORDERS = min(NUMBER_OF_BORDERS, int(min_base_x_step / (2.0 * X_DISTORT)))

    # Ok, it's the base x grid for everything
    x_grid = np.linspace(0, x_size, num=x_size, endpoint=True)

    # Random horizontal differences between base points
    x_steps = np.random.random_integers(min_base_x_step, max_base_x_step, size=(BASE_POINTS_NUMBER,))
    # Horizontal positions of base points
    x = np.concatenate(([0], np.cumsum(x_steps), [x_size]))

    # Vertical positions of base points
    distribution = np.random.random_integers(INIT_FORM_MIN, INIT_FORM_MAX, size=(BASE_POINTS_NUMBER + 2,))
    # I mentioned, that we are going to start somewhere near the top left corner, right?
    y = START_DEPTH - distribution + horizon_trend * x / x_size

    # The list of our borders between layers
    curves = []

    for i in range(NUMBER_OF_BORDERS):
        # Small random movement of base points horizontally (except the first and the last, of course)
        dx = np.concatenate((np.array([0]),
                            np.sort(np.random.uniform(-X_DISTORT, X_DISTORT, size=(BASE_POINTS_NUMBER,))),
                            np.array([0])))

        # Not so small vertical movements.
        # Alarm: we control min_depth_step and max_depth_step, but they are not strict limits! (See below.)
        # It's an expected behaviour, but be aware.
        dy = np.random.uniform(min_depth_step, max_depth_step, size=(BASE_POINTS_NUMBER + 2,))

        # Move points
        x = x + dx
        y = y - dy

        # And now create the curve as the spline and append it to the list.
        # So, min_depth_step and max_depth_step can be exceeded, the nature of the splines causes it.
        curves.append(interpolate.interp1d(x, y, kind='cubic'))

    # You may uncomment the lines below for visual debugging
    #     plt.plot(x, y, 'o')
    #     plt.plot(x_grid, curves[i](x_grid), '-')
    #
    # plt.gca().set_aspect('equal')
    # plt.ylim(0, Y_SIZE)

    # The resuling mask will be here
    mask = np.ones(shape=(x_size, y_size))

    # The list of mask values for different layers
    scale_factors = []
    # Here layers_sharpness magic param comes to action
    # and sets how many neighbour layers may have different, but not very different rheology
    max_similar_layers = NUMBER_OF_BORDERS // layers_sharpness

    covered = 0
    while covered < NUMBER_OF_BORDERS:
        # How many not so different layers will be in this group
        layer_group_size = np.random.randint(1, max_similar_layers)
        # Base mask value for this group of layers
        base_val = np.random.uniform(0.1, 0.9)

        for i in range(layer_group_size):
            # Exact mask value for i-th layer, clipped by 1
            cur_val = min(base_val + np.random.uniform(0, base_val * 0.1), 1)
            scale_factors.append(cur_val)
            covered += 1

    # The matrix of repeated rows with i-th coordinates
    I = np.repeat(np.reshape(x_grid, (x_size, 1)), y_size, axis=1)
    # The matrix of repeated rows with j-th coordinates
    J = np.repeat(np.arange(y_size).reshape((1, y_size)), x_size, axis=0)

    for i in reversed(range(NUMBER_OF_BORDERS)):
        # Call the i-th curve spline, get matrix Y with the values of f_i(mask_ith_index)
        Y = curves[i](I)
        # Compare f_i(mask_ith_index) and mask_jth_index, mark the area beneath the curve
        mask[Y < J] = scale_factors[i]

    return mask