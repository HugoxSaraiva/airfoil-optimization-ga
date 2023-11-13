from pathlib import Path

def extract_coordinates_from_dat_file(path: str) -> list[list[float, float]]:
    """
    Extracts coordinates from a .dat file and returns a list of tuples
    :param path: path to file
    :return: list of tuples
    """
    with open(path, 'r') as file:
        coordinates = []
        for line in file:
            candidate = line.split()
            if len(candidate) != 2:
                continue
            x, y = candidate
            try:
                coordinates.append([float(x), float(y)])
            except ValueError:
                continue
        return coordinates

def get_upper_and_lower_surface_from_coordinates(coordinates: list[tuple[float, float]]) -> tuple[list[list[float, float]], list[list[float, float]]]:
    """
    Returns upper and lower surface coordinates from a list of airfoil coordinates, returns the coordinates sorted by the x coordinate
    :param coordinates: list of tuples
    :return: tuple of lists
    """
    upper_surface = []
    lower_surface = []
    leading_edge_index = 0
    for index, coordinate in enumerate(coordinates):
        # Finds first coordinate were the absolute value of x starts increasing
        if coordinate[0] < abs(coordinates[index+1][0]):
            leading_edge_index = index
            break
    upper_surface = coordinates[:leading_edge_index+1]
    lower_surface = coordinates[leading_edge_index:]
    upper_surface.sort(key=lambda x: x[0])
    lower_surface.sort(key=lambda x: x[0])
    return upper_surface, lower_surface

if __name__ == '__main__':
    file_path = Path(__file__)
    data_path = Path.joinpath(file_path.parent.parent, 'data')
    dat_file_path = Path.joinpath(data_path, 'NACA0012.dat')
    coordinates = extract_coordinates_from_dat_file(dat_file_path)
    print(f"Loaded {len(coordinates)} coordinates from {dat_file_path}")
    upper_surface, lower_surface = get_upper_and_lower_surface_from_coordinates(coordinates)
    print(f"Upper surface: {upper_surface}")
    print(f"Lower surface: {lower_surface}")