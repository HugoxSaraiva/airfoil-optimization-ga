
def save_coordinates_to_dat_file(file, coordinates):
    file.write('\n'.join([f'{coordinate[0]:6f} {coordinate[1]:6f}' for coordinate in coordinates]))
