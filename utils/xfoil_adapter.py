from utils.save_coordinates_to_dat_file import save_coordinates_to_dat_file
from utils.bezier_parametrization import BezierAirfoil
import numpy as np
from xfoil import XFoil
import logging
import tempfile

class XFoilAdapter():
    """
    Adapter class for xfoil to work with bezier parsec airfoils
    """
    def __init__(self, timeout=5):
        self.airfoils = []
        self.temp_files = []
        self.reynolds = None
        self.mach = None
        self.alphas = None
        self.timeout = timeout

    def __enter__(self):
        return self

    def set_airfoils(self, airfoils: list[BezierAirfoil]):
        if(len(self.temp_files) > 0):
            for file in self.temp_files:
                file.close()
            self.temp_files = []

        self.airfoils = airfoils
        for airfoil in airfoils:
            f = tempfile.NamedTemporaryFile(mode='w+t', suffix='.dat', delete=False)
            self.temp_files.append(f)
            coordinates = airfoil.get_coordinates()
            save_coordinates_to_dat_file(f, coordinates)

    def set_run_condition(self, reynolds, mach, alphas):
        self.reynolds = reynolds
        self.mach = mach
        self.alphas = alphas

    def run(self):
        if(len(self.temp_files) == 0):
            raise Exception('No airfoils set')
        if(self.reynolds == None or self.mach == None or self.alphas == None):
            raise Exception('No run condition set')
        
        logging.debug(f'Loading airfoils into xfoil: {[tempfile.name for tempfile in self.temp_files]}')
        logging.debug(f'Run condition: Re={self.reynolds}, M={self.mach}, alphas={self.alphas}')
        logging.debug(f'Temp file contents: {[temp_file.read() for temp_file in self.temp_files]}')
        xfoil_object = XFoil(
            [temp_file.name for temp_file in self.temp_files],
            reynolds=self.reynolds,
            mach=self.mach,
            alphas=self.alphas,
            disable_graphics=True,
            process_timeout=self.timeout
        )
        xfoil_object.run()
        number_of_airfoils = len(self.airfoils)
        results = [(int(index),xfoil_object.results[index]) for index in xfoil_object.results]
        results = sorted(results, key=lambda x: x[0])
        np_results = np.array([result[1] for result in results]).reshape(-1, number_of_airfoils)
        return np_results
    
    def __exit__(self, type, value, tb):
        for file in self.temp_files:
            file.close()

