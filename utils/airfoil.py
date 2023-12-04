from abc import ABC, abstractmethod
class Airfoil(ABC):
    # Method required by XFOIL
    @abstractmethod
    def get_coordinates(self):
        pass

    # Method to add stats to deap logbook
    @abstractmethod
    def get_general_info(self):
        pass