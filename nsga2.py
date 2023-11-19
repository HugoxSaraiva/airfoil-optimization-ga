import array
import random
import numpy as np
import numpy
import math
from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools
import pickle
import array
from utils.bezier_parametrization import BezierAirfoilNSGA2Adapter
from utils.bezier_parsec_parametrization import BezierParsecAirfoilNSGA2Adapter
from utils.xfoil_adapter import XFoilAdapter
from multiprocessing import Pool

creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

PARALEL_PROCESSES = 8
SEED = 42
CONTROL_POINTS_SHAPE=(6,6)
FREQ = 5
NGEN = 150
MU = 100
CXPB = 0.9

# Register the airfoil parametrization class
toolbox.register("airfoil", BezierAirfoilNSGA2Adapter, shape=CONTROL_POINTS_SHAPE)
# toolbox.register("airfoil", BezierParsecAirfoilNSGA2Adapter)

# Problem definition
BOUND_UP, BOUND_LOW  = toolbox.airfoil().get_bounds()
NDIM = len(BOUND_UP)

def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

def evaluate(individual):
    with XFoilAdapter(timeout=12) as xfoil:
        try:
            airfoil = toolbox.airfoil().from_parameters(individual)
            xfoil.set_airfoils(airfoils=[airfoil])
            xfoil.set_run_condition(
                reynolds=3e6,
                mach=0,
                alphas=[8],
            )
            results = xfoil.run()
            # We have only one run, so we can just take the first element
            run_results = results[0][0].get('result', None)
            if run_results is None:
                return 1000, 0
            cl = run_results['CL'][0]
            cd = run_results['CD'][0]
            return cd, cl
        except Exception as e:
            print('Error:' , e)
            return 1000, 0

def distance(individual):
    """A distance function to the feasibility region."""
    return (individual[0] - 5.0)**2

toolbox.register("attr_float", toolbox.airfoil().random_params_initializer)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
toolbox.register("select", tools.selNSGA2)



def main(pool=None, seed=None, checkpoint=None, max_gen=NGEN):
    if pool is not None:
        toolbox.register("map", pool.map)
    if checkpoint:
        # A file name has been given, then load the data from the file
        try:
            with open(checkpoint, "rb") as cp_file:
                cp = pickle.load(cp_file)
                population = cp["population"]
                start_gen = cp["generation"]
                pareto = cp["pareto"]
                logbook = cp["logbook"]
                random.setstate(cp["rndstate"])
                np.random.set_state(cp["nprndstate"])
                best_cl_individuals = cp["best_cl_individuals"]
                best_cd_individuals = cp["best_cd_individuals"]
                best_cl_o_cd_individuals = cp["best_cl_o_cd_individuals"]
        except FileNotFoundError:
            print("Checkpoint not found. Starting a new run.")
            return main(seed=seed)
    else:
        random.seed(seed)
        np.random.seed(seed)
        # Start a new evolution
        population = toolbox.population(n=MU)
        start_gen = 0
        pareto = tools.ParetoFront()
        logbook = tools.Logbook()
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"
        best_cl_individuals = [max(population, key=lambda ind: ind.fitness.values[1])]
        best_cd_individuals = [min(population, key=lambda ind: ind.fitness.values[0])]
        best_cl_o_cd_individuals = [max(population, key=lambda ind: ind.fitness.values[1] / ind.fitness.values[0])]
    stats = tools.Statistics(lambda ind: ind.fitness.values)

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    population = toolbox.select(population, len(population))

    # stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    # Begin the generational process
    for gen in range(start_gen, max_gen):
        # Vary the population
        offspring = tools.selTournamentDCD(population, len(population))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        population = toolbox.select(population + offspring, MU)
        pareto.update(population)
        record = stats.compile(population)
        best_cl_individual = max(population, key=lambda ind: ind.fitness.values[1])
        best_cd_individual = min(population, key=lambda ind: ind.fitness.values[0])
        best_cl_o_cd_individual = max(population, key=lambda ind: ind.fitness.values[1] / ind.fitness.values[0])
        best_cl_individuals.append(best_cl_individual)
        best_cd_individuals.append(best_cd_individual)
        best_cl_o_cd_individuals.append(best_cl_o_cd_individual)
        logbook.record(
            gen=gen, 
            evals=len(invalid_ind),
            **record)
        print(logbook.stream)
        if gen % FREQ == 0 or gen == NGEN - 1:
            # Fill the dictionary using the dict(key=value[, ...]) constructor
            cp = dict(population=population, 
                      generation=gen, 
                      pareto=pareto, 
                      best_cl_individuals=best_cl_individuals,
                      best_cd_individuals=best_cd_individuals,
                      best_cl_o_cd_individuals=best_cl_o_cd_individuals, 
                      logbook=logbook, rndstate=random.getstate(), nprndstate=np.random.get_state())

            with open(f"checkpoint_name_gen_{gen}.pkl", "wb") as cp_file:
                pickle.dump(cp, cp_file)

    print("Final population hypervolume is %f" % hypervolume(population, [11.0, 11.0]))

    return population, logbook

if __name__ == "__main__":
    with Pool(PARALEL_PROCESSES) as pool:
        population, stats = main(seed=SEED, pool=pool, max_gen=NGEN)
        print(stats)