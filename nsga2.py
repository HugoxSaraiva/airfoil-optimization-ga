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
from utils.bezier_parametrization import BezierAirfoil
from utils.xfoil_adapter import XFoilAdapter

creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

SEED = 42
CONTROL_POINTS_SHAPE=(6,6)
FREQ = 10
NGEN = 150
MU = 100
CXPB = 0.9

# Problem definition
# Functions zdt1, zdt2, zdt3, zdt6 have bounds [0, 1]
BOUND_UP, BOUND_LOW  = BezierAirfoil.get_bounds(CONTROL_POINTS_SHAPE)

# Functions zdt4 has bounds x1 = [0, 1], xn = [-5, 5], with n = 2, ..., 10
# BOUND_LOW, BOUND_UP = [0.0] + [-5.0]*9, [1.0] + [5.0]*9

# Functions zdt1, zdt2, zdt3 have 30 dimensions, zdt4 and zdt6 have 10
NDIM = BezierAirfoil.parameters_required_for_shape(CONTROL_POINTS_SHAPE)

def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

def evaluate(individual):
    with XFoilAdapter() as xfoil:
        # xfoil.set_airfoils(airfoils=[airfoil])
        # xfoil.set_run_condition(
        #     reynolds=3e6,
        #     mach=0,
        #     alphas=[8],
        # )
        try:
            airfoil = BezierAirfoil(individual, shape=CONTROL_POINTS_SHAPE)
            # results = xfoil.run()
            # # We have only one run, so we can just take the first element
            # run_results = results[0][0].get('result', None)
            # print(run_results)
            # if run_results is None:
            #     return 1000, 0
            # cl = run_results['CL'][0]
            # cd = run_results['CD'][0]
            return individual[0], individual[1]
        except Exception as e:
            print(e)
            return 1000, 0

toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
toolbox.register("select", tools.selNSGA2)



def main(seed=None, checkpoint=None):
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
    for gen in range(start_gen, NGEN):
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
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)
        if gen % FREQ == 0 or gen == NGEN - 1:
            # Fill the dictionary using the dict(key=value[, ...]) constructor
            cp = dict(population=population, generation=gen, pareto=pareto,
                      logbook=logbook, rndstate=random.getstate(), nprndstate=np.random.get_state())

            with open(f"checkpoint_name_gen_{gen}.pkl", "wb") as cp_file:
                pickle.dump(cp, cp_file)

    print("Final population hypervolume is %f" % hypervolume(population, [11.0, 11.0]))

    return population, logbook

population, stats = main(seed=SEED)
print(stats)
# print("Convergence: ", convergence(pop, optimal_front))
# print("Diversity: ", diversity(pop, optimal_front[0], optimal_front[-1]))
