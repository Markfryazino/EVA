import random
import numpy as np
import functools

import co_functions as cf
import utils


# creates the individual
def create_ind(ind_len):
    return np.random.uniform(-5, 5, size=(ind_len,))

# creates the population using the create individual function
def create_pop(pop_size, create_individual):
    return [create_individual() for _ in range(pop_size)]

# the tournament selection (roulette wheell would not work, because we can have 
# negative fitness)
def tournament_selection(pop, fits, k):
    selected = []
    for i in range(k):
        p1 = random.randrange(0, len(pop))
        p2 = random.randrange(0, len(pop))
        if fits[p1] > fits[p2]:
            selected.append(np.copy(pop[p1]))
        else:
            selected.append(np.copy(pop[p2]))

    return selected

# implements the one-point crossover of two individuals
def one_pt_cross(p1, p2):
    point = random.randrange(1, len(p1))
    o1 = np.append(p1[:point], p2[point:])
    o2 = np.append(p2[:point], p1[point:])
    return o1, o2

def arithmetic_cross(p1, p2):
    alpha = np.random.uniform()
    o1 = alpha*p1 + (1-alpha)*p2
    o2 = alpha*p2 + (1-alpha)*p1
    return o1, o2

def differential_cross(p1, p2, CR):
    o1 = np.where(np.random.rand(len(p1)) < CR, p1, p2)
    o2 = np.where(np.random.rand(len(p1)) < CR, p2, p1)
    return o1, o2

# gaussian mutation - we need a class because we want to change the step
# size of the mutation adaptively
class BiasedMutation:

    def __init__(self, step_size, max_steps=None):
        self.step_size = step_size

    def __call__(self, ind, *args, **kwargs):
        return ind + self.step_size*np.random.normal(size=ind.shape)
    
class UnBiasedMutation:

    def __init__(self, step_size, max_steps=None):
        self.step_size = step_size

    def __call__(self, ind, *args, **kwargs):
        idx = np.random.choice(len(ind))
        result = ind.copy()
        result[idx] = np.random.normal(scale=result.std())
        return ind + self.step_size*np.random.normal(size=ind.shape)
    
class Biased_Adaptive_Mutation:

    def __init__(self, step_size, max_steps):
        self.step_size = step_size
        self.max_steps = max_steps

    def __call__(self, ind, *args, **kwargs):
        return ind + (self.step_size * ind / self.max_steps)*np.random.normal(size=ind.shape)
    
class Biased_std_Mutation:

    def __init__(self, step_size, max_steps):
        self.step_size = step_size
        self.max_steps = max_steps

    def __call__(self, ind, stds, *args, **kwargs):
        return ind + self.step_size * np.random.normal(size=ind.shape, scale=stds)
    
class Differential_Mutation:
    def __init__(self, F, n_others=2):
        self.F = F
        self.n_others = n_others
    def __call__(self, ind, pop, *args, **kwargs):
        pop = np.stack(pop)
        others = pop[np.random.choice(len(pop), self.n_others, replace=False)]
        border = self.n_others // 2
        return ind + self.F * (others[:border].mean(axis=0) - others[border:].mean(axis=0))
    
class Lamarckian_Mutation:
    def __init__(self, step_size, max_steps):
        self.step_size = step_size
        self.max_steps = max_steps
    def __call__(self, ind, fit, *args, **kwargs):
        for i in range(2):
            ind = ind - cf.numerical_derivative(fit, ind) * self.step_size
        return ind


# applies a list of genetic operators (functions with 1 argument - population) 
# to the population
def mate(pop, operators):
    for o in operators:
        pop = o(pop)
    return pop

# applies the cross function (implementing the crossover of two individuals)
# to the whole population (with probability cx_prob)
def crossover(pop, cross, cx_prob):
    off = []
    for p1, p2 in zip(pop[0::2], pop[1::2]):
        if random.random() < cx_prob:
            o1, o2 = cross(p1, p2)
        else:
            o1, o2 = p1[:], p2[:]
        off.append(o1)
        off.append(o2)
    return off

# applies the mutate function (implementing the mutation of a single individual)
# to the whole population with probability mut_prob)
def mutation(pop, mutate, mut_prob, fit):
    stds = np.std(pop, axis=0)
    return [mutate(p, stds=stds, pop=pop, fit=fit) if random.random() < mut_prob else p[:] for p in pop]

# implements the evolutionary algorithm
# arguments:
#   pop_size  - the initial population
#   max_gen   - maximum number of generation
#   fitness   - fitness function (takes individual as argument and returns 
#               FitObjPair)
#   operators - list of genetic operators (functions with one arguments - 
#               population; returning a population)
#   mate_sel  - mating selection (funtion with three arguments - population, 
#               fitness values, number of individuals to select; returning the 
#               selected population)
#   mutate_ind - reference to the class to mutate an individual - can be used to 
#               change the mutation step adaptively
#   map_fn    - function to use to map fitness evaluation over the whole 
#               population (default `map`)
#   log       - a utils.Log structure to log the evolution run
def evolutionary_algorithm(pop, max_gen, fitness, operators, mate_sel, mutate_ind, *, map_fn=map, log=None, diff=False, baldwin=False):
    evals = 0
    offspring = None
    for G in range(max_gen):
        fits_objs = list(map_fn(fitness, pop))
        evals += len(pop)
        if log:
            log.add_gen(fits_objs, evals)
        
        if baldwin:
            lammut = Lamarckian_Mutation(0.01, None)
            fits_objs = list(map_fn(fitness, lammut(pop, fitness)))
        fits = [f.fitness for f in fits_objs]
        objs = [f.objective for f in fits_objs]

        if not diff:
            mating_pool = mate_sel(pop, fits, len(pop))
            offspring = mate(mating_pool, operators)
            pop = offspring[:]
        else:
            if offspring is not None:
                pop = [max(p1, p2, key=fitness) for p1, p2 in zip(pop, offspring)]
            offspring = mate(pop, operators)

    return pop

def main(
    DIMENSION = 10, # dimension of the problems
    POP_SIZE = 100, # population size
    MAX_GEN = 500, # maximum number of generations
    CX_PROB = 0.8, # crossover probability
    MUT_PROB = 0.2, # mutation probability
    MUT_STEP = 0.5, # size of the mutation steps
    REPEATS = 10, # number of runs of algorithm (should be at least 10)
    OUT_DIR = 'continuous', # output directory for logs
    EXP_ID = 'default', # the ID of this experiment (used to create log names)),
    cross_name = "one_pt_cross", # the crossover function to use
    mutation_name = "BiasedMutation", # the mutation function to use
    F = 0.8,
    CR = 0.9,
    n_others = 2,
    diff = False,
    baldwin = False
):
    # use `functool.partial` to create fix some arguments of the functions 
    # and create functions with required signatures
    cr_ind = functools.partial(create_ind, ind_len=DIMENSION)
    # we will run the experiment on a number of different functions
    fit_generators = [cf.make_f01_sphere,
                      cf.make_f02_ellipsoidal,
                      cf.make_f06_attractive_sector,
                      cf.make_f08_rosenbrock,
                      cf.make_f10_rotated_ellipsoidal]
    fit_names = ['f01', 'f02', 'f06', 'f08', 'f10']

    for fit_gen, fit_name in zip(fit_generators, fit_names):
        fit = fit_gen(DIMENSION)

        if mutation_name == "Differential_Mutation":
            mutate_ind = eval(mutation_name)(F=F, n_others=n_others)
        else:
            mutate_ind = eval(mutation_name)(step_size=MUT_STEP, max_steps=MAX_GEN)
        
        if cross_name == "differential_cross":
            cross = functools.partial(eval(cross_name), CR=CR)
        else:
            cross = eval(cross_name)

        xover = functools.partial(crossover, cross=cross, cx_prob=CX_PROB)
        mut = functools.partial(mutation, mut_prob=MUT_PROB, mutate=mutate_ind, fit=fit)

        # run the algorithm `REPEATS` times and remember the best solutions from 
        # last generations
    
        best_inds = []
        for run in range(REPEATS):
            # initialize the log structure
            log = utils.Log(OUT_DIR, EXP_ID + '.' + fit_name , run, 
                            write_immediately=True, print_frequency=5)
            # create population
            pop = create_pop(POP_SIZE, cr_ind)
            # run evolution - notice we use the pool.map as the map_fn
            pop = evolutionary_algorithm(pop, MAX_GEN, fit, [xover, mut], tournament_selection, mutate_ind, map_fn=map, log=log,
                                         diff=diff, baldwin=baldwin)
            # remember the best individual from last generation, save it to file
            bi = max(pop, key=fit)
            best_inds.append(bi)
            
            # if we used write_immediately = False, we would need to save the 
            # files now
            # log.write_files()

        # print an overview of the best individuals from each run
        for i, bi in enumerate(best_inds):
            print(f'Run {i}: objective = {fit(bi).objective}')

        # write summary logs for the whole experiment
        utils.summarize_experiment(OUT_DIR, EXP_ID + '.' + fit_name)