import numpy as np
import random
from random import choices
import time
import copy
import warnings
import sys
import statistics
import matplotlib.pyplot as plt


class Ga:
    def __init__(self, population_size, string_length, k, d, crossover_operator, isTrap, isTightlyLinked,
                 isFinalExperiment=False):
        """ Instatiates the ga object.

            :param population_size:  the number of vectors in initial population -> type=integer
            :param string_length: the length of each vector in the population -> type=integer, important: only even numbers
            :param k: the k parameter used in trap functions -> type=integer, important: string_length must be divisible by k
            :param d: the d parameter as used in trap fucntions -> type=float
            :param crossover_operator: the type of crossover to be used in reproduction -> type=string, '2X': 2-point crossover used, 'UX': Uniform crossover used
            :param isTrap: flag defining if fitness function is a trap function -> type=boolean, True: Trap function used, False: Counting-ones function used
            :param isTigthlyLinked: flag defining if trap operation is tighly linked or not -> type=boolean, True: Tightly linked, False: Not tightly linked
        """

        def create_starting_population(population_size, string_length):
            """ Creates the starting population for each experiment.

                :param population_size: the number of strings in initial population -> type=integer
                :param string_length: the length of each string to be created -> type=integer
                :return: a list of size=population_size containing strings of length=string_length
            """

            population = []
            # Create as many string as "population_size"
            for _ in range(population_size):
                temp_string = ''
                # Create a string of length="string_length" of random charactes (0 or 1)
                for _ in range(string_length):
                    temp_string += choices(['0', '1'])[0]
                population.append(temp_string)  # Append string to population

            return population

        self.k = k
        self.d = d
        self.crossover_operator = crossover_operator
        self.isTightlyLinked = isTightlyLinked
        self.isTrap = isTrap
        self.string_length = string_length
        self.population = create_starting_population(population_size, string_length)
        self.total_fails = 0
        self.total_funcs = 0
        self.generation_fitness = 0
        self.fitness_evals = 0
        self.number_of_generations = 0
        self.isFinalExperiment = isFinalExperiment
        self.ones_per_generation = []
        self.selection_error = []
        self.selection_correct = []
        self.one_fitness = []
        self.one_std = []
        self.zero_fitness = []
        self.zero_std = []

    def counting_ones_function(self, functions):
        """ Calculates vector fitness based on counting-ones function.

            :param population: the collection of vectors -> type=list
            :return: a list with fitness values corresponding to population
        """
        self.fitness_evals += 1

        return [vec.count('1') for vec in functions]

    def trap_function(self, functions):
        """ Calculates vector fitness based on trap function.

            :param population: the collection of vectors -> type=list
            :param isTigthlyLinked: flag defining if trap operation is tighly linked or not -> type=boolean, True: Tightly linked, False: Not tightly linked
            :param k: the k parameter used in trap functions -> type=integer, important: string_length must be divisible by k
            :param d: the d parameter as used in trap fucntions -> type=float
            :return: a list with fitness values corresponding to population
        """
        self.fitness_evals += 1
        fitness_scores = []
        # print(functions)
        for vec in functions:
            sub_functions = []
            if (self.isTightlyLinked):
                for i in range(0, len(vec) - self.k + 1, self.k):
                    sub_functions.append(vec[i:i + self.k])
            else:
                for i in range(int(len(vec) / self.k)):
                    temp_vec = ''
                    tempvec = vec[i::int(len(vec) / 4)]
                    sub_functions.append(temp_vec)

            # print(sub_functions)
            vec_fitness = 0
            for sub_function in sub_functions:
                fit_val = self.k - self.d - (((self.k - self.d) / (self.k - 1)) * sub_function.count('1'))
                if fit_val < 0:
                    fit_val = self.k
                # print(sub_function, fit_val)

                vec_fitness += fit_val
            fitness_scores.append(vec_fitness)

        # print(list(zip(functions, fitness_scores)))
        return fitness_scores

    def two_point_crossover(self, parent_1, parent_2):

        string_length = len(parent_1)

        # Pick random crossover points
        crossover_point1 = random.randint(1, string_length - 2)
        crossover_point2 = random.randint(crossover_point1 + 1, string_length - 1)

        # Create children by swapping bit string in between crossover points
        child_1 = parent_1[:crossover_point1] + parent_2[crossover_point1:crossover_point2] + parent_1[
                                                                                              crossover_point2:]
        child_2 = parent_2[:crossover_point1] + parent_1[crossover_point1:crossover_point2] + parent_2[
                                                                                              crossover_point2:]

        # print('Crossover:', parent_1, parent_2, 'at', crossover_point1, crossover_point2, '->', child_1, child_2)

        return child_1, child_2

    def uniform_crossover(self, parent_1, parent_2):

        string_length = len(parent_1)
        child_1 = child_2 = ""

        for i in range(string_length):
            # Children inherit bits that parents agree on
            if random.random() < 0.5:
                child_1 += parent_1[i]
                child_2 += parent_2[i]
            else:
                child_1 += parent_2[i]
                child_2 += parent_1[i]

        # print('Crossover:', parent_1, parent_2, '->', child_1, child_2)

        return child_1, child_2

    def check_stopping_criterion(self, fitness_values):

        parent1_fitness, parent2_fitness, child1_fitness, child2_fitness = fitness_values

        max_parent_fitness = max(parent1_fitness, parent2_fitness)
        if child1_fitness > max_parent_fitness or child2_fitness > max_parent_fitness:
            return True

    def found_global_optimum(self):
        for string in self.population:
            if '0' not in string:
                return True

        return False

    def step_generation(self):
        if self.isFinalExperiment:
            self.selection_error.append(0)
            self.selection_correct.append(0)
            self.one_fitness.append(0)
            self.one_std.append(0)
            self.zero_fitness.append(0)
            self.zero_std.append(0)
        errorFlag = True
        num_fitnessfunc = 0
        population_size = len(self.population)
        new_population = []
        self.number_of_generations += 1

        # Step 1: Shuffle list
        random.shuffle(self.population)
        # Step 2: Pair functions
        for i in range(0, population_size - 1, 2):
            parent_1 = self.population[i]
            parent_2 = self.population[i + 1]
            # Step 3: Create offspring using crossover
            if self.crossover_operator == 'UX':  # Uniform crossover
                child_1, child_2 = self.uniform_crossover(parent_1, parent_2)
            elif self.crossover_operator == '2X':  # Two-point crossover
                child_1, child_2 = self.two_point_crossover(parent_1, parent_2)
            else:
                sys.exit("Use '2X' for two-point crossover and 'UX' for uniform crossover")

            # Step 4: Family competition
            family = [parent_1, parent_2, child_1, child_2]
            if not self.isTrap:
                family_fitness = self.counting_ones_function(family)
            else:
                family_fitness = self.trap_function(family)
            if self.check_stopping_criterion(family_fitness):
                errorFlag = False

            family_fitness[-1] += 0.001
            family_fitness[-2] += 0.001
            best_two = list(zip(family, family_fitness))
            # print('Familiy fitness:', best_two)
            best_two.sort(key=lambda x: x[1], reverse=True)
            new_population += list(zip(*best_two))[0][:2]
            # print('New population:', new_population)
            # print(parent_1, parent_2, child_1, child_2, best_two[0][0], best_two[1][0])
            if self.isFinalExperiment:
                for p1, p2, b1, b2 in zip(parent_1, parent_2, best_two[0][0], best_two[1][0]):
                    if p1 != p2:
                        if b1 == '1' and b2 == '1':
                            self.selection_correct[self.number_of_generations - 1] += 1
                        elif b1 == '0' and b2 == '0':
                            self.selection_error[self.number_of_generations - 1] += 1

            # print((parent_1, parent_2, child_1, child_2, best_two[0][0], best_two[1][0]))
        if self.isFinalExperiment:
            temp1 = []
            temp2 = []
            self.ones_per_generation.append(sum([sample.count('1') for sample in self.population]))
            for sample in self.population:

                if sample.startswith("1"):
                    temp1.append(sample)
                else:
                    temp2.append(sample)
            temp1 = self.counting_ones_function(temp1)
            temp2 = self.counting_ones_function(temp2)
            self.one_fitness[self.number_of_generations - 1] = statistics.mean(temp1)
            self.one_std[self.number_of_generations - 1] = statistics.stdev(temp1)
            self.zero_fitness[self.number_of_generations - 1] = statistics.mean(temp2)
            self.zero_std[self.number_of_generations - 1] = statistics.stdev(temp2)
        # Replace the old population with the new one
        self.population = new_population

        # joined_population = ''.join(self.population)
        # self.generation_fitness = joined_population.count('1') / len(joined_population) * 100
        # print(self.generation_fitness)
        # print('No progress:', errorFlag)

        if self.found_global_optimum():
            return 'opt'

        if errorFlag:
            return 'err'

        return 0

    def run_ga(self):

        generation_fail_counter = 0
        generation_count = 0
        while True:
            generation_count += 1
            result = self.step_generation()
            print(''.join(self.population).count('1') / len(self.population))
            print(self.population)
            # print('Generation:', generation_count, '->', result)
            # print('P', self.population)
            if result == 0:
                generation_fail_counter = 0
                continue
            if result == 'opt':
                # print('Global optimum found at N =', len(self.population))
                return 'opt'
            if result == 'err':
                generation_fail_counter += 1
                # print('gen fail counter: ', generation_fail_counter)
                if generation_fail_counter >= 5:
                    # print('No progress for 5 generations')
                    return 'err'



def run_experiment(k, d, isTrap, isTightlyLinked, crossover_operator, start_population_size=10, string_length=40,
                   runs=20):
    run_fail_counter = 0
    history_log = []

    population_size = start_population_size
    successes = 0
    failures = 0
    population_history = [population_size]
    notfail = True

    opt_pop = None
    opt_gen = []
    opt_eva = []
    opt_cpu = []

    temp_pop = None
    temp_gen = []
    temp_eva = []
    temp_cpu = []

    while True:
        for _ in range(20):
            start = time.time()
            genetic_algorithm = Ga(population_size=population_size, string_length=string_length, k=k, d=d,
                                   crossover_operator=crossover_operator, isTrap=isTrap,
                                   isTightlyLinked=isTightlyLinked)
            result = genetic_algorithm.run_ga()
            temp_eva.append(genetic_algorithm.fitness_evals)
            temp_gen.append(genetic_algorithm.number_of_generations)
            if result == 'err':
                failures += 1
                print(failures, " failures with popsize: ", population_size)
                if failures == 2:
                    break
            if result == 'opt':
                successes += 1
                print(successes, " successes with popsize: ", population_size)
            end = time.time()
            temp_cpu.append(end - start)

        if failures >= 2:
            population_size *= 2
            population_history.append(population_size)
            temp_eva = []
            temp_gen = []
            temp_cpu = []
            failures = 0
            successes = 0
            print("Population size doubled to: ", population_size)
            if population_size > 1280:
                print("FAIL, ", population_size, " exceeds the maximum allowed")
                notfail = False
                population_history.append("FAIL")
                break
        elif successes >= 19:
            failures = 0
            successes = 0
            opt_pop = population_size
            opt_gen = temp_gen
            opt_eva = temp_eva
            opt_cpu = temp_cpu
            temp_eva = []
            temp_gen = []
            temp_cpu = []
            break

    if (notfail):
        max = population_history[-1]
        min = population_history[-2]
        population_size = int((max + min) / 2)
        population_history.append(population_size)

        while True:
            for _ in range(20):
                start = time.time()
                population_size = int((max + min) / 2)
                genetic_algorithm = Ga(population_size=population_size, string_length=string_length, k=k, d=d,
                                       crossover_operator=crossover_operator, isTrap=isTrap,
                                       isTightlyLinked=isTightlyLinked)
                result = genetic_algorithm.run_ga()
                temp_eva.append(genetic_algorithm.fitness_evals)
                temp_gen.append(genetic_algorithm.number_of_generations)

                # print(result)
                if result == 'err':
                    failures += 1
                    if failures == 2:
                        break
                if result == 'opt':
                    successes += 1
                end = time.time()
                temp_cpu.append(end - start)

            if failures >= 2:
                temp_eva = []
                temp_gen = []
                temp_cpu = []
                failures = 0
                successes = 0
                min = population_size
                population_history.append(int((max + min) / 2))
                print("Experiment with: ", population_size, " failed")
                if (max + min) / 2 % 10 != 0:
                    break
            elif successes >= 19:
                failures = 0
                successes = 0
                max = population_size
                opt_pop = population_size
                opt_gen = temp_gen
                opt_eva = temp_eva
                opt_cpu = temp_cpu
                temp_eva = []
                temp_gen = []
                temp_cpu = []
                population_history.append(int((max + min) / 2))
                print("Experiment with: ", population_size, " succeded")
                if (max + min) / 2 % 10 != 0:
                    break

        print("optimal population", opt_pop)
        print("len ", len(opt_gen), " opt pop number of generations ", opt_gen)
        print("len ", len(opt_eva), " opt pop number of evaluations ", opt_eva)
        print("len ", len(opt_cpu), " opt pop cpu time per run ", opt_cpu)
    return opt_pop, opt_gen, opt_eva, opt_cpu


def main():
    random.seed(2)
    experiment_results = []


    # Experiment 1
    print('\nExperiment 1...')
    experiment_results.append(run_experiment(k=4, d=1, isTrap=False, isTightlyLinked=True, crossover_operator='2X'))
    experiment_results.append(run_experiment(k=4, d=1, isTrap=False, isTightlyLinked=True, crossover_operator='UX'))

    # Experiment 2
    print('\nExperiment 2...')
    experiment_results.append(run_experiment(k=4, d=1, isTrap=True, isTightlyLinked=True, crossover_operator='2X'))
    experiment_results.append(run_experiment(k=4, d=1, isTrap=True, isTightlyLinked=True, crossover_operator='UX'))

    # Experiment 3
    print('\nExperiment 3...')
    experiment_results.append(run_experiment(k=4, d=2.5, isTrap=True, isTightlyLinked=True, crossover_operator='2X'))
    experiment_results.append(run_experiment(k=4, d=2.5, isTrap=True, isTightlyLinked=True, crossover_operator='UX'))

    # Experiment 4
    print('\nExperiment 4...')
    experiment_results.append(run_experiment(k=4, d=1, isTrap=True, isTightlyLinked=False, crossover_operator='2X', start_population_size=2, string_length=4))
    sys.exit()
    experiment_results.append(run_experiment(k=4, d=1, isTrap=True, isTightlyLinked=False, crossover_operator='UX'))

    # Experiment 5
    print('\nExperiment 5...')
    experiment_results.append(run_experiment(k=4, d=2.5, isTrap=True, isTightlyLinked=False, crossover_operator='2X'))
    experiment_results.append(run_experiment(k=4, d=2.5, isTrap=True, isTightlyLinked=False, crossover_operator='UX'))
    genetic_algorithm = Ga(population_size=200, string_length=40, k=0, d=0, crossover_operator="UX", isTrap=False,
                           isTightlyLinked=True, isFinalExperiment=True)
    result = genetic_algorithm.run_ga()

    plt.plot(genetic_algorithm.ones_per_generation)
    plt.show()

    plt.plot(genetic_algorithm.selection_correct)
    plt.plot(genetic_algorithm.selection_error)
    plt.show()

    plt.plot(genetic_algorithm.one_fitness)
    plt.plot(genetic_algorithm.zero_fitness)
    plt.show()

    plt.plot(genetic_algorithm.one_std)
    plt.plot(genetic_algorithm.zero_std)
    plt.show()

    print(genetic_algorithm.one_fitness)
    print(genetic_algorithm.one_std)
    print(genetic_algorithm.zero_fitness)
    print(genetic_algorithm.zero_std)
    print(statistics.mean(genetic_algorithm.one_fitness))
    print(statistics.mean(genetic_algorithm.zero_fitness))

    i = 0
    for experiment in experiment_results:
        i += 1
        print("Experiment ", i)
        if experiment[0] == None:
            print("FAIL")
            continue
        print("optimal population", experiment[0])
        print("optimal population number of generations ", experiment[1], "mean ", statistics.mean(experiment[1]),
              "std ", statistics.stdev(experiment[1]))
        print("optimal population number of evaluations ", experiment[2], "mean ", statistics.mean(experiment[1]),
              "std ", statistics.stdev(experiment[2]))
        print("optimal population cpu time per run ", experiment[3], "mean ", statistics.mean(experiment[1]), "std ",
              statistics.stdev(experiment[3]))


main()