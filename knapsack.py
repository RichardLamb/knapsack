import logging
import sys
import numpy as np
from numpy import random
import random as rnd

FILENAME = 'items.txt'
WEIGHT_CONSTRAINT = 15
POPULATION_SIZE = 10
ITERATIONS = 50
PROBABILITY_MUTATE = 0.5


class Gene:
    def __init__(self, value, weight, logger=None):
        """
        Initialises the Gene class, which represents an item which can form part of a solution (Chromosone)
        :param value: the value of the item
        :param weight: the weight of the item
        :param logger: the logger module
        """
        self.value = value
        self.weight = weight
        self.gene_id = 0
        self.logger = logger
        return


class Chromosone:

    def __init__(self, items=None, logger=None):
        """
        Initialises the Chromosone class, which is a solution comprising of a set of genes / items
        :param items: set of items
        :param logger:  the logger module
        """
        self.items = items
        self.gene_list = []
        self.logger = logger

        return

    @property
    def value(self):
        """
        :return: the knapsack's value, i.e. the sum of the item values
        """
        return int(sum([gene.value for gene in self.gene_list]))

    @property
    def weight(self):
        """
        :return: the knapsack's weight, i.e. the sum of the item weights
        """
        return int(sum([gene.weight for gene in self.gene_list]))

    @property
    def count_genes(self):
        """
        :return: the count of items in a given solution
        """
        return len(self.gene_list)

    @property
    def fitness_score(self):

        fitness_score = self.value

        if self.weight > WEIGHT_CONSTRAINT:
            fitness_score -= 1000

        return fitness_score

    def create_chromosone(self):
        """
        Creates a knapsack solution iteratively, by adding individual items selected at random
        subject to the weight constraint
        """

        item_set = self.items

        size_of_chromosone = random.randint(1, len(item_set)-1)

        self.gene_list = rnd.sample(self.items, size_of_chromosone)

        pass

    def crossover(self, other):
        """
        Creates a child solution given a second parent/solution
        :param other: the second chromosone/solution
        :return: child solution
        """

        split_position = random.randint(1, min(self.count_genes, other.count_genes) + 1)

        offspring = Chromosone(self.items, self.logger)

        offspring.gene_list = self.gene_list[:split_position] + other.gene_list[split_position:]

        return offspring


class Population:
    def __init__(self, size, items, logger=None):
        """
        Initialises the Population class, representing a set of solution
        :param size: the size of the population
        :param items: the set of possible items
        :param logger: the logger module
        """
        self.size = size
        self.chromosone_set = []
        self.answer_set = []
        self.items = items
        self.logger = logger

        return

    def create_population(self):
        """
        Creates a population sorted by value, by adding chromosones until the population size
        requirement is met
        :return: population/set of solutions
        """
        self.chromosone_set = [Chromosone(self.items, self.logger) for i in range(self.size)]

        for obj in self.chromosone_set:
            obj.create_chromosone()

        self.sort_by_value()

        if self.logger is not None:
            print('\nStart population: \n')

            for answer in self.answer_set:
                print(answer)

            print('\n' + '*' * 50)
        return

    def sort_by_value(self):
        """
        Sorts a population size by chomosone/solution value (highest first)
        :return: sorted list of chromosones
        """
        self.answer_set = [(obj, obj.value, obj.weight, obj.count_genes, obj.fitness_score) for obj in self.chromosone_set]
        self.chromosone_set.sort(key=lambda x: x.value, reverse=True)
        return self.answer_set.sort(key=lambda x: x[4], reverse=True)

    def select_best(self):
        """
        Chooses the 50% best chomosones/solutions from the population
        :return: sorted set of top solutions
        """
        self.answer_set = self.answer_set[:self.size // 2]
        self.chromosone_set = self.chromosone_set[:self.size // 2]

        if self.logger is not None:
            print('\nBest half of population: \n')

            for answer in self.answer_set:
                print(answer)

            print('\n' + '*' * 50)
        return

    def breeding(self):
        """
        Re-fills the population to the stipulated size by 'breeding' sets of existing solutions
        to create child solutions
        :return: population with breeding
        """
        parents = self.chromosone_set
        children = []
        children_needed = POPULATION_SIZE - len(parents)

        while len(children) < children_needed:
            male = parents[random.randint(0, len(parents) - 1)]
            female = parents[random.randint(0, len(parents) - 1)]

            offspring = male.crossover(female)

            if offspring.fitness_score > 0:
                children.append(offspring)

        self.chromosone_set.extend(children)

        self.sort_by_value()

        print('\nPopulation post breeding:\n')

        for answer in self.answer_set:
            print(answer)

        print('\n' + '*' * 50)
        return

    def mutate(self):
        """
        For a given population, randomly mutates items in its solution set
        :return: populations with mutations
        """
        for chromosone in self.chromosone_set:

            #  Mutate one gene in each chromosone with set probability
            if random.random() < PROBABILITY_MUTATE:

                #  Remove one randomly selected gene
                chromosone.gene_list.remove(random.choice(chromosone.gene_list))

                #  Add one gene randomly
                new_gene = self.items[random.randint(0, len(self.items) - 1)]

                chromosone.gene_list.append(new_gene)

                break

        self.sort_by_value()

        print('\nPopulation post mutating: \n')

        for answer in self.answer_set:
            print(answer)

        print('\n' + '*' * 50)
        return


def apply_crossover(population):
    """
    Applies the crossover to 2 given chromosones given a population
    :param population:
    :return: population
    """
    i = 0
    while i < len(population) - 1:
        population[i], population[i + 1] = population[i].crossover(population[1 + i])
        i += 1

    return population


def load_data(filename, logger=None):
    """
    Function loads item information from a given text file
    :param filename: Filename of data file
    :param logger: logger module
    :return: list of items with their wights and values
    """
    values = []
    items = np.genfromtxt(filename)

    for item in items:
        values.append(Gene(item[0], item[1], logger))

    if logger:
        logger.info('Extracted item data')
        logger.info('\n items:')
        for item in items:
            logger.info('weight:' + str(item[0]) + ' value:' + str(item[1]))

    return values


def main():
    """
    Iteratively applies different stages of generatic algorithm to find
    optimum solution to Knapsack problem (assuming WITH REPLACEMEMT)
    """
    logger = create_logger()

    logger.info('Iteration process initiated')

    try:
        items = load_data(FILENAME, logger)

        logger.info('Data loaded from file')

        population = Population(POPULATION_SIZE, items, logger)
        population.create_population()

        for i in range(ITERATIONS):
            print('-' * 66, '\n- Iteration ', i, '-' * 50, '\n' + '-' * 66)
            population.select_best()
            population.breeding()
            population.mutate()

        top_result = population.answer_set[0]

        logger.info('Final result calculated')

        print('\nFinal result is knapsack of value:', int(top_result[1]), ' and weight: ', int(top_result[2]))

        print('\nWith items:')

        for gene in top_result[0].gene_list:
            print('  Value: ', gene.value, 'and Weight: ', gene.weight)


    except:
        logger.exception('Got exception on main handler')


def create_logger():
    logger = logging.getLogger('Knapsack logger')
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler('knapsack.log')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


if __name__ == '__main__':
    main()

    # Testing data loading
    test_items = load_data(FILENAME)
    assert len(test_items) == 7

    # Test gene class properties
    test_genes = [Gene(i + 1, i + 1) for i in range(5)]
    assert (test_genes[0].value == 1 & test_genes[0].weight == 1)

    # Create chromosone instances for testing
    chromosone_test1 = Chromosone(test_items)
    for test_gene in test_genes:
        chromosone_test1.gene_list.append(test_gene)

    chromosone_test2 = Chromosone(test_items)
    chromosone_test2.create_chromosone()

    offspring_test = chromosone_test2.crossover(chromosone_test1)

    # Test crossover
    assert chromosone_test1.gene_list != chromosone_test2.gene_list
    assert offspring_test.gene_list != chromosone_test1.gene_list
    assert offspring_test.gene_list != chromosone_test2.gene_list
    assert len(chromosone_test1.gene_list) == len(offspring_test.gene_list)

    # Test population creation
    population_test = Population(POPULATION_SIZE, test_items)
    population_test.create_population()
    assert len(population_test.chromosone_set) == POPULATION_SIZE

    # Test sort by best
    population_test.sort_by_value()

    assert all(population_test.chromosone_set[i].value >= population_test.chromosone_set[i + 1].value
               for i in range(len(population_test.chromosone_set) - 1))

    # Test select best 50%
    population_test.select_best()
    assert len(population_test.chromosone_set) == POPULATION_SIZE / 2



