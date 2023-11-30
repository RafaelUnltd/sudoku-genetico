import pandas as pd
import numpy as np

CROSSOVER_RATE = 0.75
MAX_ITERATIONS = 500
MUTATION_RATE = 0.15
POPULATION_SIZE = 200
HALF_POPULATION_SIZE = 100

BOUNDS = [[0,1,2],[3,4,5],[6,7,8]]
MIN_VALUE = 1
MAX_VALUE = 9

def calculateSolutionError(solution):
    errors = 0
    for i in range(len(solution)):
        for j in range(len(solution)):
            currentValue = solution[i][j]
            for k in range(len(solution)):
                if solution[k][j] == currentValue and k != i:
                    errors += 1
            for k in range(len(solution)):
                if solution[i][k] == currentValue and k != j:
                    errors += 1
            
            # Check for errors in the 3x3 matrix
            verticalBounds = BOUNDS[i//3]
            horizontalBounds = BOUNDS[j//3]

            for v in verticalBounds:
                for h in horizontalBounds:
                    if solution[v][h] == currentValue and v != i and h != j:
                        errors += 1
    return errors

def calculateHeuristicInfo(error):
    return 1 / error

def evaluatePopulation(population):
    evaluation = []
    for individual in population:
        error = calculateSolutionError(individual)
        evaluation.append(error)
    return evaluation

def executeRouletteWheel(probabilities):
    randomNumber = np.random.uniform(0, 1)

    for individual in range(len(probabilities)):
        if randomNumber <= probabilities[individual]:
            return individual
        else:
            randomNumber -= probabilities[individual]

    return len(probabilities) - 1

def defineParents(population, evaluation):
    sumEval = sum(evaluation)
    probabilities = []
    for i in range(len(evaluation)):
        probabilities.append(evaluation[i] / sumEval)

    parents = []
    for _ in range(HALF_POPULATION_SIZE):
        parent1 = population[executeRouletteWheel(probabilities)]
        parent2 = population[executeRouletteWheel(probabilities)]
        parents.append((parent1, parent2))

    return parents

def crossover(sudoku, parents, mask):
    children = []
    for parent1, parent2 in parents:
        if np.random.uniform(0, 1) <= CROSSOVER_RATE:
            child1 = []
            child2 = []
            for i in range(len(mask)):
                child1.append([])
                child2.append([])
                for j in range(len(mask)):
                    if mask[i][j] == 0:
                        child1[i].append(sudoku[i][j])
                        child2[i].append(sudoku[i][j])
                    elif mask[i][j] == 1:
                        child1[i].append(parent2[i][j])
                        child2[i].append(parent1[i][j])
                    else:
                        child1[i].append(parent1[i][j])
                        child2[i].append(parent2[i][j])
            children.append(child1)
            children.append(child2)
        else:
            children.append(parent1)
            children.append(parent2)
    return children

def mutate(children, mask):
    for child in children:
        for i in range(len(child)):
            for j in range(len(child)):
                if mask[i][j] != 0:
                    if np.random.uniform(0, 1) <= MUTATION_RATE:
                        child[i][j] = np.random.randint(MIN_VALUE, MAX_VALUE+1)
    return children

def getCrossoverMask(sudoku):
    mask = []
    number = 1
    for i in range(len(sudoku)):
        mask.append([])
        for j in range(len(sudoku)):
            if sudoku[i][j] != 0:
                mask[i].append(0)
            else:
                mask[i].append(number)
                if number == 1:
                    number = 2
                else:
                    number = 1
    return mask

def getInitialPopulation(sudoku, mask):
    population = []
    for _ in range(POPULATION_SIZE):
        individual = []
        for i in range(len(mask)):
            individual.append([])
            for j in range(len(mask)):
                if mask[i][j] == 0:
                    individual[i].append(sudoku[i][j])
                else:
                    individual[i].append(np.random.randint(MIN_VALUE, MAX_VALUE+1))
        population.append(individual)
    return population

def checkErrorZero(evaluation):
    for i in range(len(evaluation)):
        if evaluation[i] == 0:
            return i
    return -1

def main():
    sudoku = pd.read_csv('problems/1.csv', header=None).values
    mask = getCrossoverMask(sudoku)
    population = getInitialPopulation(sudoku, mask)
    evaluation = evaluatePopulation(population)
    leastError = min(evaluation)
    leastErrorIndex = evaluation.index(leastError)

    indexSolution = checkErrorZero(evaluation)
    if indexSolution != -1:
        print("Solução encontrada!")
        print(population[indexSolution])
        return
    
    print('Iteração inicial, com menor erro: ', leastError)

    iterations = 0
    while True:
        iterations += 1

        # Select parents
        parents = defineParents(population, evaluation)
        # Crossover
        children = crossover(sudoku, parents, mask)
        # Mutation
        children = mutate(children, mask)
        # Evaluate
        newEval = evaluatePopulation(children)

        indexSolution = checkErrorZero(newEval)
        if indexSolution != -1:
            print("Solução encontrada!")
            print(children[indexSolution])
            return
        
        # Elitism
        individualToPersist = population[leastErrorIndex]
        individualToLeave = newEval.index(max(newEval))
        children[individualToLeave] = individualToPersist
        newEval[individualToLeave] = leastError

        if min(newEval) < leastError:
            leastError = min(newEval)
            leastErrorIndex = newEval.index(leastError)

        print('Iteração ', iterations, ', com menor erro: ', leastError)

        population = children
        evaluation = newEval

        if iterations >= MAX_ITERATIONS:
            break

    print("\nMelhor solução encontrada:")
    result = pd.read_csv('results/1.csv', header=None).values
    parity = 81
    for i in range(len(population[leastErrorIndex])):
        print(population[leastErrorIndex][i])
        for j in range(len(population[leastErrorIndex])):
            if population[leastErrorIndex][i][j] != result[i][j]:
                parity -= 1

    print("\nParidade com a solução esperada: ", parity, "/81")

main()
