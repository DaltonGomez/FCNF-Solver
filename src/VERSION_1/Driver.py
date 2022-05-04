from datetime import datetime

from src.VERSION_1.AlphaGeneticSolver.AlphaPopulation import AlphaPopulation
from src.VERSION_1.FixedChargeNetwork.FixedChargeFlowNetwork import FixedChargeFlowNetwork

"""DRIVER PY FILE"""

# TEST OF RANDOM GRAPH GENERATOR
# graphGen = GraphGenerator("r1000-3", 1000, 0.03, 50, 50, [50, 200], [10, 50], [20, 100], [1, 10], [10, 50], 3)
# graphGen.saveFCFN()

# TEST OF FCFN
flowNetwork = FixedChargeFlowNetwork()
flowNetwork.loadFCFN("small")

# WALL CLOCK TIME STAMP
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)

# TEST OF GA POPULATION
population = AlphaPopulation(flowNetwork, 32, 2, 1)
population.evolvePopulation()

"""
# TEST OF ALPHA SOLVER
ga = AlphaPopulation(flowNetwork, 2500, 2, 1)
ai = AlphaIndividual(flowNetwork)
ai.initializeAlphaValuesRandomly(0.0, 1.0)
ga.population.append(ai)
ga.solveIndividual(ga.population[0])
"""

# WALL CLOCK TIME STAMP
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End GA/Start MILP Time =", current_time)

# TEST OF MILP
# flowNetwork.executeSolver(35)
# flowNetwork.visualizeNetwork()

# WALL CLOCK TIME STAMP
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Finish Time =", current_time)
