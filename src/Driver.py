from datetime import datetime

from src.AlphaGeneticSolver.AlphaPopulation import AlphaPopulation
from src.FixedChargeNetwork.FixedChargeFlowNetwork import FixedChargeFlowNetwork

"""DRIVER PY FILE"""

# TEST OF RANDOM GRAPH GENERATOR
# graphGen = GraphGenerator("r1000-3", 1000, 0.03, 50, 50, [50, 200], [10, 50], [20, 100], [1, 10], [10, 50], 3)
# graphGen.saveFCFN()

# TEST OF FCFN
flowNetwork = FixedChargeFlowNetwork()
flowNetwork.loadFCFN("r2000-5(50,50)")

# WALL CLOCK TIME STAMP
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)

# TEST OF GA POPULATION
population = AlphaPopulation(flowNetwork, 2500, 1, 1)
population.evolvePopulation()

# WALL CLOCK TIME STAMP
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End GA/Start MILP Time =", current_time)

# TEST OF MILP
flowNetwork.executeSolver(2500)
flowNetwork.visualizeNetwork()

# WALL CLOCK TIME STAMP
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Finish Time =", current_time)
