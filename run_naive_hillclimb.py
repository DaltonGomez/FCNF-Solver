from src.AlphaGenetic.Population import Population
from src.Network.FlowNetwork import FlowNetwork

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 run_naive_hillclimb.py
"""

if __name__ == "__main__":
    # Load Network
    network = FlowNetwork()
    network = network.loadNetwork("basic_5.p")
    minTargetFlow = network.totalPossibleDemand

    # Solve with Naive Hill Climb
    hillClimb = Population(network, minTargetFlow)
    hillClimb.solveWithNaiveHillClimb(printGenerations=True, drawing=True, drawLabels=True)
