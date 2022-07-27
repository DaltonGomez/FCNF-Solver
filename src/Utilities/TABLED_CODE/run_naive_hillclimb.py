from src.AlphaGenetic.Population import Population
from src.Graph.CandidateGraph import CandidateGraph

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 run_naive_hillclimb.py
"""

if __name__ == "__main__":
    # Load FlowNetwork
    graph = CandidateGraph()
    graph = graph.loadCandidateGraph("massive_2.p")
    minTargetFlow = graph.totalPossibleDemand

    # Solve with Naive Hill Climb
    hillClimb = Population(graph, minTargetFlow)
    hillClimb.solveWithNaiveHillClimb(printGenerations=True, drawing=True, drawLabels=True)
