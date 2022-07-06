
from src.Experiments.GraphSolver import GraphSolver

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 run_graph_solver.py
"""

if __name__ == "__main__":
    inputGraph = "small_2.p"
    graphSolver = GraphSolver(inputGraph, isSolvedWithGeneticAlg=True, isSolvedWithCPLEX=False, isRace=True,
                              isDrawing=True, isLabeling=True, isGraphing=True, isOutputtingCPLEX=False)
    graphSolver.solveGraph()
