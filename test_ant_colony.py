import numpy as np

from Solvers.RelaxedLPSolverPDLP import RelaxedLPSolverPDLP
from src.Ant.Colony import Colony
from src.Network.FlowNetwork import FlowNetwork
from src.Network.SolutionVisualizer import SolutionVisualizer
from src.Solvers.MILPsolverCPLEX import MILPsolverCPLEX

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 test_ant_colony.py
"""

if __name__ == "__main__":
    # Load Network
    networkFile = "test.p"
    network = FlowNetwork()
    network = network.loadNetwork(networkFile)
    targetFlow = 50

    # Solve Exactly
    milpSolver = MILPsolverCPLEX(network, targetFlow, isOneArcPerEdge=False)
    milpSolver.buildModel()
    milpSolver.solveModel()
    milpSolver.printSolverOverview()
    exactSoln = milpSolver.writeSolution()
    exactVisualizer = SolutionVisualizer(exactSoln)
    exactVisualizer.drawGraphWithLabels()

    # Test Colony
    antColony = Colony(network, targetFlow, 10, 1)
    antSoln = antColony.solveNetwork(drawing=True, labels=False)
    antVisualizer = SolutionVisualizer(antSoln)
    antVisualizer.drawGraphWithLabels()

    # Solve with Naive LP Relaxation
    lpSolver = RelaxedLPSolverPDLP(network, targetFlow)
    alphaValues = np.full((network.numEdges, network.numArcCaps), 1.0)
    lpSolver.updateObjectiveFunction(alphaValues)
    lpSolver.solveModel()
    lpSolver.printSolverOverview()
    relaxedSoln = lpSolver.writeSolution()
    relaxedVisualizer = SolutionVisualizer(relaxedSoln)
    relaxedVisualizer.drawGraphWithLabels()
