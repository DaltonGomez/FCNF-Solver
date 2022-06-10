import numpy as np

from src.AntColony.Colony import Colony
from src.Network.FlowNetwork import FlowNetwork
from src.Network.SolutionVisualizer import SolutionVisualizer
from src.Solvers.MILPsolverCPLEX import MILPsolverCPLEX
from src.Solvers.RelaxedLPSolverPDLP import RelaxedLPSolverPDLP

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 run_ant_colony.py
"""

if __name__ == "__main__":
    # Load Network
    networkFile = ".p"
    network = FlowNetwork()
    network = network.loadNetwork(networkFile)
    targetFlow = 40

    # Solve Exactly
    milpSolver = MILPsolverCPLEX(network, targetFlow, isOneArcPerEdge=False)
    milpSolver.buildModel()
    milpSolver.solveModel()
    milpSolver.printSolverOverview()
    exactSoln = milpSolver.writeSolution()
    exactVisualizer = SolutionVisualizer(exactSoln)
    exactVisualizer.drawGraphWithLabels()

    # Solve with Naive LP Relaxation
    lpSolver = RelaxedLPSolverPDLP(network, targetFlow)
    alphaValues = np.full((network.numEdges, network.numArcCaps), 1.0)
    lpSolver.updateObjectiveFunction(alphaValues)
    lpSolver.solveModel()
    lpSolver.printSolverOverview()
    relaxedSoln = lpSolver.writeSolution()
    relaxedVisualizer = SolutionVisualizer(relaxedSoln)
    relaxedVisualizer.drawGraphWithLabels()

    # Test Colony
    antColony = Colony(network, targetFlow, 30, 10)
    antSoln = antColony.solveNetwork(drawing=True, labels=False)
    antVisualizer = SolutionVisualizer(antSoln)
    antVisualizer.drawGraphWithLabels()
