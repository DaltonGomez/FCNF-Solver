import random

import numpy as np

from src.AlphaGeneticSolver.AlphaSolverPDLP import AlphaSolverPDLP
from src.Network.FlowNetwork import FlowNetwork
from src.Network.NetworkVisualizer import NetworkVisualizer
from src.Network.SolutionVisualizer import SolutionVisualizer

network = FlowNetwork()
network = network.loadNetwork("test-6-1-1.p")
netVis = NetworkVisualizer(network, directed=True)
netVis.drawBidirectionalGraphWithSmoothedLabeledEdges()
solver = AlphaSolverPDLP(network, 86)
random.seed()
alphaList = []
for e in range(network.numEdges):
    thisEdge = []
    for c in range(network.numArcCaps):
        thisEdge.append(random.random())
    alphaList.append(thisEdge)
alphas = np.array(alphaList)
solver.updateObjectiveFunction(alphas)
solver.solveModel()
solver.printSolverOverview()
solution = solver.writeSolution()
solVis = SolutionVisualizer(solution)
solVis.drawGraphWithLabels()
