from src.ExactSolver.MILPsolverCPLEX import MILPsolverCPLEX
from src.Network.FlowNetwork import FlowNetwork

# Network Test
from src.Network.NetworkVisualizer import NetworkVisualizer
from src.Network.SolutionVisualizer import SolutionVisualizer

name = "test-6-1-1.p"
flowNetwork = FlowNetwork()
flowNetwork = flowNetwork.loadNetwork(name)
flowNetwork.drawNetworkTriangulation()

# Network Visualization Test
visualizer = NetworkVisualizer(flowNetwork, directed=True)
# visualizer.drawBidirectionalGraphWithSmoothedLabeledEdges()

# Solver Test
solver = MILPsolverCPLEX(flowNetwork, 95, isOneArcPerEdge=True)
solver.buildModel()
solver.solveModel()
solver.printAllSolverData()

# Solution Test
solution = solver.writeSolution()
solution.saveSolution()

# Solution Visualizer Test
solnVisualizer = SolutionVisualizer(solution)
solnVisualizer.drawUnlabeledGraph()
# solnVisualizer.drawGraphWithLabels()
