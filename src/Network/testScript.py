from src.ExactSolver.MILPsolverCPLEX import MILPsolverCPLEX
from src.Network.FlowNetwork import FlowNetwork

name = "test-8-1-1.p"
flowNetwork = FlowNetwork()
flowNetwork = flowNetwork.loadNetwork(name)
flowNetwork.drawNetworkTriangulation()

solver = MILPsolverCPLEX(flowNetwork, 82)
solver.buildModel()
solver.solveModel()
solver.printAllSolverData()

# visualizer = NetworkVisualizer(flowNetwork)
# visualizer.drawGraph()
