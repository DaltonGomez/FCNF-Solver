from src.ExactSolver.MILPsolverCPLEX import MILPsolverCPLEX
from src.Network.FlowNetwork import FlowNetwork
from src.Network.NetworkVisualizer import NetworkVisualizer

name = "test-6-1-1.p"
flowNetwork = FlowNetwork()
flowNetwork = flowNetwork.loadNetwork(name)
flowNetwork.drawNetworkTriangulation()

solver = MILPsolverCPLEX(flowNetwork, 82, isOneArcPerEdge=True)
solver.buildModel()
solver.solveModel()
solver.printAllSolverData()

visualizer = NetworkVisualizer(flowNetwork, directed=True)
visualizer.drawBidirectionalGraphWithSmoothedLabeledEdges()
