from src.Network.FlowNetwork import FlowNetwork
# Network Test
from src.Network.NetworkVisualizer import NetworkVisualizer
from src.Network.SolutionVisualizer import SolutionVisualizer
from src.Solvers.MILPsolverCPLEX import MILPsolverCPLEX

name = "25-1-1.p"
flowNetwork = FlowNetwork()
flowNetwork = flowNetwork.loadNetwork(name)

# Network Visualization Test
visualizer = NetworkVisualizer(flowNetwork, directed=True, supers=False)
visualizer.drawBidirectionalGraph()

# Solver Test
solver = MILPsolverCPLEX(flowNetwork, 100, isOneArcPerEdge=False)
solver.buildModel()
solver.solveModel()
solver.printAllSolverData()

# Solution Test
solution = solver.writeSolution()
solution.saveSolution()

# Solution Visualizer Test
solnVisualizer = SolutionVisualizer(solution)
# solnVisualizer.drawUnlabeledGraph()
solnVisualizer.drawGraphWithLabels()
