from src.AlphaGenetic.Population import Population
from src.Network.FlowNetwork import FlowNetwork
from src.Network.SolutionVisualizer import SolutionVisualizer
from src.Solvers.MILPsolverCPLEX import MILPsolverCPLEX

network = FlowNetwork()
network = network.loadNetwork("test-250-10-10.p")
minTargetFlow = 1200

pop = Population(network, minTargetFlow, 10)
pop.evolvePopulation(25, drawing=True)

cplex = MILPsolverCPLEX(network, minTargetFlow, isOneArcPerEdge=False)
cplex.buildModel()
cplex.solveModel()
opt = cplex.writeSolution()

vis = SolutionVisualizer(opt)
vis.drawUnlabeledGraph(leadingText="OPT")
