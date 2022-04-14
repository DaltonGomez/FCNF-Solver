from src.AlphaGeneticSolver.Population import Population
from src.ExactSolver.MILPsolverCPLEX import MILPsolverCPLEX
from src.Network.FlowNetwork import FlowNetwork
from src.Network.SolutionVisualizer import SolutionVisualizer

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
