from src.AlphaGeneticSolver.Population import Population
from src.Network.FlowNetwork import FlowNetwork

network = FlowNetwork()
network = network.loadNetwork("test-2500-80-40.p")
pop = Population(network, 4200, 2)
pop.evolvePopulation(10)
