from datetime import datetime

from AlphaGenetic.Population import Population
from FlowNetwork.SolutionVisualizer import SolutionVisualizer
from Graph.CandidateGraph import CandidateGraph
from Solvers.MILPsolverCPLEX import MILPsolverCPLEX


class GeneticRunner:
    """Class that executes a GA population's evolution (with the option to solve the MILP optimally as well)"""
    # TODO - Revise the implementation of this class to wrap a single run of the GA algo.

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self, inputGraph: str):
        """Constructor of a GeneticRunner instance"""
        # Input graph attributes
        self.graphName: str = inputGraph
        self.graph: CandidateGraph = CandidateGraph()
        self.graph.loadCandidateGraph(self.graphName)
        self.minTargetFlow: float = self.graph.totalPossibleDemand
        # Population and hyperparameter attributes
        self.population: Population = Population(self.graph, self.minTargetFlow)
        self.population.setPopulationHyperparams(populationSize=10,
                                                 numGenerations=10,
                                                 initializationStrategy="perEdge",
                                                 initializationDistribution="digital",
                                                 initializationParams=[0.0, 200000.0])
        self.population.setIndividualSelectionHyperparams(selectionMethod="tournament",
                                                          tournamentSize=3)
        self.population.setCrossoverHyperparams(crossoverMethod="onePoint",
                                                replacementStrategy="replaceWeakestTwo",
                                                crossoverRate=1.0,
                                                crossoverAttemptsPerGeneration=3)
        self.population.setMutationHyperparams(mutationMethod="randomPerEdge",
                                               mutationRate=0.05,
                                               perArcEdgeMutationRate=0.20)
        # Output Attributes
        # TODO - Add Output Attributes

    def setHyperparams(self) -> None:
        """TODO - Add docs"""

    def runGA(self, solveMILP=True) -> None:
        """TODO - Add docs"""
        # Timestamp the start of the GA evolution
        gaStartTime = datetime.now()
        print("\n\nSolving the " + self.graphName + " graph with a GA population of " +
              str(self.population.populationSize) + " for " + str(self.population.numGenerations) + " generations...")
        print("GA Start: " + str(gaStartTime) + "\n")
        # Solve the Alpha-GA
        solutionTuple = self.population.evolvePopulation(printGenerations=True, drawing=True, drawLabels=True)
        print("\nBest solution found = " + str(solutionTuple[0]))
        solVis = SolutionVisualizer(solutionTuple[1])
        solVis.drawLabeledSolution(leadingText="GA_best_")
        # Timestamp the finish of the GA evolution/start of the optimal MILP solver
        gaFinishTime = datetime.now()
        print("\nGA Finish: " + str(gaFinishTime))
        gaRuntime = gaFinishTime - gaStartTime
        print("GA Runtime in Minutes: " + str(gaRuntime.seconds / 60))
        if solveMILP is True:
            optStartTime = datetime.now()
            print("\n\nOPT Start: " + str(optStartTime))
            # Solve Optimally with CPLEX
            cplex = MILPsolverCPLEX(self.graph, self.minTargetFlow, isOneArcPerEdge=False)
            cplex.findSolution(printDetails=True)
            opt = cplex.writeSolution()
            optVis = SolutionVisualizer(opt)
            optVis.drawLabeledSolution(leadingText="OPT_")
            # Timestamp the finish of the optimal MILP solver
            optFinishTime = datetime.now()
            print("\nOPT Finish: " + str(optFinishTime))
            optRuntime = optFinishTime - gaStartTime
            print("OPT Runtime in Minutes: " + str(optRuntime.seconds / 60))
        print("\n\nProgram complete! Terminating...")
