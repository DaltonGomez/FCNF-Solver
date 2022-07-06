from datetime import datetime

from src.AlphaGenetic.Population import Population
from src.FlowNetwork.SolutionVisualizer import SolutionVisualizer
from src.Graph.CandidateGraph import CandidateGraph
from src.Solvers.MILPsolverCPLEX import MILPsolverCPLEX


class GraphSolver:
    """Class that solves a single graph using the alpha-genetic algorithm and/or the MILP model in CPLEX"""

    def __init__(self, inputGraphName: str, isSolvedWithGeneticAlg=True, isSolvedWithCPLEX=True, isRace=True,
                 isDrawing=True, isLabeling=True, isOutputtingCPLEX=False):
        """Constructor of a GraphSolver instance"""
        # Graph solver options
        self.isSolvedWithGeneticAlg: bool = isSolvedWithGeneticAlg
        self.isSolvedWithCPLEX: bool = isSolvedWithCPLEX
        self.isRace: bool = isRace
        self.isDrawing: bool = isDrawing
        self.isLabeling: bool = isLabeling
        self.isOutputtingCPLEX: bool = isOutputtingCPLEX

        # Input graph attributes
        self.graphName: str = inputGraphName
        self.graph: CandidateGraph = CandidateGraph()
        self.graph = self.graph.loadCandidateGraph(self.graphName)
        self.minTargetFlow: float = self.graph.totalPossibleDemand

        # Alpha-GA population attribute & hyperparameters
        self.geneticPop: Population = Population(self.graph, self.minTargetFlow,
                         isOneDimAlphaTable=False, isOptimizedArcSelections=True)
        self.geneticPop.setPopulationHyperparams(populationSize=20,
                                                 numGenerations=20,
                                                 initializationStrategy="perEdge",
                                                 initializationDistribution="digital",
                                                 initializationParams=[0.0, 100000.0])
        self.geneticPop.setIndividualSelectionHyperparams(selectionMethod="tournament",
                                                            tournamentSize=4)
        self.geneticPop.setCrossoverHyperparams(crossoverMethod="twoPoint",
                                                crossoverRate=1.0,
                                                crossoverAttemptsPerGeneration=1,
                                                replacementStrategy="replaceWeakestTwo")
        self.geneticPop.setMutationHyperparams(mutationMethod="randomPerEdge",
                                               mutationRate=0.05,
                                               perArcEdgeMutationRate=0.25)

        # MILP CPLEX attribute
        self.milpCplexSolver: MILPsolverCPLEX = MILPsolverCPLEX(self.graph, self.minTargetFlow,
                                          isOneArcPerEdge=False, logOutput=isOutputtingCPLEX)

        # Execution attributes
        self.geneticRuntimeInSeconds: float = -1
        self.milpRuntimeInSeconds: float = -1

    def solveGraph(self) -> None:
        """Solves the graph with the genetic algorithm and/or the MILP formulation in CPLEX"""
        # Solve the alpha-GA population
        if self.isSolvedWithGeneticAlg is True:
            # Timestamp and start the GA
            print("\n\nSolving the " + self.graphName + " graph with a GA population of " + str(
                self.geneticPop.populationSize) + " for " + str(self.geneticPop.numGenerations) + " generations...\n")
            gaStartTime = datetime.now()
            # Evolve the Alpha-GA population
            gaSolution = self.geneticPop.evolvePopulation(printGenerations=True, drawing=self.isDrawing, drawLabels=self.isLabeling)
            print("\nGenetic Algorithm Complete!\nBest solution found = " + str(gaSolution.trueCost))
            if self.isDrawing is True:
                solVis = SolutionVisualizer(gaSolution)
                if self.isLabeling is True:
                    solVis.drawLabeledSolution(leadingText="GA-BEST_")
                else:
                    solVis.drawUnlabeledSolution(leadingText="GA-BEST_")
            # Timestamp and stop the GA
            gaFinishOptStart = datetime.now()
            gaRuntime = gaFinishOptStart - gaStartTime
            self.geneticRuntimeInSeconds = gaRuntime.seconds
            print("\nGA Runtime (in seconds): " + str(self.geneticRuntimeInSeconds))
        # Solve the MILP formulation in CPLEX
        if self.isSolvedWithCPLEX is True:
            # Set time limit if CPLEX is racing GA
            if self.isRace is True:
                self.milpCplexSolver.setTimeLimit(self.geneticRuntimeInSeconds)
            self.milpCplexSolver.findSolution(printDetails=False)
            print("\nCPLEX MILP Solver Complete!\nBest solution found = " + str(self.milpCplexSolver.getObjectiveValue()))
            if self.isDrawing is True:
                opt = self.milpCplexSolver.writeSolution()
                optVis = SolutionVisualizer(opt)
                if self.isLabeling is True:
                    optVis.drawLabeledSolution(leadingText="OPT_")
                else:
                    optVis.drawUnlabeledSolution(leadingText="OPT_")
            self.milpRuntimeInSeconds = self.milpCplexSolver.getCplexRuntime()
            print("CPLEX MILP Runtime (in seconds): " + str(self.milpRuntimeInSeconds))
        print("\n\nProgram complete... Graph solved!\nTerminating execution...\n")
