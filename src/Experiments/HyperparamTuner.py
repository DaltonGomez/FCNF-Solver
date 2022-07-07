import csv
import os
from datetime import datetime
from typing import List, Dict

from src.Experiments.GAvsCPLEX import GAvsCPLEX

"""
# COMPLETE HP SEARCH-SPACE DICTIONARY:
self.hpSpace: Dict[str, List] = {
                        "populationSize": [10, 25, 50, 100],
                        "numGenerations": [10, 25, 50, 100],
                        "terminationMethod": ["setGenerations", "stagnationPeriod"],
                        "stagnationPeriod": [5],
                        "isOneDimAlphaTable": [True, False],
                        "isOptimizedArcSelections": [True, False],
                        "initializationStrategy": ["perEdge", "perArc", "reciprocalCap"],
                        "initializationDistribution": ["uniform", "gaussian", "digital"],
                        "initializationParams": [
                                                    [0, 100000],
                                                    [1, 100000],
                                                    [10, 100000]
                                                ],
                        "selectionMethod": ["tournament", "roulette", "random"],
                        "tournamentSize": [3, 5, 8],
                        "crossoverMethod": ["onePoint", "twoPoint"],
                        "crossoverRate": [0.25, 0.50, 0.75, 1.0],
                        "crossoverAttemptsPerGeneration": [1, 2, 3, 4],
                        "replacementStrategy": ["replaceWeakestTwo", "replaceParents"],
                        "mutationMethod": ["randomSingleArc", "randomSingleEdge", "randomPerArc", "randomPerEdge", "randomTotal"],
                        "mutationRate": [0.01, 0.05, 0.10, 0.25, 0.50],
                        "perArcEdgeMutationRate": [0.01, 0.05, 0.10, 0.25, 0.50]
                        }
"""


class HyperparamTuner:
    """Class that preforms a grid search over various hyperparameter values for the alpha-GA population"""

    def __init__(self, inputGraphs: List[str], runsPerGraph: int):
        """Constructor of a HyperparamTuner instance"""
        # Hyperparameter tuner attributes/options
        self.tuningRunID: str = "HyperparamTuner--" + datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        self.inputGraphs: List[str] = inputGraphs
        self.runsPerGraph: int = runsPerGraph

        # Hyperparameter search space - defined as a dictionary where the keys are the hyperparam and the values are lists of candidate values
        self.hpSpace: Dict[str, List] = {
                        "isOneDimAlphaTable": [True, False],
                        "isOptimizedArcSelections": [True, False],
                        "populationSize": [10, 25, 50, 100],
                        "numGenerations": [10, 25, 50, 100],
                        "initializationStrategy": ["perEdge", "perArc"],
                        "initializationDistribution": ["digital"],
                        "initializationParams": [
                                                    [0, 100000],
                                                    [1, 100000],
                                                    [10, 100000]
                                                ],
                        "selectionMethod": ["tournament", "roulette", "random"],
                        "tournamentSize": [3, 5, 8],
                        "crossoverMethod": ["onePoint", "twoPoint"],
                        "crossoverRate": [0.30, 0.70, 1.0],
                        "crossoverAttemptsPerGeneration": [1, 2, 3, 4],
                        "replacementStrategy": ["replaceWeakestTwo", "replaceParents"],
                        "mutationMethod": ["randomPerArc", "randomPerEdge", "randomTotal"],
                        "mutationRate": [0.01, 0.05, 0.10, 0.25, 0.50],
                        "perArcEdgeMutationRate": [0.01, 0.05, 0.10, 0.25, 0.50]
                        }

    def conductGridSearch(self) -> None:
        """Conducts the hyperparameter grid search tuning over the HP space and input graphs"""
        # NOTE - Not comprehensive of all hyperparameters that could be tuned
        # Write CSV to disc and timestamp start
        self.createCSV()
        startTime = datetime.now()
        # Iterate over 1D alpha table and optimized arcs post-processing options
        for is1dAlpha in self.hpSpace["isOneDimAlphaTable"]:
            for isOptArcs in self.hpSpace["isOptimizedArcSelections"]:
                # Iterate over all graphs, each for n runs
                for graphName in self.inputGraphs:
                    for runNum in range(self.runsPerGraph):
                        # Iterate over remaining HPs
                        for popSize in self.hpSpace["populationSize"]:
                            for numGens in self.hpSpace["numGenerations"]:
                                for initStrat in self.hpSpace["initializationStrategy"]:
                                    for initDist in self.hpSpace["initializationDistribution"]:
                                        for initParams in self.hpSpace["initializationParams"]:
                                            for select in self.hpSpace["selectionMethod"]:
                                                for crossMeth in self.hpSpace["crossoverMethod"]:
                                                    for crossRate in self.hpSpace["crossoverRate"]:
                                                        for crossAPG in self.hpSpace["crossoverAttemptsPerGeneration"]:
                                                            for replace in self.hpSpace["replacementStrategy"]:
                                                                for mutateMeth in self.hpSpace["mutationMethod"]:
                                                                    for mutateRate in self.hpSpace["mutationRate"]:
                                                                        for perAeMutate in self.hpSpace["perArcEdgeMutationRate"]:
                                                                            if select == "tournament":
                                                                                for tourny in self.hpSpace["tournamentSize"]:
                                                                                    # Instantiate GA pop
                                                                                    tunerRun = GAvsCPLEX(graphName,
                                                                                                    isSolvedWithGeneticAlg=True,
                                                                                                    isOneDimAlphaTable=is1dAlpha,
                                                                                                    isOptimizedArcSelections=isOptArcs,
                                                                                                    isSolvedWithCPLEX=True,
                                                                                                    isRace=True,
                                                                                                    isDrawing=False,
                                                                                                    isLabeling=False,
                                                                                                    isGraphing=False,
                                                                                                    isOutputtingCPLEX=True)
                                                                                    # Set hyperparameter settings
                                                                                    tunerRun.geneticPop.setPopulationHyperparams(
                                                                                        populationSize=popSize,
                                                                                        numGenerations=numGens)
                                                                                    tunerRun.geneticPop.setInitializationHyperparams(
                                                                                        initializationStrategy=initStrat,
                                                                                        initializationDistribution=initDist,
                                                                                        initializationParams=initParams)
                                                                                    tunerRun.geneticPop.setIndividualSelectionHyperparams(
                                                                                        selectionMethod=select,
                                                                                        tournamentSize=tourny)
                                                                                    tunerRun.geneticPop.setCrossoverHyperparams(
                                                                                        crossoverMethod=crossMeth,
                                                                                        crossoverRate=crossRate,
                                                                                        crossoverAttemptsPerGeneration=crossAPG,
                                                                                        replacementStrategy=replace)
                                                                                    tunerRun.geneticPop.setMutationHyperparams(
                                                                                        mutationMethod=mutateMeth,
                                                                                        mutationRate=mutateRate,
                                                                                        perArcEdgeMutationRate=perAeMutate)
                                                                                    # Evolve and save results
                                                                                    thisRunData = tunerRun.solveGraphWithoutPrints()
                                                                                    self.writeRowToCSV(thisRunData)
        totalTuningRuntime = datetime.now() - startTime
        totalTuningMinutes = totalTuningRuntime.seconds/60
        self.writeRowToCSV(["Total Tuning Runtime (in min):", totalTuningMinutes])
        print("\n\nTuning Experiment Complete!!!\nTotal Tuning Runtime (in minutes): " + str(totalTuningMinutes))

    def createCSV(self) -> None:
        """Creates a CSV file for the output data of the run and writes a header"""
        # GA Specific data
        runHeader = ["Run ID", "Graph Name", "Num Nodes", "Num Sources", "Num Sinks", "Num Edges",
                    "Num Arc Caps", "Target Flow", "is Src/Sink Capped?", "is Src/Sink Charged?", "Pop Size",
                    "Num Gens", "is 1D Alphas?", "is Optimized Arcs?", "termination", "stagnation",
                    "Init Strategy", "Init Dist", "Init Param 0", "Init Param 1", "Selection", "Tourny Size",
                    "Crossover", "CO Rate", "CO Attempts/Gen", "Replacement Strategy", "Mutation", "Mutate Rate",
                    "Per Arc/Edge Mutate Rate", "GA Best Obj Val", "GA Runtime (sec)", "CPLEX Obj Val",
                    "CPLEX Runtime (sec)", "Time Limit", "Status", "Status Code", "Best Bound",
                    "MILP Gap", "GA Gap", "MILP Gap - GA GAP"]
        # Build Output Header
        outputHeader = [["MULTI-GA vs. CPLEX RESULTS OUTPUT", self.tuningRunID, "Runs Per Graph=" + str(self.runsPerGraph)],
                        ["Graphs"],
                        self.inputGraphs,
                        ["Tuning Data"],
                        runHeader]
        # Create CSV File
        currDir = os.getcwd()
        csvName = self.tuningRunID + ".csv"
        catPath = os.path.join(currDir, csvName)
        csvFile = open(catPath, "w+", newline="")
        with csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(outputHeader)

    def writeRowToCSV(self, outputRow: list) -> None:
        """Appends the most recent data onto a .csv file"""
        currDir = os.getcwd()
        csvName = self.tuningRunID + ".csv"
        catPath = os.path.join(currDir, csvName)
        csvFile = open(catPath, "a", newline="")
        with csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(outputRow)

    def setHyperparamSpace(self, hyperparam: str, searchSpace: list) -> None:
        """Updates the key-value pair of the hyperparameter search space dictionary"""
        self.hpSpace[hyperparam] = searchSpace
