import csv
import os
from datetime import datetime
from typing import List, Dict

from src.Experiments.GAvsMILP import GAvsMILP

"""
# COMPLETE HP SEARCH-SPACE DICTIONARY:
self.hpSpace: Dict[str, List] = {
                        "populationSize": [10, 25, 50, 100],
                        "numGenerations": [10, 25, 50, 100],
                        "terminationMethod": ["setGenerations", "stagnationPeriod"],
                        "stagnationPeriod": [5],
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
                        "replacementStrategy": ["replaceWeakestTwo", "replaceParents", "replaceRandomTwo"],
                        "mutationMethod": ["randomSingleArc", "randomSingleEdge", "randomPerArc", "randomPerEdge", "randomTotal"],
                        "mutationRate": [0.01, 0.05, 0.10, 0.25, 0.50],
                        "perArcEdgeMutationRate": [0.01, 0.05, 0.10, 0.25, 0.50],
                        "isDaemonUsed": [True, False],
                        "annealingConstant": [0.25, 0.5, 1, 2],
                        "daemonStrategy": ["globalBinary", "globalMean", "globalMedian", "personalMean", "personalMedian"],
                        "daemonStrength": [0.5, 1, 2]
                        }
"""


class HyperparamTuner:
    """Class that preforms a grid search over various hyperparameter values for the alpha-GA population"""

    def __init__(self, inputGraphs: List[str], runsPerGraph: int, isDaemonUsed=True,
                 tuneOneDimAlpha=True, tuneManyDimAlpha=False, tuneOptimizedArcs=True, tuneNonOptimizedArcs=False):
        """Constructor of a HyperparamTuner instance"""
        # Hyperparameter tuner attributes/options
        self.tuningRunID: str = "HyperparamTuner--" + datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        self.inputGraphs: List[str] = inputGraphs
        self.runsPerGraph: int = runsPerGraph
        self.isDaemonUsed: bool = isDaemonUsed
        self.tuneOneDimAlpha: bool = tuneOneDimAlpha
        self.tuneManyDimAlpha: bool = tuneManyDimAlpha
        self.tuneOptimizedArcs: bool = tuneOptimizedArcs
        self.tuneNonOptimizedArcs: bool = tuneNonOptimizedArcs

        # Hyperparameter search space - defined as a dictionary where the keys are the hyperparam and the values are lists of candidate values
        self.hpSpace: Dict[str, List] = {
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
                        "replacementStrategy": ["replaceWeakestTwo", "replaceParents", "replaceRandomTwo"],
                        "mutationMethod": ["randomPerArc", "randomPerEdge", "randomTotal"],
                        "mutationRate": [0.01, 0.05, 0.10, 0.25, 0.50],
                        "perArcEdgeMutationRate": [0.01, 0.05, 0.10, 0.25, 0.50],
                        "isDaemonUsed": [True, False],
                        "annealingConstant": [0.25, 0.5, 1, 2],
                        "daemonStrategy": ["globalBinary", "globalMean", "globalMedian", "personalMean", "personalMedian"],
                        "daemonStrength": [0.5, 1, 2]
                        }
        self.numTotalRuns = self.computeTotalRuns()

    def computeTotalRuns(self) -> int:
        """Computes the total number of runs (i.e. GA population evolutions) to complete the grid search"""
        if self.isDaemonUsed is True:
            totalRuns = len(self.inputGraphs) * self.runsPerGraph
            for hyperparam in self.hpSpace.keys():
                totalRuns *= len(self.hpSpace[hyperparam])
        else:
            totalRuns = len(self.inputGraphs) * self.runsPerGraph
            trimmedHpSpaceKeys = list(self.hpSpace.keys())
            for daemonHP in ["isDaemonUsed", "annealingConstant", "daemonStrategy", "daemonStrength"]:
                trimmedHpSpaceKeys.remove(daemonHP)
            for hyperparam in trimmedHpSpaceKeys:
                totalRuns *= len(self.hpSpace[hyperparam])
        if self.tuneOneDimAlpha is True and self.tuneManyDimAlpha is True:
            totalRuns *= 2
        if self.tuneOptimizedArcs is True and self.tuneNonOptimizedArcs is True:
            totalRuns *= 2
        return totalRuns

    def runTuningExperiment(self) -> None:
        """Conducts the hyperparameter grid search tuning over the HP space and input graphs"""
        # NOTE - Nested FOR loops are not comprehensive of all hyperparameters that could be tuned
        self.numTotalRuns = self.computeTotalRuns()
        print("\nStarting a tuning experiment with " + str(self.numTotalRuns) + " total runs...\n")
        # Write CSV to disc and timestamp start
        self.createCSV()
        startTime = datetime.now()
        # Run tuning experiments based on n-Dim alpha, arc optimization settings and if the daemon is used
        if self.tuneOneDimAlpha is True:
            if self.tuneOptimizedArcs is True:
                if self.isDaemonUsed is True:
                    self.conductGridSearchWithDaemon(isOneDimAlpha=True, isArcOptimized=True)
                else:
                    self.conductGridSearchWithOutDaemon(isOneDimAlpha=True, isArcOptimized=True)
            if self.tuneNonOptimizedArcs is True:
                if self.isDaemonUsed is True:
                    self.conductGridSearchWithDaemon(isOneDimAlpha=True, isArcOptimized=False)
                else:
                    self.conductGridSearchWithOutDaemon(isOneDimAlpha=True, isArcOptimized=False)
        if self.tuneManyDimAlpha is True:
            if self.tuneOptimizedArcs is True:
                if self.isDaemonUsed is True:
                    self.conductGridSearchWithDaemon(isOneDimAlpha=False, isArcOptimized=True)
                else:
                    self.conductGridSearchWithOutDaemon(isOneDimAlpha=False, isArcOptimized=True)
            if self.tuneNonOptimizedArcs is True:
                if self.isDaemonUsed is True:
                    self.conductGridSearchWithDaemon(isOneDimAlpha=False, isArcOptimized=False)
                else:
                    self.conductGridSearchWithOutDaemon(isOneDimAlpha=False, isArcOptimized=False)
        totalTuningRuntime = datetime.now() - startTime
        totalTuningMinutes = totalTuningRuntime.seconds/60
        self.writeRowToCSV(["Total Tuning Runtime (in min):", totalTuningMinutes])
        print("\n\nTuning Experiment Complete!!!\nTotal Tuning Runtime (in minutes): " + str(totalTuningMinutes))

    def conductGridSearchWithDaemon(self, isOneDimAlpha: bool, isArcOptimized: bool) -> None:
        """Completes the grid search of the hyperparams with the given n-Dim alpha and arc optimization settings"""
        # Iterate over all graphs
        for graphName in self.inputGraphs:
            # Iterate over HPs
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
                                                                for isDaemon in self.hpSpace["isDaemonUsed"]:
                                                                    for annealConst in self.hpSpace["annealingConstant"]:
                                                                        for daemonStrat in self.hpSpace["daemonStrategy"]:
                                                                            for daemonStrength in self.hpSpace["daemonStrength"]:
                                                                                # Iterate over runs
                                                                                for runNum in range(self.runsPerGraph):
                                                                                    # Search tourny size only if tournament is selection strategy
                                                                                    if select == "tournament":
                                                                                        for tourny in self.hpSpace["tournamentSize"]:
                                                                                            # Instantiate GA pop
                                                                                            tunerRun = GAvsMILP(
                                                                                                graphName,
                                                                                                isSolvedWithGeneticAlg=True,
                                                                                                isOneDimAlphaTable=isOneDimAlpha,
                                                                                                isOptimizedArcSelections=isArcOptimized,
                                                                                                isSolvedWithMILP=True,
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
                                                                                            tunerRun.geneticPop.setDaemonHyperparams(
                                                                                                isDaemonUsed=isDaemon,
                                                                                                annealingConstant=annealConst,
                                                                                                daemonStrategy=daemonStrat,
                                                                                                daemonStrength=daemonStrength)
                                                                                            # Evolve and save results
                                                                                            thisRunData = tunerRun.solveGraphWithoutPrints()
                                                                                            self.writeRowToCSV(thisRunData)

    def conductGridSearchWithOutDaemon(self, isOneDimAlpha: bool, isArcOptimized: bool) -> None:
        """Completes the grid search of the hyperparams with the given n-Dim alpha and arc optimization settings, not searching daemon HPs"""
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
                                                                            tunerRun = GAvsMILP(
                                                                                graphName,
                                                                                isSolvedWithGeneticAlg=True,
                                                                                isOneDimAlphaTable=isOneDimAlpha,
                                                                                isOptimizedArcSelections=isArcOptimized,
                                                                                isSolvedWithMILP=True,
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
                                                                            tunerRun.geneticPop.setDaemonHyperparams(
                                                                                isDaemonUsed=False,
                                                                                annealingConstant=0.5,
                                                                                daemonStrategy="globalMean",
                                                                                daemonStrength=1.0)
                                                                            # Evolve and save results
                                                                            thisRunData = tunerRun.solveGraphWithoutPrints()
                                                                            self.writeRowToCSV(thisRunData)

    def createCSV(self) -> None:
        """Creates a CSV file for the output data of the run and writes a header"""
        # GA Specific data
        runHeader = ["Run ID", "Graph Name", "Num Nodes", "Num Sources", "Num Sinks", "Num Edges",
                    "Num Arc Caps", "Target Flow", "is Src/Sink Capped?", "is Src/Sink Charged?", "Pop Size",
                    "Num Gens", "is 1D Alphas?", "is Optimized Arcs?", "termination", "stagnation",
                    "Init Strategy", "Init Dist", "Init Param 0", "Init Param 1", "Selection", "Tourny Size",
                    "Crossover", "CO Rate", "CO Attempts/Gen", "Replacement Strategy", "Mutation", "Mutate Rate",
                    "Per Arc/Edge Mutate Rate", "is Daemon Used?", "Annealing Constant", "Daemon Strategy",
                    "Daemon Strength", "GA Best Obj Val", "GA Runtime (sec)", "MILP Obj Val",
                    "MILP Runtime (sec)", "Time Limit", "Status", "Status Code", "Best Bound",
                    "MILP Gap", "GA Gap", "MILP Gap - GA GAP"]
        # Build Output Header
        outputHeader = [["HYPERPARAM TUNING RESULTS", self.tuningRunID, "Runs Per Graph=" + str(self.runsPerGraph)],
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
