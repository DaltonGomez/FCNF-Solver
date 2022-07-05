from Utilities.old_code.OLD_AlphaGeneticResults import AlphaGeneticResults

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 run_alpha_ga_results_batch.py
"""

if __name__ == "__main__":
    # Input Parameters
    numNetworks = 2
    nodeSizeRange = [25, 400]
    srcSinkSet = [1, 5, 10]
    arcCostLookupTable = [
        [100, 10, 1]
    ]
    srcSinkCapacityRange = [100, 200]
    srcSinkChargeRange = [10, 25]
    targetAsPercentTotalDemand = 0.50
    numTrials = 10

    # Build Networks
    experiment = AlphaGeneticResults(numNetworks=numNetworks, nodeSizeRange=nodeSizeRange, srcSinkSet=srcSinkSet,
                                     arcCostLookupTable=arcCostLookupTable, srcSinkCapacityRange=srcSinkCapacityRange,
                                     srcSinkChargeRange=srcSinkChargeRange,
                                     targetAsPercentTotalDemand=targetAsPercentTotalDemand, numTrials=numTrials)

    # Run Experiment
    experiment.runExperiment()
