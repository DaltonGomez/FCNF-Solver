from src.AlphaGenetic.AlphaGeneticTuning import AlphaGeneticTuning

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 run_alpha_ga_tuning.py
"""

if __name__ == "__main__":
    # Input Parameters
    numNetworks = 4
    nodeSizeRange = [25, 200]
    srcSinkSet = [1, 5, 10]
    arcCostLookupTable = [
        [100, 10, 1]
    ]
    srcSinkCapacityRange = [100, 200]
    srcSinkChargeRange = [10, 25]
    targetAsPercentTotalDemand = 0.50
    numTrials = 5

    # Build Networks
    tuner = AlphaGeneticTuning(numNetworks=numNetworks, nodeSizeRange=nodeSizeRange, srcSinkSet=srcSinkSet,
                               arcCostLookupTable=arcCostLookupTable, srcSinkCapacityRange=srcSinkCapacityRange,
                               srcSinkChargeRange=srcSinkChargeRange,
                               targetAsPercentTotalDemand=targetAsPercentTotalDemand, numTrials=numTrials)

    # Run Experiment
    tuner.runTuning()
