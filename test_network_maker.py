from src.Network.GraphMaker import GraphMaker

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 test_network_maker.py
"""

"""
# ORIGINAL COST DETERMINING TABLE ADAPTED FROM SIMCCS MODEL (see data/SimCCS_models/Pipeline Cost Model.xlsm)
arcCostLookupTable = [
        [0.19, 0.01238148, 0.010685656],
        [0.54, 0.0140994, 0.004500589],
        [1.13, 0.016216406, 0.002747502],
        [3.25, 0.02169529, 0.00170086],
        [6.86, 0.030974863, 0.001407282],
        [12.26, 0.041795733, 0.001290869],
        [19.69, 0.055473249, 0.001235064],
        [35.13, 0.077642542, 0.001194592],
        [56.46, 0.104715966, 0.001175094],
        [83.95, 0.136751956, 0.001164578],
        [119.16, 0.172747686, 0.001098788]
    ]

    srcSinkCapacityRange = [0.01, 2]
    srcSinkChargeRange = [0.01, 2]
"""

if __name__ == "__main__":
    # Input parameters
    name = ""
    numNodes = 10
    numSources = 2
    numSinks = 2
    # Arc Cost Determining Table
    # Format: cap, fixed cost, variable cost
    arcCostLookupTable = [
        [100, 10, 1]
    ]
    # Source/sink cap and cost ranges
    srcSinkCapacityRange = [100, 200]
    srcSinkChargeRange = [10, 25]

    # Make graph
    graphMaker = GraphMaker(name, numNodes, numSources, numSinks)
    graphMaker.setArcCostLookupTable(arcCostLookupTable=arcCostLookupTable)
    graphMaker.setSourceSinkGeneralizations(True, True, capacityRange=srcSinkCapacityRange,
                                            chargeRange=srcSinkChargeRange)
    generatedNetwork = graphMaker.generateNetwork()
    generatedNetwork.drawNetworkTriangulation()
    generatedNetwork.saveNetwork()
