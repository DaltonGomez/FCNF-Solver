from src.AlphaGeneticSolver.AlphaIndividual import AlphaIndividual
from src.FixedChargeNetwork.FixedChargeFlowNetwork import FixedChargeFlowNetwork

"""USED FOR EXPERIMENTATION WITH ALPHA VALUES"""

# TEST OF FCFN
flowNetwork = FixedChargeFlowNetwork()
flowNetwork.loadFCFN("r100-3")

# TEST OF ALPHA INDIVIDUAL
alphaFN = AlphaIndividual(flowNetwork)
alphaFN.initializeAlphaValuesConstantly(0.15)
alphaFN.executeAlphaSolver(100)
alphaFN.visualizeAlphaNetwork(frontCatName="1")
# alphaFN.allUsedPaths()

# TEST OF MILP
flowNetwork.executeSolver(100)
flowNetwork.visualizeNetwork()

"""
# TEST OF CONSTANT ALPHA VALUES ACROSS INDIVIDUALS
alphaFN = AlphaIndividual(flowNetwork)
alphaFN.initializeAlphaValuesConstantly(0)
alphaFN.executeAlphaSolver(80)
alphaFN.visualizeAlphaNetwork(frontCatName="1")

alphaFN2 = AlphaIndividual(flowNetwork)
alphaFN2.initializeAlphaValuesConstantly(0.5)
alphaFN2.executeAlphaSolver(80)
alphaFN2.visualizeAlphaNetwork(frontCatName="2")

alphaFN3 = AlphaIndividual(flowNetwork)
alphaFN3.initializeAlphaValuesConstantly(1)
alphaFN3.executeAlphaSolver(80)
alphaFN3.visualizeAlphaNetwork(frontCatName="3")

alphaFN4 = AlphaIndividual(flowNetwork)
alphaFN4.initializeAlphaValuesConstantly(10)
alphaFN4.executeAlphaSolver(80)
alphaFN4.visualizeAlphaNetwork(frontCatName="4")
"""

"""
# ALPHA VALUES VERSES LP SELECTED FLOW
alphaFN = AlphaIndividual(flowNetwork)
alphaFN.initializeAlphaValuesRandomly()
alphaFN.executeAlphaSolver(80)
alphaFN.visualizeAlphaNetwork(frontCatName="1")
for i in range(alphaFN.FCFN.numEdges):
    print("e" + str(i) + "\t\t" + str(alphaFN.alphaValues[i]) + "\t\t" + str(alphaFN.FCFN.edgesDict["e" + str(i)].flow))

alphaFN2 = AlphaIndividual(flowNetwork)
for i in range(alphaFN2.FCFN.numEdges):
    alphaFN2.alphaValues[i] = 1 - alphaFN.alphaValues[i]
alphaFN2.executeAlphaSolver(80)
alphaFN2.visualizeAlphaNetwork(frontCatName="2")
for i in range(alphaFN2.FCFN.numEdges):
    print("e" + str(i) + "\t\t" + str(alphaFN2.alphaValues[i]) + "\t\t" + str(alphaFN2.FCFN.edgesDict["e" + str(i)].flow))
"""

"""
# TEST OF INITIALIZING ON RANGES
alphaFN = AlphaIndividual(flowNetwork)
alphaFN.initializeAlphaValuesRandomlyOnRange(0, 0.5)
alphaFN.executeAlphaSolver(80)
alphaFN.visualizeAlphaNetwork(frontCatName="1")

alphaFN2 = AlphaIndividual(flowNetwork)
alphaFN2.initializeAlphaValuesRandomlyOnRange(0.5, 1)
alphaFN2.executeAlphaSolver(80)
alphaFN2.visualizeAlphaNetwork(frontCatName="2")

alphaFN3 = AlphaIndividual(flowNetwork)
alphaFN3.initializeAlphaValuesRandomlyOnRange(0, 0.25)
alphaFN3.executeAlphaSolver(80)
alphaFN3.visualizeAlphaNetwork(frontCatName="3")

alphaFN4 = AlphaIndividual(flowNetwork)
alphaFN4.initializeAlphaValuesRandomlyOnRange(0.75, 1)
alphaFN4.executeAlphaSolver(80)
alphaFN4.visualizeAlphaNetwork(frontCatName="4")
"""

"""
# TEST OF "OPTIMAL" ALPHA VALUES
flowNetwork2 = FixedChargeFlowNetwork()
flowNetwork2.loadFCFN("r100-2")
alphaFN = AlphaIndividual(flowNetwork2)
for i in range(flowNetwork.numEdges):
    flow = alphaFN.FCFN.edgesDict["e" + str(i)].flow
    if flow > 0:
        alpha = 1/flow
    else:
        alpha = 1  # UNSURE IF THIS SHOULD BE SET HIGH OR LOW???
    alphaFN.alphaValues[i] = alpha
alphaFN.executeAlphaSolver(80)
alphaFN.visualizeAlphaNetwork(frontCatName="1")
"""
