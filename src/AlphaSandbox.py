from datetime import datetime

from src.AlphaGeneticSolver.AlphaPopulation import AlphaPopulation
from src.FixedChargeNetwork.FixedChargeFlowNetwork import FixedChargeFlowNetwork

"""USED FOR EXPERIMENTATION"""

# WALL CLOCK TIME STAMP
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)

# TEST OF RANDOM GRAPH GENERATOR
# graphGen = GraphGenerator("r2000-4(50,50)", 2000, 0.04, 50, 50, [200, 2000], [500, 500], [20, 200], [1, 30], [100, 100], 3)
# graphGen.saveFCFN()

# INITIALIZATIONS
flowNetwork = FixedChargeFlowNetwork()
# flowNetwork.loadFCFN("small")
flowNetwork.loadFCFN("r100-3(10,10)")
# flowNetwork.loadFCFN("r2000-4(50,50)")

# Solve individual initially and print data
GA = AlphaPopulation(flowNetwork, 999, 1, 1)
GA.initializePopulation([0.0, 1.0])
GA.visualizeIndividual(0, 0, )

# TEST OF MILP
# flowNetwork.executeSolver(35)
# flowNetwork.visualizeNetwork(catName="-OPT")


"""
# TEST OF EXPLICITLY MANIPULATING ALPHA VALUES

# INITIALIZATIONS
flowNetwork = FixedChargeFlowNetwork()
flowNetwork.loadFCFN("small")
# flowNetwork.loadFCFN("r100-3(10,10)")
# flowNetwork.loadFCFN("r2000-4(50,50)")

# Solve individual initially and print data
GA = AlphaPopulation(flowNetwork, 35, 1, 1)
GA.initializePopulation([0.0, 1.0])
individual = GA.population[0]
print(individual.alphaValues)
individual.allUsedPaths()
individual.printAllPathData()
GA.visualizeIndividual(0, 0, graphType="withLabels")

# Solve a second mutate individual
mutatedIndividual = AlphaIndividual(flowNetwork)
mutatedIndividual.alphaValues = individual.alphaValues
for path in individual.paths:
    for edge in path.edges:
        edgeID = int(edge.lstrip("e"))
        mutatedIndividual.alphaValues[edgeID] = 100
print(mutatedIndividual.alphaValues)
GA.solveIndividual(mutatedIndividual)
mutatedIndividual.allUsedPaths()
mutatedIndividual.printAllPathData()
GA.population.append(mutatedIndividual)
GA.visualizeIndividual(0, 1, graphType="withLabels")

# TEST OF MILP
flowNetwork.executeSolver(35)
flowNetwork.visualizeNetwork(catName="-OPT")
"""

"""
# TEST OF LARGE INSTANCE SIZE (TWICE SOLVED W/ TIMESTAMPS)
# WALL CLOCK TIME STAMP
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)

# TEST OF FCFN
flowNetwork = FixedChargeFlowNetwork()
flowNetwork.loadFCFN("r2000-5(50,50)")

# TODO- Delete Timestamp
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Network Loaded =", current_time)
# TODO- Delete Timestamp

# TEST OF GENETIC ALGORITHM
GA = AlphaPopulation(flowNetwork, 2500, 2, 1)
# TODO- Delete Timestamp
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Population/Solver Initialized =", current_time)
# TODO- Delete Timestamp

GA.initializePopulation([0, 1.0])

# TODO- Delete Timestamp
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Two Instances Solved =", current_time)
# TODO- Delete Timestamp

GA.visualizeIndividual(0, 0)
GA.visualizeIndividual(0, 1)

# TODO- Delete Timestamp
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Stop Time =", current_time)
# TODO- Delete Timestamp
"""

"""
# Test of Path-Based Crossover
ga = AlphaPopulation(flowNetwork, 20, 2, 1)
ga.initializePopulation([0.0, 1.0])
print(ga.population[0].alphaValues)
print(ga.population[1].alphaValues)
pOnePaths = ga.densityBasedPathSelection(0, 1, "mostDense")
print(pOnePaths[0].edges)
pTwoPaths = ga.densityBasedPathSelection(1, 1, "mostDense")
print(pTwoPaths[0].edges)
ga.pathBasedCrossover(0, 1, pOnePaths, pTwoPaths, "replaceWeakestTwo")
print(ga.population[0].alphaValues)
print(ga.population[1].alphaValues)
"""

"""
# Test of Random-Point Crossover
ga = AlphaPopulation(flowNetwork, 20, 4, 1)
ga.initializePopulation([0.0, 1.0])
print(ga.population[0].alphaValues)
print(ga.population[1].alphaValues)
print(ga.population[2].alphaValues)
print(ga.population[3].alphaValues)
ga.randomOnePointCrossover(0, 1, "fromLeft", "replaceParents")
print(ga.population[0].alphaValues)
print(ga.population[1].alphaValues)
print(ga.population[2].alphaValues)
print(ga.population[3].alphaValues)
"""

"""
# Test of Two-Point Crossover
ga = AlphaPopulation(flowNetwork, 20, 2, 1)
ga.initializePopulation([0.0, 1.0])
print(ga.population[0].alphaValues)
print(ga.population[1].alphaValues)
ga.randomTwoPointCrossover(0, 1, "replaceParents")
print(ga.population[0].alphaValues)
print(ga.population[1].alphaValues)
"""

"""
# Test of Density Based Path Selection Operators
ga = AlphaPopulation(flowNetwork, 200, 1, 2)
ga.initializePopulation([0.0, 1.0])
ga.population[0].printAllPathData()
print(len(ga.population[0].paths))
selectedPaths = ga.densityBasedPathSelection(0, 2, "mostDense")
selectedPaths = ga.rouletteWheelPathSelection(0, 2, "mostDense")
selectedPaths = ga.tournamentPathSelection(0, 4, 2, "mostDense")
for path in selectedPaths:
    print(path.edges)
    print(path.flowPerCostDensity)
"""

"""
# Test of RandomOnePointCrossover
ga = AlphaPopulation(flowNetwork, 20, 2, 1)
ga.initializePopulation([0.0, 1.0])
print(ga.population[0].alphaValues)
print(ga.population[1].alphaValues)
ga.randomOnePointCrossoverWithParentReplacement(0, 1, "fromLeft")
print(ga.population[0].alphaValues)
print(ga.population[1].alphaValues)
"""

"""
# Test of rouletteWheelSelection(3)
ga = AlphaPopulation(flowNetwork, 200, 5, 1)
ga.initializePopulation([0.0, 1.0])
extraInd = AlphaIndividual(flowNetwork)
extraInd.initializeAlphaValuesRandomly(0, 500)
extraInd.executeAlphaSolver(200)
ga.population.append(extraInd)
ga.printAllCosts()
print(ga.rouletteWheelSelection(3))
"""

"""
# Test of path-based mutation
ga = AlphaPopulation(flowNetwork, 200, 1, 1)
ga.initializePopulation([0.0, 1.0])
for i in range(10):
    ga.population[0].executeAlphaSolver(200)
    ga.population[0].visualizeAlphaNetwork(endCatName=str(i))
    ga.mostDensePathMutation(0, 0 + i/20, 1 + i/20)
    # ga.mostDensePathMutation(0, 0 + i / 20, 1 + i / 20)
"""

# TEST OF ALPHA INDIVIDUAL
# alphaFN = AlphaIndividual(flowNetwork)
# alphaFN.executeAlphaSolver(100)
# alphaFN.visualizeAlphaNetwork(endCatName="1")
# alphaFN.allUsedPaths()
# alphaFN.pathsVsElementsCost()
# alphaFN.printAllPathData()

# TEST OF MILP
# flowNetwork.executeSolver(200)
# flowNetwork.visualizeNetwork()

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
