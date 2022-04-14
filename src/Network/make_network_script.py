from src.Network.GraphMaker import GraphMaker

name = "test-50-3-3-1"
numNodes = 50
numSources = 3
numSinks = 3

graphMaker = GraphMaker(name, numNodes, numSources, numSinks)
# Uncomment to tune how the network generates costs and to turn on generalizations
graphMaker.setCostDeterminingHyperparameters(possibleArcCaps=[100])
graphMaker.setSourceSinkGeneralizations(True, True)

generatedNetwork = graphMaker.generateNetwork()
generatedNetwork.drawNetworkTriangulation()
generatedNetwork.saveNetwork()
