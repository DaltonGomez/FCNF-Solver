from src.Network.GraphMaker import GraphMaker

name = "test"
numNodes = 3
numSources = 1
numSinks = 1

graphMaker = GraphMaker(name, numNodes, numSources, numSinks)
# Uncomment to tune how the network generates costs and to turn on generalizations
graphMaker.setCostDeterminingHyperparameters(possibleArcCaps=[100])
graphMaker.setSourceSinkGeneralizations(True, True)

generatedNetwork = graphMaker.generateNetwork()
generatedNetwork.drawNetworkTriangulation()
generatedNetwork.saveNetwork()
