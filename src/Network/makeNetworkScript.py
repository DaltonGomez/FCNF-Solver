from src.Network.GraphMaker import GraphMaker

name = "test-6-1-1"
numNodes = 6
numSources = 1
numSinks = 1

graphMaker = GraphMaker(name, numNodes, numSources, numSinks)
# Uncomment to tune how the network generates costs and to turn on generalizations
# graphMaker.setCostDeterminingHyperparameters()
# graphMaker.setSourceSinkGeneralizations()

generatedNetwork = graphMaker.generateNetwork()
generatedNetwork.drawNetworkTriangulation()
generatedNetwork.saveNetwork()
