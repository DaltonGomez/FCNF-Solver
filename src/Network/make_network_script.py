from src.Network.GraphMaker import GraphMaker

name = "test-20-2-2"
numNodes = 20
numSources = 2
numSinks = 2

graphMaker = GraphMaker(name, numNodes, numSources, numSinks)
# Uncomment to tune how the network generates costs and to turn on generalizations
# graphMaker.setCostDeterminingHyperparameters()
graphMaker.setSourceSinkGeneralizations(True, [100, 300], True, [200, 300])

generatedNetwork = graphMaker.generateNetwork()
generatedNetwork.drawNetworkTriangulation()
generatedNetwork.saveNetwork()
