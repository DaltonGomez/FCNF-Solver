from src.Network.GraphMaker import GraphMaker

name = ""
numNodes = 8
numSources = 2
numSinks = 2

graphMaker = GraphMaker(name, numNodes, numSources, numSinks)
# Uncomment to tune how the network generates costs and to turn on generalizations
graphMaker.setCostDeterminingHyperparameters(possibleArcCaps=[100])
graphMaker.setSourceSinkGeneralizations(True, True)

generatedNetwork = graphMaker.generateNetwork()
generatedNetwork.drawNetworkTriangulation()
generatedNetwork.saveNetwork()
