from src.Network.GraphMaker import GraphMaker

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 test_network_maker.py
"""

if __name__ == "__main__":
    name = "1000-1-10"
    numNodes = 1000
    numSources = 10
    numSinks = 10

    graphMaker = GraphMaker(name, numNodes, numSources, numSinks)
    # Uncomment to tune how the network generates costs and to turn on generalizations
    graphMaker.setCostDeterminingHyperparameters(possibleArcCaps=[100])
    graphMaker.setSourceSinkGeneralizations(True, True)

    generatedNetwork = graphMaker.generateNetwork()
    generatedNetwork.drawNetworkTriangulation()
    generatedNetwork.saveNetwork()
