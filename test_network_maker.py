from src.Network.GraphMaker import GraphMaker

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 test_network_maker.py
"""

if __name__ == "__main__":
    name = " "
    numNodes = 5
    numSources = 1
    numSinks = 1

    graphMaker = GraphMaker(name, numNodes, numSources, numSinks)
    # Uncomment to tune how the network generates costs and to turn on generalizations
    graphMaker.setCostDeterminingHyperparameters(possibleArcCaps=[100])
    graphMaker.setSourceSinkGeneralizations(True, True)

    generatedNetwork = graphMaker.generateNetwork()
    generatedNetwork.drawNetworkTriangulation()
    generatedNetwork.saveNetwork()
