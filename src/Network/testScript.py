from src.Network.FlowNetwork import FlowNetwork
from src.Network.NetworkVisualizer import NetworkVisualizer

name = "test-50-3-3.p"
flowNetwork = FlowNetwork()
flowNetwork = flowNetwork.loadNetwork(name)
flowNetwork.drawNetworkTriangulation()

visualizer = NetworkVisualizer(flowNetwork)
visualizer.drawGraph()

flowNetwork.printAllNodeData()
