from datetime import datetime

from src.Network.FlowNetwork import FlowNetwork

networkFile = "medium.p"
network = FlowNetwork()
network = network.loadNetwork(networkFile)
targetFlow = 600

now = datetime.now()
timestamp = now.strftime("%d/%m/%Y %H:%M:%S")
print("Starting Tuning Experiment At:")
print(timestamp)

for i in range(4):
    tuner = ACOTuningExperiment(network, targetFlow)
    tuner.runExperiment()

now = datetime.now()
timestamp = now.strftime("%d/%m/%Y %H:%M:%S")
print("Ending Tuning Experiment At:")
print(timestamp)
