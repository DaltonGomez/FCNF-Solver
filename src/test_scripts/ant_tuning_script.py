from datetime import datetime

from src.AntExperiments.AntTuningExperiment import AntTuningExperiment
from src.Network.FlowNetwork import FlowNetwork

networkFile = "25-1-1.p"
network = FlowNetwork()
network = network.loadNetwork(networkFile)
targetFlow = 100

now = datetime.now()
timestamp = now.strftime("%d/%m/%Y %H:%M:%S")
print("Starting Tuning Experiment At:")
print(timestamp)

for i in range(4):
    tuner = AntTuningExperiment(network, targetFlow)
    tuner.runExperiment()

now = datetime.now()
timestamp = now.strftime("%d/%m/%Y %H:%M:%S")
print("Ending Tuning Experiment At:")
print(timestamp)
