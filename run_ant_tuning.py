from datetime import datetime

from src.AntColony.AntColonyTuning import AntColonyTuning
from src.FlowNetwork.CandidateGraph import CandidateGraph

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 run_ant_tuning.py
"""

if __name__ == "__main__":
    graphFile = "25-1-1.p"
    graph = CandidateGraph()
    graph = graph.loadCandidateGraph(graphFile)
    targetFlow = 100

    now = datetime.now()
    timestamp = now.strftime("%d/%m/%Y %H:%M:%S")
    print("Starting Tuning Experiment At:")
    print(timestamp)

    for i in range(4):
        tuner = AntColonyTuning(graph, targetFlow)
        tuner.runExperiment()

    now = datetime.now()
    timestamp = now.strftime("%d/%m/%Y %H:%M:%S")
    print("Ending Tuning Experiment At:")
    print(timestamp)
