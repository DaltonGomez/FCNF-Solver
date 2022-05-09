from src.AntExperiments.AntResultsExperiment import AntResultsExperiment

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 run_ant_result.py
"""

if __name__ == "__main__":
    numAnts = 50
    numEpisodes = 15
    networkList = [
        "25-1-1",
    ]

    experiment = AntResultsExperiment(networkList, numAnts, numEpisodes)
    experiment.runExperiment()

"""
# ORIGINAL NETWORK LIST:
networkList = [
    "25-1-1",
    "25-1-5",
    "25-1-10",
    "50-1-1",
    "50-1-5",
    "50-1-10",
    "100-1-1",
    "100-1-5",
    "100-1-10",
    "200-1-1",
    "200-1-5",
    "200-1-10",
    "300-1-1",
    "300-1-5",
    "300-1-10",
    "400-1-1",
    "400-1-5",
    "400-1-10",
    ]
"""

"""
# SECOND NETWORK LIST:
networkList = [
    "25-1-1-2",
    "25-1-5-2",
    "25-1-10-2",
    "50-1-1-2",
    "50-1-5-2",
    "50-1-10-2",
    "100-1-1-2",
    "100-1-5-2",
    "100-1-10-2",
    "200-1-1-2",
    "200-1-5-2",
    "200-1-10-2",
    "300-1-1-2",
    "300-1-5-2",
    "300-1-10-2",
    "400-1-1-2",
    "400-1-5-2",
    "400-1-10-2" 
    ]
"""

"""
# THIRD NETWORK LIST:
networkList = [
    "25-1-1-3",
    "25-1-5-3",
    "25-1-10-3",
    "50-1-1-3",
    "50-1-5-3",
    "50-1-10-3",
    "100-1-1-3",
    "100-1-5-3",
    "100-1-10-3",
    "200-1-1-3",
    "200-1-5-3",
    "200-1-10-3",
    "300-1-1-3",
    "300-1-5-3",
    "300-1-10-3",
    "400-1-1-3",
    "400-1-5-3",
    "400-1-10-3"   
    ]
"""

"""
# NOT RUN
networkList = [
    "500-1-1",
    "500-1-5",
    "500-1-10",
    "1000-1-1",
    "1000-1-5",
    "1000-1-10"
    ]
"""
