from ortools.graph import pywrapgraph
# SOURCE: https://developers.google.com/optimization/flow/mincostflow

def main():
    """MinCostFlow simple interface example."""
    # Instantiate a SimpleMinCostFlow solver.
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()

    # Define four parallel arrays: sources, destinations, capacities,
    # and unit costs between each pair. For instance, the arc from node 0
    # to node 1 has a capacity of 15.
    start_nodes = [0, 0, 1, 1, 1, 2, 2, 3, 4]
    end_nodes = [1, 2, 2, 3, 4, 3, 4, 4, 2]
    capacities = [15, 8, 20, 4, 10, 15, 4, 20, 5]
    unit_costs = [4, 4, 2, 2, 6, 1, 3, 2, 3]

    # Define an array of supplies at each node.
    supplies = [20, 0, 0, -5, -15]

    # Add each arc.
    for arc in zip(start_nodes, end_nodes, capacities, unit_costs):
        min_cost_flow.AddArcWithCapacityAndUnitCost(arc[0], arc[1], arc[2],
                                                    arc[3])

    # Add node supply.
    for count, supply in enumerate(supplies):
        min_cost_flow.SetNodeSupply(count, supply)

    # Find the min cost flow.
    status = min_cost_flow.Solve()

    if status != min_cost_flow.OPTIMAL:
        print('There was an issue with the min cost flow input.')
        print(f'Status: {status}')
        exit(1)
    print('Minimum cost: ', min_cost_flow.OptimalCost())
    print('')
    print(' Arc   Flow / Capacity  Cost')
    for i in range(min_cost_flow.NumArcs()):
        cost = min_cost_flow.Flow(i) * min_cost_flow.UnitCost(i)
        print('%1s -> %1s    %3s   / %3s   %3s' %
              (min_cost_flow.Tail(i), min_cost_flow.Head(i),
               min_cost_flow.Flow(i), min_cost_flow.Capacity(i), cost))


if __name__ == '__main__':
    main()
