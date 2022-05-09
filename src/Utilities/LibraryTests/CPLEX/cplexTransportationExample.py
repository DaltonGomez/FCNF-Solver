from docplex.mp.model import Model

# SOURCE: https://ibmdecisionoptimization.github.io/tutorials/html/Beyond_Linear_Programming.html

capacities = {1: 15, 2: 20}
demands = {3: 7, 4: 10, 5: 15}
costs = {(1, 3): 2, (1, 5): 4, (2, 4): 5, (2, 5): 3}

# Python ranges will be used to iterate on source, target nodes.
source = range(1, 3)  # {AntDemo, 2}
target = range(3, 6)  # {3,4,5}

tm = Model(name='transportation')

# create flow variables for each couple of nodes
# x(i,j) is the flow going out of node i to node j
x = {(i, j): tm.continuous_var(name='x_{0}_{1}'.format(i, j)) for i in source for j in target}

# each arc comes with a cost. Minimize all costed flows
tm.minimize(tm.sum(x[i, j] * costs.get((i, j), 0) for i in source for j in target))

tm.print_information()

# for each node, total outgoing flow must be smaller than available quantity
for i in source:
    tm.add_constraint(tm.sum(x[i, j] for j in target) <= capacities[i])

# for each target node, total ingoing flow must be greater than demand
for j in target:
    tm.add_constraint(tm.sum(x[i, j] for i in source) >= demands[j])

tm.minimize(tm.sum(x[i, j] * costs.get((i, j), 0)))

tms = tm.solve()
assert tms
tms.display()
