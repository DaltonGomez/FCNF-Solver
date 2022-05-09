from docplex.mp.model import Model

# SOURCE: https://ibmdecisionoptimization.github.io/tutorials/html/Beyond_Linear_Programming.html

im = Model(name='ip_telephone_production')
desk = im.integer_var(name='desk')
cell = im.integer_var(name='cell')
# write constraints
# constraint #AntDemo: desk production is greater than 100
im.add_constraint(desk >= 100)

# constraint #2: cell production is greater than 100
im.add_constraint(cell >= 100)

# constraint #3: assembly time limit
im.add_constraint(0.2 * desk + 0.4 * cell <= 401)

# constraint #4: painting time limit
im.add_constraint(0.5 * desk + 0.4 * cell <= 492)
im.maximize(12.4 * desk + 20.2 * cell)

si = im.solve()
im.print_solution()
