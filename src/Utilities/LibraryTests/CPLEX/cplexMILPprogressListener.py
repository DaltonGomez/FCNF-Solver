from docplex.mp.model import Model
from docplex.mp.progress import *

"""
Tutorial on adding a Progress Listener to a CPLEX instance to extract runtime data
SOURCE: https://dataplatform.cloud.ibm.com/exchange/public/entry/view/6e2bffa5869dacbae6500c7037ecd36f
"""

def build_hearts(r, **kwargs):
    # initialize the model
    mdl = Model('love_hearts_%d' % r, log_output=False, **kwargs)

    # the dictionary of decision variables, one variable
    # for each circle with i in (1 .. r) as the row and
    # j in (1 .. i) as the position within the row
    idx = [(i, j) for i in range(1, r + 1) for j in range(1, i + 1)]
    a = mdl.binary_var_dict(idx, name=lambda ij: "a_%d_%d" % ij)

    # the constraints - enumerate all equilateral triangles
    # and prevent any such triangles being formed by keeping
    # the number of included circles at its vertexes below 3

    # for each row except the last
    for i in range(1, r):
        # for each position in this row
        for j in range(1, i + 1):
            # for each triangle of side length (k) with its upper vertex at
            # (i, j) and its sides parallel to those of the overall shape
            for k in range(1, r - i + 1):
                # the sets of 3 points at the same distances clockwise along the
                # sides of these triangles form k equilateral triangles
                for m in range(k):
                    u, v, w = (i + m, j), (i + k, j + m), (i + k - m, j + k - m)
                    mdl.add(a[u] + a[v] + a[w] <= 2)

    mdl.maximize(mdl.sum(a))
    return mdl


m5 = build_hearts(5)
m5.print_information()
# connect a listener to the model
progressRecorder = ProgressDataRecorder(clock="gap")
m5.add_progress_listener(progressRecorder)
m5.solve(clean_before_solve=False)

print(progressRecorder.recorded)
print(type(progressRecorder.recorded))
print(type(progressRecorder.recorded[0]))
print(len(progressRecorder.recorded[0]))
print(type(progressRecorder.recorded[0][2]))

for progress in progressRecorder.iter_recorded:
    print(progress)
m5.print_solution()