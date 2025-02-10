import bnlearn as bn
from contextlib import redirect_stdout
from pgmpy.factors.discrete import TabularCPD
from tabulate import tabulate
import time

def execute_baysian_network():
    cpd_A = TabularCPD(variable='A', variable_card=2, values=[[0.13], [0.87]])  # P(A)
    cpd_B = TabularCPD(variable='B', variable_card=2, values=[[0.38], [0.62]])  # P(B)
    cpd_C = TabularCPD(
        variable='C', variable_card=2,
        values=[
            [0.65, 0.02, 0.94, 0.82],
            [0.35, 0.98, 0.06, 0.18]   
        ],
        evidence=['A', 'B'], evidence_card=[2, 2]
    )
    cpd_D = TabularCPD(
        variable='D', variable_card=2,
        values=[
            [0.97, 0.54],
            [0.03, 0.46]
        ],
        evidence=['C'], evidence_card=[2]
    )
    cpd_E = TabularCPD(variable='E', variable_card=2, values=[[0.05], [0.95]])
    cpd_F = TabularCPD(variable='F', variable_card=2, values=[[0.71], [0.29]])
    cpd_G = TabularCPD(
        variable='G', variable_card=2,
        values=[
            [0.81, 0.93, 0.55, 0.99, 0.24, 0.52, 0.79, 0.68],
            [0.19, 0.07, 0.45, 0.01, 0.76, 0.48, 0.21, 0.32]
        ],
        evidence=['D', 'E', 'F'], evidence_card=[2, 2, 2]
    )
    cpd_H = TabularCPD(
        variable='H', variable_card=2,
        values=[
            [0.21, 0.72],
            [0.79, 0.28]
        ],
        evidence=['G'], evidence_card=[2]
    )
    cpd_I = TabularCPD(
        variable='I', variable_card=2,
        values=[
            [0.66, 0.88],
            [0.34, 0.12]
        ],
        evidence=['C'], evidence_card=[2]
    )
    cpd_J = TabularCPD(
        variable='J', variable_card=2,
        values=[
            [0.44, 0.09],
            [0.56, 0.91]
        ],
        evidence=['C'], evidence_card=[2]
    )

    startTotal = time.time()
    edges = [('A', 'C'), ('B', 'C'), ('C', 'D'), ('D', 'G'), ('E', 'G'), ('F', 'G'), ('G', 'H'), ('C', 'I'), ('C', 'J')]
    model_complete = bn.make_DAG(edges, [cpd_A, cpd_B, cpd_C, cpd_D, cpd_E, cpd_F, cpd_G, cpd_H, cpd_I, cpd_J])
    dot = bn.plot_graphviz(model_complete)
    dot.render(f"Ground_Truth_Model")
    for i in range(0, 10):
        fullIterationStart = time.time()
        print(f"Starting Iteration {i + 1}")
        j = 100
        while j <= 100000:
            sampleSizeStart = time.time()
            print(f"Starting Sample Size {j} of Iteration {i + 1}")
            df = bn.sampling(model_complete, n=j)
            model_new = bn.structure_learning.fit(df)
            model_new_with_params = bn.parameter_learning.fit(model_new, df)
            model = bn.make_DAG(edges)
            model_with_params = bn.parameter_learning.fit(model, df)
            dot = bn.plot_graphviz(model_new_with_params)
            dot.render(f"Iteration_{i + 1}_Sample_Size_{j}_model_new_with_params")
            print(f"Learned DAG Structure:")
            with open('nul', 'w') as f:
                with redirect_stdout(f):
                    cpd = bn.print_CPD(model_with_params)
            
            for key in cpd.keys():
                print(tabulate(cpd[key], tablefmt="grid", headers="keys"))    
            
            print(f"Iteration {i + 1} Sample Size {j} Learned CPD Learned Structure")
            with open('nul', 'w') as f:
                with redirect_stdout(f):
                    cpd = bn.print_CPD(model_new_with_params)
            
            for key in cpd.keys():
                print(tabulate(cpd[key], tablefmt="grid", headers="keys"))    

            print(f"Sample Size {j} of Iteration {i + 1} took {time.time() - sampleSizeStart} seconds")
            j *= 10

        print(f"Iteration {i + 1} took {time.time() - fullIterationStart} seconds")

    print(f"10 iterations took {time.time() - startTotal}\n")

execute_baysian_network()