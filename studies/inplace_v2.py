from devito import TimeFunction, Grid, Eq
from devito.data.allocators import ExternalAllocator
from utils import generate_data, external_initializer
import numpy as np

shape=(10,10)
space_order=2
time_order=2
save=2
numpy_array, devito_grid = generate_data(shape=shape, space_order=space_order, save=save)

f1 = TimeFunction(name='f1', 
                  grid=devito_grid, 
                  space_order=space_order,
                  allocator=ExternalAllocator(numpy_array), 
                  initializer=external_initializer,
                  time_order=time_order,
                  save=save)

# print(f1.data)
equation = Eq(f1.forward, f1 + f1.backward + 1)
# print(equation)

##

from devito.operator import Operator
from devito.ir.equations import LoweredEq

class InPlaceOperator(Operator):

    def __init__(self, *args, **kwargs):
        super(InPlaceOperator, self).__init__(*args, **kwargs)

    def _specialize_exprs(self, expressions):

        expr_in = expressions[0]
        mapping = []
        exprs_out = []

        from sympy import symbols
        in1, in2, out1, out2 = symbols("in1, in2, out1, out2")

        from devito.symbolics import retrieve_indexed
        symbols = retrieve_indexed(expr_in)
        symbols_in_order = [symbols[1], symbols[2], symbols[0]]
        extended_symbols_in_order = [in2, in1] + symbols_in_order + [out1, out2]

        # first pass
        mapping.append({extended_symbols_in_order[2]:extended_symbols_in_order[0],
                        extended_symbols_in_order[3]:extended_symbols_in_order[1],
                        extended_symbols_in_order[4]:extended_symbols_in_order[2]})
        for i in range(4):
            mapping.append({extended_symbols_in_order[i+2]:extended_symbols_in_order[i+3],
                            extended_symbols_in_order[i+1]:extended_symbols_in_order[i+2],
                            extended_symbols_in_order[i+0]:extended_symbols_in_order[i+1]})

        tmp_expr = expr_in
        for i in range(5):        
            exprs_out.append(tmp_expr.xreplace(mapping[i]))
            tmp_expr = tmp_expr.xreplace(mapping[i])

        # exprs_out.append(tmp_expr3)

        print("\nInput expr:\n %s\n\nOutput exprs:" % expr_in)
        for n, i in enumerate(exprs_out):
            print(" %s) %s" % (n+1,i))

        # from slack
        expected_strs = [\
            "Eq(f1[time - 1, x, y], in1 + in2 + 1)", 
            "Eq(f1[time, x, y], in1 + f1[time - 1, x, y] + 1)",
            "Eq(f1[time + 1, x, y], f1[time, x, y] + f1[time - 1, x, y] + 1)",
            "Eq(out1, f1[time, x, y] + f1[time - 1, x, y] + 1)",
            "Eq(out2, f1[time, x, y] + out1 + 1)"]

        print ('\n*RESULT*')
        for n, i in enumerate(exprs_out):
            print (' %s:' % (n+1), end='')
            try:
                assert str(exprs_out[n]) == expected_strs[n]
                print (' PASS')
            except:
                print(" FAIL\n\tExpected:\n\t %s\n\tReturned:\n\t %s" % \
                    (expected_strs[n], exprs_out[n]))

        # from IPython import embed; embed()
        return [LoweredEq(i) for i in expressions]

operator = InPlaceOperator(equation)
# print(operator)