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

        my_expr = expressions[0]

        from sympy import symbols
        in1, in2, out1, out2 = symbols("in1, in2, out1, out2")


        from devito.symbolics import retrieve_indexed
        symbols = retrieve_indexed(my_expr)

        symbols_in_order = [symbols[1], symbols[2], symbols[0]]

        first_subs = my_expr.xreplace({symbols_in_order[0]:in1})

        expected_str = "Eq(f1[time + 1, x, y], in1 + f1[time, x, y] + 1)"
        try:
            assert str(first_subs) == expected_str 
        except:
            print("Expected: \n %s \nReturned: \n %s" % \
                (expected_str, first_subs))

        # from IPython import embed; embed()
        return [LoweredEq(i) for i in expressions]

operator = InPlaceOperator(equation)
# print(operator)
