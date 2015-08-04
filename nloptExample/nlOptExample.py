__author__ = 'Thushan Ganegedara'

import nlopt
import numpy
from math import sqrt



class Example(object):

    def ex(self):
        def myfunc(x, grad):
            if grad.size > 0:
                grad[0] = 0.0
                grad[1] = 0.5 / sqrt(x[1])
            print x
            return sqrt(x[1])

        def myconstraint(x, g, a, b):
            if g.size > 0:
                g[0] = 3 * a * (a*x[0] + b)**2
                g[1] = -1.0
            return (a*x[0] + b)**3 - x[1]

        opt = nlopt.opt(nlopt.LD_SLSQP, 2)
        opt.set_lower_bounds([-float('inf'), 0])
        opt.set_min_objective(lambda x,grad : myfunc(x,grad))
        opt.add_inequality_constraint(lambda x,grad: myconstraint(x,grad,2,0), 1e-8)
        opt.add_inequality_constraint(lambda x,grad: myconstraint(x,grad,-1,1), 1e-8)
        opt.set_xtol_rel(1e-12)
        opt_x = opt.optimize([1.234, 5.678])
        minf = opt.last_optimum_value()
        print "optimum at ", opt_x[0],opt_x[1]
        print "minimum value = ", minf
        print "result code = ", opt.last_optimize_result()

if __name__ == '__main__':

    e = Example()
    e.ex()