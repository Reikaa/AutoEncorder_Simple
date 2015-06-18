__author__ = 'Thushan Ganegedara'

import nlopt
import numpy
from math import sqrt



#class Example(object):

def myfunc(x,grad,a):
    if grad.size > 0:
        grad[0] = 0.0
        grad[1] = 0.5 / sqrt(x[1])
    return sqrt(x[1]+a)

def myconstraint(x, grad, a, b):
    if grad.size > 0:
        grad[0] = 3 * a * (a*x[0] + b)**2
        grad[1] = -1.0
    return (a*x[0] + b)**3 - x[1]

if __name__ == '__main__':


    opt = nlopt.opt(nlopt.LD_MMA, 2)
    opt.set_lower_bounds([-float('inf'), 0])
    opt.set_min_objective(lambda x,grad: myfunc(x,grad,0.5))
    opt.add_inequality_constraint(lambda x,grad: myconstraint(x,grad,2,0), 1e-8)
    opt.add_inequality_constraint(lambda x,grad: myconstraint(x,grad,-1,1), 1e-8)
    opt.set_xtol_rel(1e-4)
    init_val = numpy.asarray([1.234, 5.678])
    x = opt.optimize(init_val)
    minf = opt.last_optimum_value()
    print "optimum at ", x[0],x[1]
    print "minimum value = ", minf
    print "result code = ", opt.last_optimize_result()

