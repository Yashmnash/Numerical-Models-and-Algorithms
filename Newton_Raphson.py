# Newton Raphson Method
# By Yash Desai


def function(x):
    return (x - 3)**2

# dfunction is ther derivative of function
def dfunction(x):
    return 2*(x-3)



def newton_raphson(f, df, initial, e):

    x = initial

    while(f(x) > e):

        x = x - (function(x)/dfunction(x))

    return x
    
# Shows convergence to root from below and above.
print(newton_raphson(function,dfunction, 2, 0.0001))
print(newton_raphson(function,dfunction, 4, 0.0001))
