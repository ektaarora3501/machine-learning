from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import mglearn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def estimate_coeff(x,y):
    n = np.size(x)
    # calculating mean of x and y
    m_x = np.mean(x)
    m_y = np.mean(y)

    s_xy = np.sum(x*y) -n*m_x*m_y
    s_xx = np.sum(x*x) -n*m_x*m_x

    b1= s_xy / s_xx
    b0= m_y- b1*m_x

    return(b0,b1)


def plot_regression_line(x,y,b):
    plt.scatter(x,y,color='m',marker='o',s=30)

    y_pred = b[0] + b[1]*x
    plt.plot(x,y_pred,color="g")

    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


x=np.array([0,1,2,3,4,5,6,7,8,9])
y=np.array([1,3,2,5,7,8,8,9,10,12])

b=estimate_coeff(x,y)

print("value of estimated coefficient b[0]={} b1={} ".format(b[0],b[1]))

plot_regression_line(x,y,b)
