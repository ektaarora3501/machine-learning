# creating a linear regression model for iris_dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def plot_regression_line(x,y,b):
    plt.scatter(x,y,color="g",marker='o',s=30)
    y_pred= b[0]+b[1]*x
    plt.plot(x,y_pred,color="m")
    plt.show()


def estimate_coeff(x,y):
    n=np.size(x)
    m_x=np.mean(x)
    m_y=np.mean(y)

    s_xx=np.sum(x*x) - n*m_x*m_x
    s_xy=np.sum(y*x) - n*m_y*m_x
    b1= s_xy / s_xx
    b0= m_y - b1*m_x

    return(b0,b1)


x=np.array([8,6,5,7,3])
y=np.array([1,3,3,2,5])

b=estimate_coeff(x,y)
print('expected value of b[0] ={} and b[1]={}'.format(b[0],b[1]))
# to create a plot ....
plot_regression_line(x,y,b)
