
# coding: utf-8

# # 1 Matrix operations
# 
# ## 1.1 Create a 4*4 identity matrix

# In[56]:

#This project is designed to get familiar with python list and linear algebra
#You cannot use import any library yourself, especially numpy

A = [[1,2,3], 
     [2,3,3], 
     [1,2,5]]

B = [[1,2,3,5], 
     [2,3,3,5], 
     [1,2,5,1]]

#TODO create a 4*4 identity matrix 
I = []
for i in range(4):
    row = []
    for j in range(4):
        if j == i:
            row.append(1)
        else:
            row.append(0)
    I.append(row)
print I


# ## 1.2 get the width and height of a matrix. 

# In[57]:

#TODO Get the height and weight of a matrix.
def shape(M):
    Num_of_rows = len(M)
    Num_of_cols = len(M[0])
    return Num_of_rows, Num_of_cols


# ## 1.3 round all elements in M to certain decimal points

# In[58]:

# TODO in-place operation, no return value
# TODO round all elements in M to decPts

def matxRound(M, decPts=4):
    for i in M:
        for j in range(len(i)):
            i[j] = round(i[j], decPts)
    pass


# ## 1.4 compute transpose of M

# In[60]:

#TODO compute transpose of M
def transpose(M):
    new_matrix = []
    new_num_of_cols, new_num_of_rows = shape(M)
    for i in range(new_num_of_rows):
        row = []
        for j in range(new_num_of_cols):
            row.append(M[j][i])
        new_matrix.append(row)
    return new_matrix


# ## 1.5 compute AB. return None if the dimensions don't match

# In[81]:

# TODO compute matrix multiplication AB, return None if the dimensions don't match
def matxMultiply(A, B):
    rows_A, cols_A = shape(A)
    rows_B, cols_B = shape(B)
    if not cols_A == rows_B:
        return None
    else:
        new_matrix = []
        for i in range(rows_A):
            row = []
            for j in range(cols_B):
                row.append(sum_of_multiplication(A, B, i, j))
            new_matrix.append(row)
    return new_matrix


def sum_of_multiplication(A, B, row_of_A, col_of_B):
    """dot product of row and column
    """
    sum_of_multiplication = 0
    for i in range(len(A[row_of_A])):
        for j in range(len(B)):
            if j == i:
                sum_of_multiplication += A[row_of_A][i]*B[j][col_of_B]
    return sum_of_multiplication


# ## 1.6 Test your implementation

# In[74]:

# TODO test the shape function
A = [[1, 2, 3],
     [2, 3, 3],
     [1, 2, 5],
     [2, 3, 4]]
a, b = shape(A)
assert a == 4 and b == 3, "shape function test not pass"

# TODO test the round function
A = [[1.23333, 2.33334],
     [3.532334, 4.56755]]
B = [[1.23, 2.33],
     [3.53, 4.57]]
matxRound(A, 2)
assert A == B, "round function test not pass"

# TODO test the transpose funtion
A = [[1, 2, 3],
     [2, 3, 3],
     [1, 2, 5],
     [2, 3, 4]]
B = [[1, 2, 1, 2],
     [2, 3, 2, 3],
     [3, 3, 5, 4]]
C = [[14, 17, 20, 20],
     [17, 22, 23, 25],
     [20, 23, 30, 28],
     [20, 25, 28, 29]]
assert B == transpose(A), "transpose function test not pass"

# TODO test the matxMultiply function, when the dimensions don't match
assert matxMultiply(A, C) is None, "matxMultiply function test not pass"

# TODO test the matxMultiply function, when the dimensions do match
assert C == matxMultiply(A, B), "matxMultiply function test not pass"



# # 2 Gaussian Jordan Elimination
# 
# ## 2.1 Compute augmented Matrix 
# 
# $ A = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n}\\
#     a_{21}    & a_{22} & ... & a_{2n}\\
#     a_{31}    & a_{22} & ... & a_{3n}\\
#     ...    & ... & ... & ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn}\\
# \end{bmatrix} , b = \begin{bmatrix}
#     b_{1}  \\
#     b_{2}  \\
#     b_{3}  \\
#     ...    \\
#     b_{n}  \\
# \end{bmatrix}$
# 
# Return $ Ab = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n} & b_{1}\\
#     a_{21}    & a_{22} & ... & a_{2n} & b_{2}\\
#     a_{31}    & a_{22} & ... & a_{3n} & b_{3}\\
#     ...    & ... & ... & ...& ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn} & b_{n} \end{bmatrix}$

# In[75]:

#TODO construct the augment matrix of matrix A and column vector b, assuming A and b have same number of rows
def augmentMatrix(A, b):
    for i in range(len(A)):
        A[i].append(b[i][-1])
    return A


# ## 2.2 Basic row operations
# - exchange two rows
# - scale a row
# - add a scaled row to another

# In[77]:

# TODO r1 <---> r2
# TODO in-place operation, no return value
def swapRows(M, r1, r2):
    M[r1], M[r2] = M[r2], M[r1]
    pass


# TODO r1 <--- r1 * scale
# TODO in-place operation, no return value
def scaleRow(M, r, scale):
    if scale == 0:
        raise ValueError
    for i in range(len(M[r])):
        M[r][i] = M[r][i] * scale
    pass


# TODO r1 <--- r1 + r2*scale
# TODO in-place operation, no return value
def addScaledRow(M, r1, r2, scale):
    scaled_row = [i * scale for i in M[r2]]
    M[r1] = [(M[r1][i] + scaled_row[i]) for i in range(len(M[r1]))]
    pass
                                            


# ## 2.3  Gauss-jordan method to solve Ax = b
# 
# ### Hint：
# 
# Step 1: Check if A and b have same number of rows
# Step 2: Construct augmented matrix Ab
# 
# Step 3: Column by column, transform Ab to reduced row echelon form [wiki link](https://en.wikipedia.org/wiki/Row_echelon_form#Reduced_row_echelon_form)
#     
#     for every column of Ab (except the last one)
#         column c is the current column
#         Find in column c, at diagnal and under diagnal (row c ~ N) the maximum absolute value
#         If the maximum absolute value is 0
#             then A is singular, return None （Prove this proposition in Question 2.4）
#         else
#             Apply row operation 1, swap the row of maximum with the row of diagnal element (row c)
#             Apply row operation 2, scale the diagonal element of column c to 1
#             Apply row operation 3 mutiple time, eliminate every other element in column c
#             
# Step 4: return the last column of Ab
# 
# ### Remark：
# We don't use the standard algorithm first transfering Ab to row echelon form and then to reduced row echelon form.  Instead, we arrives directly at reduced row echelon form. If you are familiar with the stardard way, try prove to yourself that they are equivalent. 

# In[113]:

# TODO implement gaussian jordan method to solve Ax = b

""" Gauss-jordan method to solve x such that Ax = b.
        A: square matrix, list of lists
        b: column vector, list of lists
        decPts: degree of rounding, default value 4
        epsilon: threshold for zero, default value 1.0e-16    
    return x such that Ax = b, list of lists 
    return None if A and b have same height
    return None if A is (almost) singular
"""
from copy import deepcopy
def eliminate_other_elmts(M, row_index, col_index):
    """eliminate other elements in  the same column
    """
    dimension = len(M)
    for i in range(dimension):
        if i == row_index:
            continue
        scale = - M[i][col_index]
        addScaledRow(M, i, row_index, scale)
    pass


def gj_Solve(A, b, decPts = 4, epsilon = 1.0e-16):
    A_1 = deepcopy(A)
    if not len(A_1) == len(b):
        return None  # return None if A and b have different height
    dimension = len(A_1)
    augmentMatrix(A_1, b)
    for col in range(dimension):
        max_of_the_col = 0
        row_of_max = 0
        for row in range(col, dimension):
            if abs(A_1[row][col]) > max_of_the_col:
                max_of_the_col = abs(A_1[row][col])
                row_of_max = row
        if abs(max_of_the_col) < epsilon:
            return None  # return None if A is (almost) singular
        if max_of_the_col > A_1[col][col]:
            swapRows(A_1, row_of_max, col)
        scaleRow(A_1, col, 1. / (A_1[col][col]))
        eliminate_other_elmts(A_1, col, col)
        last_col_Decimal = transpose(A_1)[-1]
        last_col = [round(i, decPts) for i in last_col_Decimal]

    return [[last_col[i]] for i in range(len(last_col))]


# ## 2.4 Prove the following proposition:
# 
# **If square matrix A can be divided into four parts: ** 
# 
# $ A = \begin{bmatrix}
#     I    & X \\
#     Z    & Y \\
# \end{bmatrix} $, where I is the identity matrix, Z is all zero and the first column of Y is all zero, 
# 
# **then A is singular.**
# 
# Hint: There are mutiple ways to prove this problem.  
# - consider the rank of Y and A
# - consider the determinate of Y and A 
# - consider certain column is the linear combination of other columns

# TODO Please use latex （refering to the latex in problem may help）
# 
# TODO Proof：
# 
# **Since I is invertible, also I is identity matrix**
#    
# $ det(A) = det(I) \cdot det(Y - Z \cdot I^{-1} \cdot X)\\
#   = 1 \cdot det(Y-0\cdot I^{-1} \cdot X)\\
#   = det(Y)\\$ 
#   
# **Since first column of Y is all zero:**
# 
# $  Y^{T} = \begin{bmatrix}
#        0 & 0 & 0&\dots & 0\\
#        a_{21} & a_{22}&a_{23}  & \dots & a_{2n}\\
#        a_{31} & a_{32}&a_{33}  & \dots & a_{3n}\\
#        \dots & \dots & \dots&\dots & \dots \\
#        a_{n1} & a_{n2} &a_{n3} & \dots & a_{nn}\\
#        \end{bmatrix}$
#        
# $ det(Y^{T}) = a_{11} Y11 - a_{12} Y12 + a_{13} Y13 - . . . + a_{1n} Y1n \\
#  = 0 \cdot Y11 - 0\cdot Y12 + 0\cdot Y13 - . . . + 0\cdot Y1n  \\
#  = 0 \\
#           $
#           
#  **Since:**
#  
#  $ det(Y) = det(Y^{T})\\$
#  
#  **So:**
#   
#  $ det(Y) = 0 \\$ 
#  
#  **Then:**
#  
#  $ det(A) = 0 \\ $
#   
#   **Then A is singular**
#   
#   
#  

# ## 2.5 Test your gj_Solve() implementation

# In[131]:

# construct A and b where A is singular
# construct A and b where A is not singular
# solve x for  Ax = b 
# compute Ax
# compare Ax and b

# TODO
A_singular = [[0, 2, 3],
              [0, 3, 3],
              [0, 2, 5]]
A = [[1, 2, 1],
     [2, 3, 3],
     [1, 2, 5]]
b = [[2], [2], [4]]
x = gj_Solve(A, b, decPts = 4, epsilon = 1.0e-16)
assert gj_Solve(A_singular, b, decPts = 4, epsilon = 1.0e-16) is None, "Should not compute Ax if A is singular"
Ax = matxMultiply(A, x)
assert Ax == b, "Ax is not equal to b"


# # 3 Linear Regression: 
# 
# ## 3.1 Compute the gradient of loss function with respect to parameters 
# ## (Choose one between two 3.1 questions)
# 
# We define loss funtion E as 
# $$
# E(m, b) = \sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$
# 
# Proves that 
# $$
# \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)}
# $$
# 
# $$
# \frac{\partial E}{\partial b} = \sum_{i=1}^{n}{-2(y_i - mx_i - b)}
# $$
# 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} = 2X^TXh - 2X^TY
# $$
# 
# $$ 
# \text{, where }
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}
# $$

# TODO Please use latex （refering to the latex in problem may help）
# 
# TODO Proof：
# 
# **Since:**
# 
# $
# E(m, b) = \sum_{i=1}^{n}{(y_i - mx_i - b)^2}\\
# $
# 
# **Then,according to chain rule:**
# 
# $
# \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{2(y_i - mx_i - b)(-mx_i)'}\\
# \frac{\partial E}{\partial b} = \sum_{i=1}^{n}{2(y_i - mx_i - b)(-b)'}\\
# $
# 
# 
# **Then:**
# 
# $
# \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)}\\
# \frac{\partial E}{\partial b} = \sum_{i=1}^{n}{-2(y_i - mx_i - b)}\\
# $
# 
# 
# **Since:**
# 
# $ 
# \text{, where }
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}
# $
# 
# **Then:**
# 
# $
# 2X^TXh = 2 \begin{bmatrix}
#     x_1 & x_2&...&x_n \\
#     1 & 1 & ... &1\\    
#     \end{bmatrix} 
# \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix}
# \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix} \\
#   = 2 \begin{bmatrix}
#     x_1^{2}+x_2^{2}+...+x_n^{2} &x_1+x_2+...x_n \\
#     x_1+x_2+...x_n &n\\    
#     \end{bmatrix}
#     \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix} \\
# =2 \begin{bmatrix}
#     (x_1^{2}m+x_1b)+(x_2^{2}m+x_2b)+...+(x_n^{2}m+x_nb)\\
#     (mx_1+b)+(mx_2+b)+...+(mx_n+b)\\    
#     \end{bmatrix}
# $
# 
# 
# $
# 2X^TY = 2 \begin{bmatrix}
#     x_1 & x_2&...&x_n \\
#     1 & 1 & ... &1\\    
#     \end{bmatrix}
#     \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n  \\
# \end{bmatrix}
# = 2 \begin{bmatrix}
#     x_1y_1+x_2y_2+ ... + x_ny_n \\
#     y_1+y_2+...+y_n\\    
#     \end{bmatrix}
# $
# 
# **Then:**
# 
# $
# 2X^TXh - 2X^TY = 2 \begin{bmatrix}
#     (x_1^{2}m+x_1b)+(x_2^{2}m+x_2b)+...+(x_n^{2}m+x_nb)\\
#     (mx_1+b)+(mx_2+b)+...+(mx_n+b)\\    
#     \end{bmatrix} - 2 \begin{bmatrix}
#     x_1y_1+x_2y_2+ ... + x_ny_n \\
#     y_1+y_2+...+y_n \\    
#     \end{bmatrix}
#     = 2 \begin{bmatrix}
#     (x_1^{2}m+x_1b-x_1y_1)+(x_2^{2}m+x_2b-x_2y_2)+...(x_n^{2}m+x_nb-x_ny_n)\\
#     (mx_1+b-y_1)+(mx_2+b-y_2)+...+(mx_n+b-y_n)\\
#     \end{bmatrix}
#     = \begin{bmatrix}
#     \sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)}\\
#     \sum_{i=1}^{n}{-2(y_i - mx_i - b)}\\
#     \end{bmatrix}
#     =  \begin{bmatrix}
#     \frac{\partial E}{\partial m}\\
#     \frac{\partial E}{\partial b}\\
#     \end{bmatrix}
# $

# ## 3.1 Compute the gradient of loss function with respect to parameters 
# ## (Choose one between two 3.1 questions)
# 
# Proves that 
# $$
# E = Y^TY -2(Xh)^TY + (Xh)^TXh
# $$
# 
# $$
# \frac{\partial E}{\partial h} = 2X^TXh - 2X^TY
# $$
# 
# $$ 
# \text{,where }
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}
# $$

# TODO Please use latex （refering to the latex in problem may help）
# 
# TODO Proof：

# ## 3.2  Linear Regression
# ### Solve equation $X^TXh = X^TY $ to compute the best parameter for linear regression.

# In[138]:

#TODO implement linear regression 
'''
points: list of (x,y) tuple
return m and b
'''
def linearRegression(points):
    X_value = [i[0] for i in points]
    Y_value = [i[1] for i in points]
    X = [[i,1] for i in X_value]
    Y = [[i] for i in Y_value]   
    A = matxMultiply(transpose(X), X)
    b = matxMultiply(transpose(X), Y)
    m, b = (i[0] for i in gj_Solve(A, b, decPts=4, epsilon = 1.0e-16))
    return m, b


# ## 3.3 Test your linear regression implementation

# In[159]:

#TODO Construct the linear function
m = 3  # set m for original linear function for testing
b = 10  # set b for original linear function for testing

#TODO Construct points with gaussian noise
import random
num_points = 100
X_pure = [random.uniform(1,100) for i in range(num_points)]
Y_pure = [i * m + b for i in X_pure] 
mu = 0  # set mu for gauss noise
sigma = 2  # set sigma for gauss noise
X_noise = [random.gauss(mu, sigma) for _ in range(num_points)]
Y_noise = [random.gauss(mu, sigma) for _ in range(num_points)]
X_value = [x + y for x,y in zip(X_pure, X_noise)]
Y_value = [x + y for x,y in zip(Y_pure, Y_noise)]
points = [(x, y) for x,y in zip(X_value,Y_value)]

#TODO Compute m and b and compare with ground truth
m_result, b_result = linearRegression(points)
error_m = m_result - m
error_b = b_result - b
print error_m, error_b


# ## 4.1 Unittests
# 
# Please make sure yout implementation passed all the unittests

# In[160]:

import unittest
import numpy as np

from decimal import *

class LinearRegressionTestCase(unittest.TestCase):
    """Test for linear regression project"""

    def test_shape(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.randint(low=-10,high=10,size=(r,c))
            self.assertEqual(shape(matrix.tolist()),(r,c))


    def test_matxRound(self):

        for decpts in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()
            dec_true = [[Decimal(str(round(num,decpts))) for num in row] for row in mat]

            matxRound(mat,decpts)
            dec_test = [[Decimal(str(num)) for num in row] for row in mat]

            res = Decimal('0')
            for i in range(len(mat)):
                for j in range(len(mat[0])):
                    res += dec_test[i][j].compare_total(dec_true[i][j])

            self.assertEqual(res,Decimal('0'))


    def test_transpose(self):
        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()
            t = np.array(transpose(mat))

            self.assertEqual(t.shape,(c,r))
            self.assertTrue((matrix.T == t).all())


    def test_matxMultiply(self):

        for _ in range(10):
            r,d,c = np.random.randint(low=1,high=25,size=3)
            mat1 = np.random.randint(low=-10,high=10,size=(r,d)) 
            mat2 = np.random.randint(low=-5,high=5,size=(d,c)) 
            dotProduct = np.dot(mat1,mat2)

            dp = np.array(matxMultiply(mat1,mat2))

            self.assertTrue((dotProduct == dp).all())


    def test_augmentMatrix(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            A = np.random.randint(low=-10,high=10,size=(r,c))
            b = np.random.randint(low=-10,high=10,size=(r,1))

            Ab = np.array(augmentMatrix(A.tolist(),b.tolist()))
            ab = np.hstack((A,b))

            self.assertTrue((Ab == ab).all())

    def test_swapRows(self):
        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            r1, r2 = np.random.randint(0,r, size = 2)
            swapRows(mat,r1,r2)

            matrix[[r1,r2]] = matrix[[r2,r1]]

            self.assertTrue((matrix == np.array(mat)).all())

    def test_scaleRow(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            rr = np.random.randint(0,r)
            with self.assertRaises(ValueError):
                scaleRow(mat,rr,0)

            scale = np.random.randint(low=1,high=10)
            scaleRow(mat,rr,scale)
            matrix[rr] *= scale

            self.assertTrue((matrix == np.array(mat)).all())
    
    def test_addScaleRow(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            r1,r2 = np.random.randint(0,r,size=2)

            scale = np.random.randint(low=1,high=10)
            addScaledRow(mat,r1,r2,scale)
            matrix[r1] += scale * matrix[r2]

            self.assertTrue((matrix == np.array(mat)).all())


    def test_gj_Solve(self):

        for _ in range(10):
            r = np.random.randint(low=3,high=10)
            A = np.random.randint(low=-10,high=10,size=(r,r))
            b = np.arange(r).reshape((r,1))
            

            x = gj_Solve(A.tolist(),b.tolist())
            
            if np.linalg.matrix_rank(A) < r:
                self.assertEqual(x,None)
            else:
                Ax = np.dot(A,np.array(x))
                loss = np.mean((Ax - b)**2)
                
                self.assertTrue(loss<0.1)


suite = unittest.TestLoader().loadTestsFromTestCase(LinearRegressionTestCase)
unittest.TextTestRunner(verbosity=3).run(suite)


# In[ ]:



