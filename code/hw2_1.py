import numpy as np # for matrix 

# define matrix where x0 =1 x1 = star wars, x2 = god father, x3 = parasite 
tilda_X = np.array([
    [1,5,1,2], 
    [1,1,3,5],
    [1,1,5,3], 
    [1,2,1,1], 
    [1,2,1,2]
])
Y = np.array([[5], [3], [1], [2], [2]])
tilda_X_transpose = tilda_X.T
# part 1- (a) : get w and b parameter 

# calculate x^t * x 
X_transpo_X = np.dot(tilda_X_transpose,tilda_X)
print("tildaX^T * tildaX", X_transpo_X)

# check determinant of (x^t * x ) 
det_X = np.linalg.det(X_transpo_X)
print("Check determinant to see whether we can get inverse of (x^t * x )", det_X)

def calculate_w_b(): 
    if (det_X > 0 ): 
        # calculate inverse of x^t * x 
        inverse_X = np.linalg.inv(X_transpo_X)  
        print("inverse of (X transpo * X)\n", inverse_X)
        # calculate w = (x^t*x)-1 * ( x^t * y )
        matrix = np.dot(tilda_X_transpose, Y)
        w = np.dot(inverse_X ,matrix)
        print("Check W matrix\n", w)
        print("Check w size : ", w.shape)
        
        # y = tila w * tild x where w0 = b and x0 = 1 based on lecture note. 
        if (w.size > 0): 
            b = w[0] # first row of vector w = b 
            return w, b 

w , b = calculate_w_b(); 
print("W :\n", w , "\n", "b : \n", b)

        
# part1 -b use ridge gression 
# w_tilda = (x_tilda transpo * x_tilda + alpha * I ) ^ -1 * ( x tilda transpo * y )
print("Part1- b Ridge regression method \n")
alpha = 0.1
I = np.eye(tilda_X.shape[1])  # get identity matrix of tilda x and .shape[1] return # of feature
# multiply tilda_x_transpo * tilda_x 
X_transpo_X = np.dot(tilda_X_transpose, tilda_X)
print("tilda_X_transpose : \n" ,tilda_X_transpose)
print("tilda_X: \n", tilda_X)
print("dot prodcut of (tilda_X)^t * (tilda_X):\n", X_transpo_X)
# do alpha * Identi matrix 
alpha_I = alpha * I
print("I : \n", I)
print("alpha * I :\n", alpha_I)

fParameter = X_transpo_X + alpha_I
print("(tildaX^t * tildaX + alpha*I): \n", fParameter)
inverse_fParameter = np.linalg.inv(fParameter) 
print("Inver of above():\n", inverse_fParameter)

sendParameter  = np.dot(tilda_X_transpose, Y) # do dot product between matrix. 
print("(tildaX * Y):"); print("(tildaX)^t :\n", tilda_X_transpose , "\nY:\n", Y)
print("(tildaX^t * y):\n", sendParameter)
tilda_w = fParameter * sendParameter; 
print("tilda w : \n" , tilda_w); 
