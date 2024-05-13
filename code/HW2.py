import math 
import pandas as pd 

# count frequency >> use this for counting Nn or Ny and use it to get weighted entropy 
def countFrequency(attr):  
    freq_dictonary = {}
    for val in attr: 
        freq_dictonary[val] = attr.count(val);
    return freq_dictonary
 
# calculate ex) I(3,2) then it is 3/5 * log2(3/5) + ... 
def getI(num1, num2): # num1 represent number of yes , num2 represent number of no 
    total = num1 + num2 
    if  total == 0 :
        return 0; 
    else:
        prob1 = num1/total 
        prob2 = num2/total
        # check probabilty before using math.log2() function 
        if prob1 == 0 or prob2 == 0: 
            return 0; 
        else: 
            entropy = -(prob1 * math.log2(prob1) + prob2 * math.log2(prob2))
            return entropy; 
        

# calculate entrophy H(D) or  H(Dy, Dn)
def get_entropy_H(attribute, y): # pass attribute ex) H(D|age) H(X1|y)
    totalN = len(y);  # should be 10 
    print("total N : ", totalN); 
    
    if totalN <= 0 : 
        print("total N is smaller than 0, check data again! "); 
        return 0; 
    else: 
        # check number of different selection 
       attribute = attribute.tolist(); # cover to lsit type to pass as parameter of countFrequency function 
       freq_dictionary =  countFrequency(attribute); 
       entropy = 0 
       numY = 0 # init. so in xj row check corresponding index in y to count number of0 
      
       for value, count in freq_dictionary.items(): 
           # if 2 appeared 3 times then key=2, value = 3 
           # for each xi #of 0 is yes, # 1 is no
           for index , attrbuteValue in enumerate(attribute): # iterate over xj col element 
               if(value == attrbuteValue) and (y[index] == 0): # xi check yi check whether yi is 1 or 0 
                   numY+=1;  # count number of yes (0) in y 
           
           numN = count - numY # number of no 
           eachValueEntropy  = (count /totalN)* getI(numY, numN)
           entropy +=  eachValueEntropy
  
       return entropy; 


# count number of zeros and ones in sub rigion. 
def countZeros_Ones(list,y) :  
    num_of_zero = 0
    num_of_ones = 0
   
    # covert numpy type y into dictionary 
    dict_y = {}
    for i in range(len(y)):
        dict_y[i] = y[i]
        
    for index in list:  # Iterate over the list
        if dict_y[index] == 0:  # Use the key to access the value in dict_y
            num_of_zero += 1
        elif dict_y[index] == 1:
            num_of_ones += 1
    return num_of_zero, num_of_ones     
        
            
# calculate Gain 
def IG(D, index, value):
    # Gain(D, Dy, Dx) = H(D) - H(Dy, Dn)
    # D: a dataset, tuple (X, y) where X is the data, y the classes
    # index: the index of the attribute (column of X) to split on
    # value: value of the attribute at index to split at  Xi ≤ value.
    X, y = D
    totalCount = len(y); 
    # entropyD = get_entropy_H(X[:,index], y)  # H(x1|y)  # calculate gentropy of specific colum attribute. 
    num1 = 0; num2 = 0;   # count # of 0's and # of 1's 
    for label in y:
        if label == 0: 
            num1+=1;  # error 
        else:
            num2+=1;

    H_D = getI(num1, num2)
    # get IG = H(D) - H(Dy, Dn) so H(D) - entropyD 
    
    #count number of entry that Xi <= value 
    # count number of entry in xi that is xi <=value ans let as passCount and totalcount - pasCOunt = filaCOunt 
    passCount = 0 # Xi <= V 
    failCount = 0 # Xi > v
    pass_list = []; 
    fail_list = []; 
    
    # iterate over each column in X and split. 
    for i, xi in enumerate(X[:, index]):  
        if xi <= value:
            passCount += 1
            pass_list.append(i) # append index 
        elif xi > value:
            failCount += 1
            fail_list.append(i) # append index 
    
    # get entropy after spliting baesd on value V 
    pass_zeros, pass_ones = countZeros_Ones(pass_list,y); 
    fail_zeros, fail_ones = countZeros_Ones(fail_list,y); 
    
    entropy_after_split = (passCount / totalCount ) * getI(pass_zeros, pass_ones) + (failCount / totalCount) * getI(fail_zeros, fail_ones) 
    return H_D - entropy_after_split; 


 # calculate Gini 
def calculate_Gini(frequency_dict, total_count): 
    p = 0; 
    for count in frequency_dict.values(): 
        proportion = count / total_count
        p +=(proportion)**2
    g_d = 1- p
    return g_d

# Gini index split  
# return Gini index in index with give value split 
def G(D, index, value):
    X, y = D
    total_count = len(y) # denominator n 
    y_list = y.tolist()  # convert y to list 
    frequency_dict =  countFrequency(y_list)
    
    # calcualte Gini G(D) before split. 
    g_d = calculate_Gini(frequency_dict, total_count); 

    # Iterate through the data points and count the occurrences of each class label
    # in the subsets resulting from the split
    passCount = 0 # Xi <= V 
    failCount = 0 # Xi > v
    listOfX_inpassCount = []; 
    listOfX_infailCount = []; 
    frequency_dict_pass = {}
    frequency_dict_fail = {}
    # iterate over each column in X. 
    for i , xi in enumerate(X[:, index]):  
        if xi <= value:
            passCount += 1
            listOfX_inpassCount.append(i) # append index positoin 
        elif xi > value:
            failCount += 1
            listOfX_infailCount.append(i)
            
    # get entropy after spliting baesd on value V 
    pass_zeros, pass_ones = countZeros_Ones(listOfX_inpassCount,y); 
    
    # prepare dictionary 
    for val in listOfX_inpassCount: 
        if val in frequency_dict_pass:
            frequency_dict_pass[val] +=1 
        else:
            frequency_dict_pass[val] = 1 
            
    for val2 in listOfX_infailCount: 
        if val2 in frequency_dict_fail:
            frequency_dict_fail[val2] +=1 
        else:
            frequency_dict_fail[val2] = 1 
            
    # get G(Dy) and G(Dn)
    G_Dy = calculate_Gini(frequency_dict_pass, total_count)
    G_Dn = calculate_Gini(frequency_dict_fail, total_count)
    
    # Calculate the proportion of data points in each subset
    Ny_N = (pass_zeros + pass_ones) / total_count  # Ny/ N 
    Nn_N = 1 - Ny_N # Nn / n 
    
    # Calculate the weighted average of Gini impurities after the split
    gini_after = Ny_N * (G_Dy) +  Nn_N * (G_Dn)
   
    # Calculate the Gini index for the split //
    # gini_index = g_d - gini_after
    gini_index = gini_after 
    return gini_index
    


def CART(D, index, value):
    # cart (Dy, Dn) = 2 * (ny/n) * (nn/n) simaga i =1 to 2 | p(ci|Dy) - p(ci|Dn)| 
    X, y = D 
    total_count = len(y) # n 
    # get given index colum in x and split based on value. 
    passCount = 0 # Xi <= V 
    failCount = 0 # Xi > v
    listOfX_inpassCount = []; 
    listOfX_infailCount = []; 

    # iterate over each column in X. 
    for xi in X[:, index]:
        if xi <= value:
            passCount += 1
            listOfX_inpassCount.append(xi) # add element 
        elif xi > value:
            failCount += 1
            listOfX_infailCount.append(xi)
    
    ny = passCount ; 
    nn = failCount; 

    pass_list = []
    fail_list =[]
 
    # for each class ci get P(ci|Dy) and  P(ci|Dn)  
    for i , xi in enumerate(X[:, index]):   
        if xi <= value : 
            pass_list.append(i); 
        else : 
            fail_list.append(i); 
            
    pass_zeros, pass_ones = countZeros_Ones(pass_list,y); 
    fail_zeros, fail_ones = countZeros_Ones(fail_list,y); 
    y_list = countFrequency(y.tolist())

    abs_sum_pro = 0; 
    for ci, freqency  in y_list.items():   # ci = 0, frequency : how many time 0 appear  so 4 
        p = 0; p_ci_Dy = 0 ; p_ci_Dn = 0; 
        if ci == 0 : 
            if ny == 0: 
                p_ci_Dy = 0 
            else: 
                p_ci_Dy = pass_zeros / ny ;    
            if nn == 0 : 
                p_ci_Dn = 0
            else: 
                 p_ci_Dn = fail_zeros / nn ; 
                
        elif ci == 1 : 
            if ny == 0: 
                p_ci_Dy = 0 
            else : 
                 p_ci_Dy = pass_ones / ny ; 
            if nn == 0 : 
                p_ci_Dn = 0
            else: 
                 p_ci_Dy = fail_ones / nn ; 
                 
        p = abs(p_ci_Dy - p_ci_Dn)
        abs_sum_pro += p; 
    cart_m = 2 * (ny/total_count) * (nn/total_count) * abs_sum_pro
    return cart_m 

def bestSplit(D, criterion):
    X, y = D 
    #  criterion: one of "IG", "GINI", "CART"
    # return tuple( index i , split value); 
    if criterion == "GINI": 
        score_star = float('inf')  #  for gini , smaller value is better to split 
    else: 
        score_star = float('-inf') # for gain and cart, higher value is better to split,
    # init best split tuple (i, value)
    best_splits_list = [] ;
    
    # iterate over all attribute in X 
    for i in range(X.shape[1]): # x1,x2 ,,, xn 
        split_points = []; 
        for value in  X[:, i]:  # if X0 colum has {0,0,0,0,1,1,2,2,0,0} then split points will have unique values so {0,1,2}
            if value not in split_points: 
                split_points.append(value)
            else:
                pass
        best_split = None 
        best_score = score_star
        
        for point in split_points: # iterate over possible split points. 
            if criterion == 'IG':
                score = IG(D, i, point)
            elif criterion == 'GINI':
                score =G(D, i, point)
            elif criterion == 'CART':
                score = CART(D, i, point)
               
               
            # if score > score_start (initialized one) update it and give best split tuple 
            if (criterion != 'GINI' and score > best_score) or (criterion == 'GINI' and score < best_score): 
                    best_score = score
                    best_split = (i, point) 
        best_splits_list.append(best_split)
    return best_split, best_splits_list
    


def load(filename):     # X : attributes, Y : classes 
	# create X and Y list  # check whether X should be row wise or colum wise?????
    df = pd.read_csv(filename, sep=",", header=None) 
    X = df.iloc[:, :-1].to_numpy()  # get each row except last col  [ row , col] 
    y = df.iloc[:, -1].to_numpy()  # get  last col as y classes 
    return X, y 
    
    
def classifyIG(train, test):
    # train  a tuple (X, y), 
    # test the test set, same format as train
    # return list of predicted classes for observations in test (in order)
	
   # call best split and best split list from bestsplit() function 
   X_train,y_train = train
   X_test, y_test = test
   
   best_split_value, _ = bestSplit(train, "IG"); # get best split value based on training data set using IG 
   single_index, split_value = best_split_value  # (9, 1)
   predicted_class_list = [] 
   errors = 0; # increment error count when predicted clasfier for test data is not same as true calss fof test data 
   
   for value in X_test[: , single_index]: # iterate over this single index colum elements from X_test data set 
       if value <= split_value:  # passed so classify as 0 
            predicted_class_list.append(0) 
       else:
            predicted_class_list.append(1)  # represent class 1 
    
    # iterate over y_test and predicted_class_list 
   for i in range(len(y_test)):
        if predicted_class_list[i] != y_test[i]:
            errors += 1;
   return predicted_class_list, errors 
  
 

def classifyG(train, test):
   X_train,y_train = train
   X_test, y_test = test
   
   best_split_value, _ = bestSplit(train, "GINI"); # get best split value based on training data set using IG 
   single_index, split_value = best_split_value  # (9, 1)
   predicted_class_list = [] 
   errors = 0; 
   
   for value in X_test[: , single_index]: # iterate over this single index colum elements from X_test data set 
       if value  <= split_value:  # passed so classify as 0 
            predicted_class_list.append(0) 
       else:
            predicted_class_list.append(1)  # represent class 1 
       # iterate over y_test and predicted_class_list 
   for i in range(len(y_test)):
        if predicted_class_list[i] != y_test[i]:
            errors += 1;
   return predicted_class_list, errors 


def classifyCART(train, test):
   X_train,y_train = train
   X_test, y_test = test
   
   best_split_value, _ = bestSplit(train, "CART"); # get best split value based on training data set using IG 
   single_index, split_value = best_split_value  # (9, 2)
   predicted_class_list = [] 
   errors = 0; 
   
   for value in X_test[: , single_index]: # iterate over this single index colum elements from X_test data set 
       if value  <= split_value:  # passed so classify as 0 
            predicted_class_list.append(0) 
       else:
            predicted_class_list.append(1)  # represent class 1 
            
      # iterate over y_test and predicted_class_list 
   for i in range(len(y_test)):
        if predicted_class_list[i] != y_test[i]:
            errors += 1;
   return predicted_class_list, errors 


def main():
	# load file 

    filename = "train.txt" 
    X, y = load(filename)
   
    # init 
    index = 0; # test x0 colum 
    value = 1; # split value for x <=value or x > value 
    
    # test information gain. 
    Gain_value = IG((X,y), index, value) 
    print("Information gain values : ", Gain_value)
    
    # get gain with index = 2 and value = 1 
    Gain_value2 = IG((X,y), 4, 1)
    print("Information gain value index =4 , value = 1 :", Gain_value2)
    # test Gini index 
    g = G((X,y), index ,value)
    print("Gini index: ", g); 
    
    # test cart 
    cart_measure = CART((X,y), 0, 1)
    print("Cart measure value at index=0, split value= 1: \n", cart_measure)
	
   # test part2 - d,e 
   # return a dataset D,
    best_split_value1, best_split_test_IG = bestSplit((X,y), "IG")
    print("Best split in IG : \n",  best_split_test_IG )
    print("IG : best split value : ", best_split_value1)
    
    best_split_value2,  best_split_test_GINI = bestSplit((X,y), "GINI")
    print("Best split in GINI : \n",  best_split_test_GINI)
    print("GINI : best split value : ", best_split_value2)
    
    best_split_value3, best_split_test_CART = bestSplit((X,y), "CART")
    print("Best split in CART : \n",  best_split_test_CART )
    print("CART : best split value : ", best_split_value3)
    
    # part2 - e, g 
    X_train, y_train = load("train.txt")
    X_test, y_test = load("test.txt")
    
    predicted_class_usingIG, errors_ig = classifyIG((X_train, y_train), (X_test, y_test))
    print("\npredicted  class using IG : " , predicted_class_usingIG)
    print("Number of classification error (where predicted class != actural class from test data): ", errors_ig)
    
    predicted_class_usingGINI , errors_gini = classifyG((X_train, y_train), (X_test, y_test))
    print("\npredicted  class using GINI : " , predicted_class_usingGINI)
    print("Number of classification error (where predicted class != actural class from test data): ", errors_gini)
  
    predicted_class_usingCART , errors_cart = classifyCART((X_train, y_train), (X_test, y_test))
    print("\npredicted  class using CART : " , predicted_class_usingCART)
    print("Number of classification error (where predicted class != actural class from test data): ", errors_cart)
 


if __name__=="__main__": 
	"""__name__=="__main__" when the python script is run directly, not when it 
	is imported. When this program is run from the command line (or an IDE), the 
	following will happen; if you <import HW2>, nothing happens unless you call
	a function.
	"""
	main()