from re import X
import pandas as pd
import numpy
import math
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def histograms(df_general):
    #make histogram
    plt.hist(df_general.scaled_longitude,label='longitude')
    plt.hist(df_general.scaled_latitude,label='latitude')
    plt.hist(df_general.scaled_housing_median_age,label='housing_median_age')
    plt.hist(df_general.scaled_total_rooms,label='total_rooms')
    plt.hist(df_general.scaled_total_bedrooms,label='total_bedrooms')
    plt.hist(df_general.scaled_population,label='population')
    plt.hist(df_general.scaled_households,label='households')
    plt.hist(df_general.scaled_median_income,label='median_income')
    plt.hist(df_general.LESS_THAN_1H_OCEAN,label='LESS_THAN_1H_OCEAN')
    plt.hist(df_general.INLAND,label='INLAND')
    plt.hist(df_general.ISLAND,label='ISLAND')
    plt.hist(df_general.NEAR_BAY,label='NEAR_BAY')
    plt.hist(df_general.NEAR_OCEAN,label='NEAR_OCEAN')
    plt.hist(df_general.scaled_median_house_value,label='median_house_value')
    plt.legend()
    plt.show() 
#

def data_diagrams(df_general):
    #make first graph
    x = df_general.scaled_longitude.tolist()
    y = df_general.scaled_latitude.tolist()
    colors = df_general.NEAR_OCEAN.tolist()
    plt.scatter(x,y,c=colors,alpha=0.5)
    plt.xlabel('scaled_longitude')
    plt.ylabel('scaled_latitude')
    plt.title('This graph (1/4)')
    plt.colorbar(label='NEAR_OCEAN')
    plt.show()

    #make second graph
    x = df_general.scaled_total_rooms.tolist()
    y = df_general.scaled_total_bedrooms.tolist()
    colors = df_general.LESS_THAN_1H_OCEAN.tolist()
    plt.scatter(x,y,c=colors,alpha=0.5)
    plt.xlabel('scaled_total_rooms')
    plt.ylabel('scaled_total_bedrooms')
    plt.title('This graph (2/4)')
    plt.colorbar(label='LESS_THAN_1H_OCEAN')
    plt.show()

    #make third graph
    x = df_general.scaled_households.tolist()
    y = df_general.scaled_median_income.tolist() 
    colors = df_general.INLAND.tolist()
    sizes = df_general.scaled_population.tolist()
    plt.scatter(x,y,c=colors,s= sizes,alpha=0.5)
    plt.xlabel('scaled_households')
    plt.ylabel('scaled_median_income')
    plt.colorbar(label='INLAND')
    plt.title('Graph (3/4) - Circle size according to scaled_population')
    plt.show()

    #make fourth graph
    x = df_general.scaled_median_house_value.tolist()
    y = df_general.ISLAND.values.tolist()
    colors = df_general.NEAR_BAY.tolist()
    sizes = df_general.scaled_housing_median_age.tolist()
    plt.scatter(x,y,c=colors,s= sizes,alpha=0.5)
    plt.xlabel('scaled_median_house_value')
    plt.ylabel('ISLAND')
    plt.colorbar(label='NEAR_BAY')
    plt.title('Graph (4/4) - Circle size according to scaled_housing_median_age')
    plt.show()
#
#


def perceptron_algorithm(vlist,dx):
    #add 1 on each data
    for x in vlist:
        x.append(1)
    
    
    #set wights + w0
    wt=[3,5,6,8,3,2,5,4,3,2,1,5,2,8]
    #perceptron algorithm
    #number of current loop
    t=0
    #learning rate
    rt=1
    #yt is the list with wrong classified data
    yt=[]
    
    while (len(yt)!=0 or t==0):
        
        yt.clear()
        c=0
        #checking
        for x in vlist:
            if dx[c]*(wt[0]*x[0]+wt[1]*x[1]+wt[2]*x[2]+wt[3]*x[3]+wt[4]*x[4]+wt[5]*x[5]+wt[6]*x[6]+wt[7]*x[7]+wt[8]*x[8]+wt[9]*x[9]+wt[10]*x[10]+wt[11]*x[11]+wt[12]*x[12]+wt[13]*x[13])>=0:
                
                yt.append([x,dx[c]])
            c+=1
        
       
        #define w(t+1)

        #mltiply dx*x
        for k,d in yt:
            for i in range(len(k)):
                k[i]=d*k[i]
    
        #multiply learning rate*k[i] (k[i] now is dx*x from the previous loop)
        for k,d in yt:
            for i in range(len(k)):
                k[i]=rt*k[i]
    
        #do wt-k[i] (k[i] now is learning rate*dx*x from previous loops)
        for k,z in yt:
            for i in range(len(wt)):
                wt[i]=wt[i]-k[i]
        
        
        
        t=t+1
        #in 1000th stop the algorithm
        if(t>=1000):
            #print(yt)
            break
    
    return wt
#
#

#
#
def LineAdd(a,arr1,b,arr2):
    arrCombine = []
    for i in range(len(arr1)):
        arrCombine.append(a*arr1[i]+b*arr2[i])
        #print(arr1[i],b,arr2[i],a*arr1[i]+b*arr2[i])
    return arrCombine
def LineMultiply(a,arr1,b,arr2):
    result = 0
    for i in range(len(arr1)):
        result+=a*arr1[i]*b*arr2[i]
    return result

def LeastSquares(X,Y):
    #X = {x1,x2,...}, xi = [...]
    #Y = {y1,y2,...}, yi in {+1,-1}
    #startingW = [0,0,...], len(w) = len(xi)
    w = [0,0,0,0,0,0,0,0,0,0,0,0,0]
    for i in range(len(Y)):
        w = LineAdd( 1 , w , (Y[i]-LineMultiply(1,w,1,X[i]))/(i+1) , X[i] ) 
    return w
#
#

def createMultiLayerNonLinear(X,Y):
    #X = {x1,x2,...}, xi = [...]
    #Y = {y1,y2,...}, yi = the value of the target variable corresponding to xi
    #creating the neuronic network
    model = Sequential()
    #each layer in the network uses the relu algorithm
    model.add(Dense(units=64, activation='relu'))#the first (hidden) layer is 64 nodes long
    model.add(Dense(units=32, activation='relu'))#the second (hidden) layer is 32 nodes long
    model.add(Dense(units=1, activation='relu')) #the last layer (the output) is 1 node long
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))
    model.fit(X, Y, epochs=40, batch_size=10, verbose=2)
    return model

def mae_mse(yi,yi_bar,mode):
    #find mean absolute error and mean square error
    sum_mae=0
    sum_mse=0
    #for each data from the test set
    if mode == 0:# for perseptron and least squares
        for i in range(2064):
            if yi[i]-yi_bar[i]==0:
                #if yi and yi_bar are the same then add zero
                sum_mae+=0
                sum_mse+=0
            else:
                #if yi and yi_bar are not the same then add one
                sum_mae+=1
                sum_mse+=1
    elif mode == 1:#for Multi Layer Non Linear Newral network
        for i in range(2064):
            sum_mae+=abs(yi[i]-yi_bar[i])
            sum_mse+=pow(yi[i]-yi_bar[i],2)
    #divide each sum with the number of test-set data
    mae=(1/2064)*sum_mae
    mse=(1/2064)*sum_mse
    #print the results
    print('Mean absolute error is : '+str(mae)+' and the mean square error is : '+str(mse))
#
#

def tenfoldcrossvalidation(ndarray,model,dx):
    #do the 10-fold cross validation
    kf = KFold(n_splits=10)
    kf.get_n_splits(ndarray)
    
    #set those lists - we need them for the 10 fold cross validation
    #result has 1 or -1 according to the model's decision for each data
    results=[]
    #yi is the original/calculated from the model cluster-value
    yi=[]
    #yi bar is the true cluster-value
    yi_bar=[]
    
    dx_train=[]
    
    #for each train bucket and test bucket
    for i, (train_index, test_index) in enumerate(kf.split(ndarray)):
        if i==0:
            print('-----------'+str(i+1)+'st-FOLD-----------')
        elif i==1:
            print('-----------'+str(i+1)+'nd-FOLD-----------')
        else:
            print('-----------'+str(i+1)+'th-FOLD-----------')
        #convert the train_index to list
        trainlist = (ndarray[train_index]).tolist()
        dx_train.clear()
        


        #find the true values from the train set
        for k in train_index:
            dx_train.append(dx[k])
        
        #call the classification model

        if model==1:
            wt = perceptron_algorithm(trainlist,dx_train)
        elif model==2:
            wt = LeastSquares(trainlist,dx_train)
        

        
        #print the classifier
        if len(wt)==14:
            print("The classifier is : g="+str(wt[0])+"*x1+"+str(wt[1])+"*x2+"+str(wt[2])+"*x3+"+str(wt[3])+"*x4+"+str(wt[4])+"*x5+"+str(wt[5])+"*x6+"+str(wt[6])+"*x7+"+str(wt[7])+"*x8+"+str(wt[8])+"*x9+"+str(wt[9])+"*x10+"+str(wt[10])+"*x11+"+str(wt[11])+"*x12+"+str(wt[12])+"*x13+"+str(wt[13]))
        else:
            print("The classifier is : g="+str(wt[0])+"*x1+"+str(wt[1])+"*x2+"+str(wt[2])+"*x3+"+str(wt[3])+"*x4+"+str(wt[4])+"*x5+"+str(wt[5])+"*x6+"+str(wt[6])+"*x7+"+str(wt[7])+"*x8+"+str(wt[8])+"*x9+"+str(wt[9])+"*x10+"+str(wt[10])+"*x11+"+str(wt[11])+"*x12+"+str(wt[12])+"*x13+")
        
        
        #for each feature from the current data find the result according to the classifier
        for [a,b,c,d,e,f,g,h,i,j,k,l,m] in ndarray[test_index]:
            #if the weights are 14, then the classification model is perceptron and we need to add the last weight which is w0
            if len(wt)==14:
                result = a*wt[0]+b*wt[1]+c*wt[2]+d*wt[3]+e*wt[4]+f*wt[5]+g*wt[6]+h*wt[7]+i*wt[8]+g*wt[9]+k*wt[10]+l*wt[11]+m*wt[12]+wt[13]
                results.append(result)
            else:
                result = a*wt[0]+b*wt[1]+c*wt[2]+d*wt[3]+e*wt[4]+f*wt[5]+g*wt[6]+h*wt[7]+i*wt[8]+g*wt[9]+k*wt[10]+l*wt[11]+m*wt[12]
                results.append(result)
        
        #we need yi for the new train-test set
        yi.clear()
        #decide 1 or -1 according to the classifier value from the current data
        for i in results:
            if i>0:
                yi.append(1)
    
            elif i<0:
                yi.append(-1)
    
        
    
        #we need yi bar for the new train-test set
        yi_bar.clear()
        #find the true values from the test set
        for k in test_index:
            yi_bar.append(dx[k])
    
        mae_mse(yi,yi_bar,0)
#
#

def tenfoldcrossvalidation_for_multilayer_neural_network(ndarray,scaled_median_house_value):
    #do the 10-fold cross validation
    kf = KFold(n_splits=10)
    kf.get_n_splits(ndarray)
    #for each train bucket and test bucket
    for i, (train_index, test_index) in enumerate(kf.split(ndarray)):
        if i==0:
            print('-----------'+str(i+1)+'st-FOLD-----------')
        elif i==1:
            print('-----------'+str(i+1)+'nd-FOLD-----------')
        else:
            print('-----------'+str(i+1)+'th-FOLD-----------')
        #convert the train_index to list
        trainlist = (ndarray[train_index]).tolist()
        scaled_mhv_list = []
        for [x] in scaled_median_house_value.tolist():
            scaled_mhv_list.append(x)

        scaled_mhv_list_for_train = []
        for k in train_index:
            scaled_mhv_list_for_train.append(scaled_mhv_list[k])

        scaled_mhv_list_for_test = []
        for k in test_index:
            scaled_mhv_list_for_test.append(scaled_mhv_list[k])
        


        output = createMultiLayerNonLinear(trainlist,scaled_mhv_list_for_train)
        testlist = (ndarray[test_index]).tolist()
        results = output.predict(testlist).tolist()

        result_list = []
        for [x] in results:
            result_list.append(x)

        mae_mse(scaled_mhv_list_for_test,result_list,1)
#
#




#read csv file
thisfile=pd.read_csv('your_source_file', delimiter=',')

#get one hot vector for ocean proximity
ocean_proximity = pd.get_dummies(thisfile,columns=['ocean_proximity'])

#remove from this dataset the non-ocean proximity columns
ocean_proximity = ocean_proximity.drop(['longitude','latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value'],axis=1)

#rename the ocean proximity column names
ocean_proximity.columns = ['LESS_THAN_1H_OCEAN', 'INLAND','ISLAND', 'NEAR_BAY', 'NEAR_OCEAN']

#set the part of dataframe without the ocean proximity
first_part=thisfile.drop(['median_house_value','ocean_proximity'],axis=1)

#find mean value from each column
file_mean= thisfile.mean(numeric_only= True) 

#for each column fill the null values with the column -  mean
first_part['longitude']=first_part['longitude'].fillna(file_mean.longitude)
first_part['latitude']=first_part['latitude'].fillna(file_mean.latitude)
first_part['housing_median_age']=first_part['housing_median_age'].fillna(file_mean.housing_median_age)
first_part['total_rooms']=first_part['total_rooms'].fillna(file_mean.total_rooms)
first_part['total_bedrooms']=first_part['total_bedrooms'].fillna(file_mean.total_bedrooms)
first_part['population']=first_part['population'].fillna(file_mean.population)
first_part['households']=first_part['households'].fillna(file_mean.households)
first_part['median_income']=first_part['median_income'].fillna(file_mean.median_income)





#scaling - scaledvalues is a ndarray with the scaled values from vlist
scale = StandardScaler()

#scale the firts part dataframe
nd_first_part_scaled = scale.fit_transform(first_part)


#scale the median house value - we need this for histograms and graphs
scaled_median_house_value = scale.fit_transform(thisfile[['median_house_value']])

#convert the ndarraylist from scaled first part to dataframe
df_first_part_scaled = pd.DataFrame(nd_first_part_scaled, columns = ['scaled_longitude','scaled_latitude', 'scaled_housing_median_age', 'scaled_total_rooms', 'scaled_total_bedrooms', 'scaled_population', 'scaled_households', 'scaled_median_income'])


#unite the first part with ocean proximity - the result is one dataframe
df = pd.concat([df_first_part_scaled, ocean_proximity], axis=1, join='inner')

#make a dataframe with scaled median house value
df_mhv_scaled = pd.DataFrame(scaled_median_house_value, columns = ['scaled_median_house_value'])

#make a dataframe with all data - we need them for histograms and diagrams
df_general = pd.concat([df, df_mhv_scaled], axis=1, join='inner')

histograms(df_general)



data_diagrams(df_general)

#initialize dx
dx=[]

#classify data according to mean value of median house values
for x in thisfile.median_house_value:
    if x>file_mean.median_house_value:
        dx.append(1)
    else:
        dx.append(-1)


#convert the df dataframe to ndarray
ndarray = df.to_numpy()

#convert the df dataframe to list
vlist=df.values.tolist()

wt_general = perceptron_algorithm(vlist,dx)
print("The general classifier for perceptron is : g="+str(wt_general[0])+"*x1+"+str(wt_general[1])+"*x2+"+str(wt_general[2])+"*x3+"+str(wt_general[3])+"*x4+"+str(wt_general[4])+"*x5+"+str(wt_general[5])+"*x6+"+str(wt_general[6])+"*x7+"+str(wt_general[7])+"*x8+"+str(wt_general[8])+"*x9+"+str(wt_general[9])+"*x10+"+str(wt_general[10])+"*x11+"+str(wt_general[11])+"*x12+"+str(wt_general[12])+"*x13+"+str(wt_general[13]))

wt_general = LeastSquares(vlist,dx)
print("The general classifier for least squares is : g="+str(wt_general[0])+"*x1+"+str(wt_general[1])+"*x2+"+str(wt_general[2])+"*x3+"+str(wt_general[3])+"*x4+"+str(wt_general[4])+"*x5+"+str(wt_general[5])+"*x6+"+str(wt_general[6])+"*x7+"+str(wt_general[7])+"*x8+"+str(wt_general[8])+"*x9+"+str(wt_general[9])+"*x10+"+str(wt_general[10])+"*x11+"+str(wt_general[11])+"*x12+"+str(wt_general[12])+"*x13")




print('------------------------------Going to 10-fold cross validation------------------------------')
print('-----------------------Going to pereceptron algorithm-----------------------')
tenfoldcrossvalidation(ndarray,1,dx)
print('-----------------------Going to least square errors algorithm-----------------------')
tenfoldcrossvalidation(ndarray,2,dx)
print('-----------------------Going to multilayer neural network-----------------------')
tenfoldcrossvalidation_for_multilayer_neural_network(ndarray,scaled_median_house_value)
