#!/usr/bin/env python3

from matplotlib import pyplot as plt
import visualization as v
import speedModel as sm
import numpy as np 
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor as KNN


def main():
  np.random.seed(0)       #For repeatability

  #Read in the training data
  print('Reading data from hardcoded file')
  with open('genTrain.csv','r') as f:
    x = pd.read_csv(f,nrows= 50000  )

  #Get a validation set
  print('Creating Validation and Training sets')
  vFrac = 0.2
  print('Splitting off a validation set of size ' + str(vFrac))
  val = x.sample(frac = vFrac)
  x = x[~x.index.isin(val.index)]    #Get the complement

  #Reset the indices
  x = x.reset_index(drop=True)
  val = val.reset_index(drop=True)

  print('X and Validation Shapes:')
  print(x.shape)
  print(val.shape)

  # n = 1000
  n = 300

  #Write my own fit, as I need to have access to the actual neighbors in question:
  #Weighted average of points, using weight like c*speed/d (with appropriate scaling) or speed/exp(d)
  #so we weight close points over fast points (that probably are just for higher windspeeds)
  #only really want to get avg for main,jib (don't care about bs, probably inaccurate anyways)
  #X is the fitted set, xPred is the set to predict, n_neighbors is the number of neighbors to use.
  #Note that this controller doesn't explicitly predict resultant speed from it's controls; rather, it simply
  #guesses at the right controls and hopes things work out.
  def myPred(x,xPred,n_neighbors,printMax=False):
    # print('in myPred(), x has null?: ' + str(x.isnull().any().any()))
    ds,inds = knn.kneighbors(xPred,n_neighbors=n_neighbors)
    print('Maximum distance point used is : ' + str(ds.max())) if printMax else None
    out=pd.DataFrame(columns=['mainOut','jibOut','boatSpeedOut'])
    for d,ind,i in zip(ds,inds,range(ds.shape[0])):   #This is still dealing with vectors of prediction points
      # print(i)
      # print(d)
      # print(x.shape)
      # print(ind)          #ind is still a vector of indices corresponding to the kth closest point to ith prediction point or sometihng like that
      # for i in ind:
      #   print(i in x.index.values)
      # print(x.loc[ind,'boatSpeed'].head())
      # print('\n')

      #My weight function, not complete yet
      w = 10**(x.loc[ind,'boatSpeed'])/(d+1)                            
      
      out.loc[i,'mainOut'] = ((x.loc[ind,'main'] * w).sum())/w.sum()
      out.loc[i,'jibOut'] = ((x.loc[ind,'jib'] * w).sum())/w.sum()
    
    out.index=xPred.index   #Make indices correspond
    return out
  
  #Wraps the KNN prediction in a nice API
  #Refs external vars x,n
  #Intended to handle only scalar inputs
  def knnController(windSpeed,windDir):
    ret = myPred(x,pd.DataFrame([[windSpeed,windDir]], columns=['windSpeed','windDir']),n)
    return (ret.iloc[0,0],ret.iloc[0,1])

  def myDist(x,y):
    #2 degress off course = 10 knots windspeed difference
    # print('x0: ' + str(x[0]))
    # print('x1: ' + str(x[1]))
    return (((x[0]-y[0])**2)/100 + ((x[1]-y[1])**2)/4)**0.5


  print('Fitting Data with KNN Regression (' + str(n) + ' neighbors)\n...\n')
  knn = KNN(n_neighbors=5,metric='pyfunc',func=myDist)

  # print(x.loc[:,['windSpeed','windDir']].shape)
  knn.fit(x.loc[:,['windSpeed','windDir']],x.loc[:,['main','jib','boatSpeed']])


  # print('Predicting Training Data')
  # out = myPred(x,x.loc[:,['windSpeed','windDir']],n) 

  #Calc MSE w.r.t optimal
  # errs = []
  # for w,sail in zip(x.index, out.index):
  #   mySpeed = sm.resultantSpeed(x.loc[w,'windSpeed'],x.loc[w,'windDir'],out.loc[sail,'mainOut'],out.loc[sail,'jibOut'])
  #   errs.append(mySpeed-sm.peekOptimal(x.loc[w,'windSpeed'],x.loc[w,'windDir'])[2])
  # mse= (np.array(errs)**2).mean()
  # print('Training Data MSE w.r.t optimal: ' + str(mse) + '\n')

  # if(input('Fit to Validation data? [Y/n]:') == 'Y'):
  # print('Fitting Validation Data')
  # out = myPred(x,val.loc[:,['windSpeed','windDir']],n,printMax=False)
  
  v.vizControlStrategy(knnController)

  if(input('Compare to Optimal Data? [Y/n]:') == 'Y'):
    mse, perc = sm.coarseErrorvOpt(knnController)
    print("MSE versus optimal is " + str(mse))
    print("Percent Error versus optimal is " + str(perc))




if __name__=="__main__":
  main()