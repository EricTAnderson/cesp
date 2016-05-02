#!/usr/bin/env python3

from matplotlib import pyplot as plt
import visualization as v
import speedModel as sm
import planingModel as pm
import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RFR
import sys


def main(argv):
  np.random.seed(0)       #For repeatability
  fName = 'genTrain.csv'
  if len(argv) >=2:
    fName = argv[1]
  #Read in the training data
  print('Reading data from hardcoded file: ' + fName)
  with open(fName,'r') as f:
    x = pd.read_csv(f)#, nrows= 10000)

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
  
  f = 30
  print('Fitting Data with Random Forest (Forest size of : ' + str(f) + ')\n...\n')
  rf = RFR(n_estimators=f, verbose=2, oob_score=True).fit(x.loc[:,x.columns.difference(['boatSpeed'])],x.loc[:,'boatSpeed'])
  
  print('Predicting Training Data')
  yhat = rf.predict(x.loc[:,x.columns.difference(['boatSpeed'])])
  mse = ((yhat-x.loc[:,'boatSpeed'])**2).mean()
  print('Training Data MSE: ' + str(mse) + '\n')

  print('Predicting Validation Data')
  yhat = rf.predict(val.loc[:,val.columns.difference(['boatSpeed'])])
  mse = ((yhat-val.loc[:,'boatSpeed'])**2).mean()
  print('Validation Data MSE: ' + str(mse) + '\n')

  def forestController(windSpeed,windDir):
    
    #SailPos needs to be a single vector for the optimizer. Store in (main,jib) order
    def rfbs(sailPos,sign=1.0):
      a = pd.DataFrame()
      a['jib'] = sailPos[:,1]
      a['main'] = sailPos[:,0]
      a['windDir'] =  pd.Series([windDir for x in range(len(a.index))])
      a['windSpeed'] = pd.Series([windSpeed for x in range(len(a.index))])
      return sign*rf.predict(a)
    
    # Approximate optimization
    stride=2
    res = 90/stride+1
    sailPos=np.zeros((res**2,2))
    i = 0;
    for m in range (0,91,stride):
      for j in range(0,91,stride):
        sailPos[i,0]=m
        sailPos[i,1]=j
        i = i+1

    spds = rfbs(sailPos)
    i = np.argmax(spds)
    return(sailPos[i,0],sailPos[i,1])
    
    '''
    #Old 'optimizer', the above one (adapted from linear code) is better)
    query = pd.DataFrame(columns = x.columns.difference(['boatSpeed'])) #Empty dataframe with correct columns
    for m in range(19):       #Check every combination of angles (5 deg resolution)
      for j in range(19):
        query.loc[m*19+j,:] = {'windSpeed':windSpeed,'windDir':windDir,'main':m*5,'jib':j*5}
    pred = pd.DataFrame(rf.predict(query))
    ind = pred.idxmax()[0]
    # print((query.loc[ind,'main'],query.loc[ind,'jib']))
    return (query.loc[ind,'main'],query.loc[ind,'jib'])
    '''

  if(input('Compare to Optimal Data? [Y/n]:') == 'Y'):
    mse, perc = pm.coarseErrorvOpt(forestController)
    print("MSE versus optimal is " + str(mse))
    print("Percent Error versus optimal is " + str(perc))

  print("Visualizing control strategy")
  print("Model used: PLANING MODEL")          #CHANGE THIS EACH TIME YOU CHANGE HARDCODING AS SANITY CHECK!!!
  v.vizControlStrategy(forestController,model=pm,rawData=x)

if __name__=="__main__":
  main(sys.argv)