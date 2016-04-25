#!/usr/bin/env python3

from matplotlib import pyplot as plt
import visualization as v
import speedModel as sm
import planingModel as pm
import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RFR


def main():
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

  # print('Predicting Training Data')
  # yhat = rf.predict(x.loc[:,x.columns.difference(['boatSpeed'])])
  # mse = ((yhat-x.loc[:,'boatSpeed'])**2).mean()
  # print('Training Data MSE: ' + str(mse) + '\n')

  # print('Predicting Validation Data')
  # yhat = rf.predict(val.loc[:,val.columns.difference(['boatSpeed'])])
  # mse = ((yhat-val.loc[:,'boatSpeed'])**2).mean()
  # print('Validation Data MSE: ' + str(mse) + '\n')

  def forestController(windSpeed,windDir):
    query = pd.DataFrame(columns = x.columns.difference(['boatSpeed'])) #Empty dataframe with correct columns
    for m in range(19):       #Check every combination of angles (5 deg resolution)
      for j in range(19):
        query.loc[m*19+j,:] = {'windSpeed':windSpeed,'windDir':windDir,'main':m*5,'jib':j*5}
    pred = pd.DataFrame(rf.predict(query))
    ind = pred.idxmax()[0]
    # print((query.loc[ind,'main'],query.loc[ind,'jib']))
    return (query.loc[ind,'main'],query.loc[ind,'jib'])

  if(input('Compare to Optimal Data? [Y/n]:') == 'Y'):
    mse, perc = sm.coarseErrorvOpt(forestController)
    print("MSE versus optimal is " + str(mse))
    print("Percent Error versus optimal is " + str(perc))

  print("Visualizing control strategy")
  print("Model used: PLANING MODEL")
  v.vizControlStrategy(forestController,model=pm,rawData=x)

if __name__=="__main__":
  main()