#!/usr/bin/env python3


from matplotlib import pyplot as plt
import visualization as v
import speedModel as sm
import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression as LR
from preProcess import polyExpand

def main():

  np.random.seed(0)       #For repeatability

  #Read in the training data
  print('Reading data from hardcoded file')
  with open('genTrain.csv','r') as f:
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


  ############################
  #     RAW DATA VIZ         #
  ############################

  # v.vizRawData(x.loc[:20,:])
  # v.vizRawData(x.loc[:1500,:])
  # v.sliceRawData(x)

  print('Fitting Data with Expanded Linear Regression\n...\n')
  #Best expansion in 1 through 10 is 7
  expandFactor=7
  xExpand = polyExpand(x,expandFactor)
  valExpand = polyExpand(val,expandFactor)

  print("Sanity check, xExpand shape is: " + str(xExpand.shape))
  print("Sanity check, valExpand shape is: " + str(valExpand.shape))

  print("Fitting LR")
  lr = LR().fit(xExpand,x.loc[:,'boatSpeed'])

  print('Predicting Training Data')
  yhat = lr.predict(xExpand)
  mse = ((yhat-x.loc[:,'boatSpeed'])**2).mean()
  print('Training Data MSE: ' + str(mse))
  print('Training R2: ' + str(lr.score(xExpand,x.loc[:,'boatSpeed'])) + '\n')


  print('Predicting Validation Data')
  yhat = lr.predict(valExpand)
  mse = ((yhat-val.loc[:,'boatSpeed'])**2).mean()
  print('Validation Data MSE: ' + str(mse))
  print('Validation R2: ' + str(lr.score(valExpand,val.loc[:,'boatSpeed'])) + '\n')
  


  def lrController(wSpd,wDir):

    #SailPos needs to be a single vector for the optimizer. Store in (main,jib) order
    def lrbs(sailPos,sign=1.0):
      a = pd.DataFrame()
      a['main'] = sailPos[:,0]
      a['jib'] = sailPos[:,1]
      a['windSpeed'] = pd.Series([wSpd for x in range(len(a.index))])
      a['windDir'] =  pd.Series([wDir for x in range(len(a.index))])
      return sign*lr.predict(polyExpand(a,expandFactor))


    #Constrained optimization, args = -1.0 to actually get max
    # print("Beginning optimization")
    # optRes = Opt.minimize(lrbs,np.zeros(2),args=(-1.0,),bounds=[(0,90),(0,90)], method='SLSQP',
    #   options={'disp':True})
    # print("Message from Optimizer:")
    # print(optRes.message)
    # aStar=optRes.x

    # if not optRes.success:
    #   print("Optimization failed, exiting now")
    #   sys.exit(1)

    #Approximate optimization
    maxSpeed = 0.0
    maxM = 0
    maxJ = 0
    stride=2
    res = 90/stride+1
    sailPos=np.zeros((res**2,2))
    i = 0;
    for m in range (0,91,stride):
      for j in range(0,91,stride):
        sailPos[i,0]=m
        sailPos[i,1]=j
        i = i+1


    spds = lrbs(sailPos)
    i = np.argmax(spds)
    return(sailPos[i,0],sailPos[i,1])


  print("Comparing Controller to Optimal")
  # mse, perc = sm.coarseErrorvOpt(lrController)
  # print("MSE versus optimal is " + str(mse))
  # print("Percent Error versus optimal is " + str(perc))
  v.vizControlStrategy(lrController)



if __name__ == "__main__":
  main()