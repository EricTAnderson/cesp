#!/usr/bin/env python3

# exploreFit.py 				Eric Anderson (3/16)
# A basic fit of generated data, as a sanity check.  We should be able to
# fit pretty well to the generated data, as it is, for the most part, just
# generated by linear interpolation.  We'll separate the data into train,test
# and validation sets using our data splitter helper.

# This file uses several methods for fitting and is intended merely as a proof of concept.

from matplotlib import pyplot as plt
import visualization as v
import speedModel as sm
import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.ensemble import RandomForestRegressor as RFR
import sklearn.mixture as Mix
import scipy.optimize as Opt

#Returns an X data frame (*without* boatspeed) with polynomial expansion up to power Power.
#Useful preprocessor for linear regression
def polyExpand(data, power=1):
  ret=pd.DataFrame(data.loc[:,data.columns.difference(['boatSpeed'])])
  ret.columns=list(range(ret.shape[1]))       #Zero index the dataframe
  
  p = ret.shape[1]

  #Init conditions.  Lots of pointers here
  nextCol= ret.shape[1]   #Where to write to next
  offsets = [0,1,2,3]     #Offsets for each j value
  offset=0                #Each j iter we can calc offsets[i+1], so we need holder for (yet unused) old value
  eltsInPrevLayers = ret.shape[1]   #Total # elts in previous layers (pointer to start of THIS layer)
  nPrevPow = ret.shape[1]           #  # elts in last layer
  
  for trash in range(1,power):      #Do for a bunch of powers
    startPrevPow = eltsInPrevLayers- nPrevPow     #Pointer to start of PREVIOUS layer

    # print("  Power is: " + str(trash))
    # print("  eltsInPrevLayers: " + str(eltsInPrevLayers))
    # print("  nPrevPow: " + str(nPrevPow))
    # print("  offsets: " + str(offsets))
    # print("  startPrevPow: " + str(startPrevPow))

    for i in range(p):
      
      # print("\ti is " + str(i))
      # print("\t\toffset is " + str(offset))
      
      for j in range(offset,nPrevPow):
        # print("\t\tj is " + str(j))
        ret[nextCol] = ret.iloc[:,i]*ret.iloc[:,j+startPrevPow]
        nextCol = nextCol + 1
      if i != p-1:
        offset = offsets[i+1]
        offsets[i+1] = nextCol- eltsInPrevLayers
    offset=0
    nPrevPow = nextCol- eltsInPrevLayers
    eltsInPrevLayers=nextCol

  return ret

#Given a controller, returns the MSE versus optimal for a grid of sailing conditions
# And also the average percent error
def coarseErrorvOpt(controller):
  actualSpeed = []
  optSpeed = []
  for wSpd in range(1,18):
    for wDir in range(50,180):
      m,j = controller(wSpd,wDir)[:2]
      actualSpeed.append(sm.resultantSpeed(wSpd,wDir,m,j))
      optSpeed.append(sm.peekOptimal(wSpd,wDir)[2])
  actualSpeed = np.array(actualSpeed)
  optSpeed = np.array(optSpeed)
  mse = ((actualSpeed - optSpeed)**2).mean()
  percent = abs(np.array(actualSpeed - optSpeed)).mean()
  return (mse,percent)


def main():
  np.random.seed(0)       #For repeatability

  #Read in the training data
  fileName = 'planeTrain.csv'
  print('Reading data from hardcoded file: ' + fileName)
  with open(fileName,'r') as f:
    x = pd.read_csv(f)#, nrows= 1000)

  #Get a validation set
  print('Creating Validation and Training sets')
  vFrac = 0.2
  print('Splitting off a validation set of size ' + str(vFrac))
  val = x.sample(frac = vFrac)
  x = x[~x.index.isin(val.index)]    #Get the complement

  #Reset the indices
  x = x.reset_index(drop=True)
  val = val.reset_index(drop=True)

  #Break into x and y
  # y = x.loc[:,'boatSpeed']
  # yVal = val.loc[:,'boatSpeed']
  # x.drop(['boatSpeed'],axis=1,inplace=True); 
  # val.drop(['boatSpeed'],axis=1,inplace=True); 

  print('X and Validation Shapes:')
  print(x.shape)
  print(val.shape)


  ############################
  #     RAW DATA VIZ         #
  ############################

  # v.vizRawData(x.loc[:20,:])
  # v.vizRawData(x.loc[:5000,:])
  v.sliceRawData(x)
  return

  ############################
  #     Fitted Models        #
  ############################

  print('--------------------\n      FITS\n--------------------\n')

  ######################################
  #   LINEAR REGRESSION  CONTROLLER    #
  ######################################

  
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
    # print("My optimization finds:")
    # print("main: " + str(sailPos[i,0]))
    # print("jib: " + str(sailPos[i,1]))
    # print("speed: " + str(spds[i]))
    return(sailPos[i,0],sailPos[i,1])


  # print(lrController(10,60))
  print("Comparing Controller to Optimal")
  mse, perc = coarseErrorvOpt(lrController)
  print("MSE versus optimal is " + str(mse))
  print("Percent Error versus optimal is " + str(perc))
  v.vizControlStrategy(lrController)


  ############################
  #     KNN Controller       #
  ############################
  
  '''
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


  print('Predicting Training Data')
  out = myPred(x,x.loc[:,['windSpeed','windDir']],n) 

  #Calc MSE w.r.t optimal
  errs = []
  for w,sail in zip(x.index, out.index):
    mySpeed = sm.resultantSpeed(x.loc[w,'windSpeed'],x.loc[w,'windDir'],out.loc[sail,'mainOut'],out.loc[sail,'jibOut'])
    errs.append(mySpeed-sm.peekOptimal(x.loc[w,'windSpeed'],x.loc[w,'windDir'])[2])
  mse= (np.array(errs)**2).mean()
  print('Training Data MSE w.r.t optimal: ' + str(mse) + '\n')

  # if(input('Fit to Validation data? [Y/n]:') == 'Y'):
  print('Fitting Validation Data')
  out = myPred(x,val.loc[:,['windSpeed','windDir']],n,printMax=False)
  #Calc MSE w.r.t optimal
  errs = []
  for w,sail in zip(val.index, out.index):
    mySpeed = sm.resultantSpeed(val.loc[w,'windSpeed'],val.loc[w,'windDir'],out.loc[sail,'mainOut'],out.loc[sail,'jibOut'])
    errs.append(mySpeed-sm.peekOptimal(val.loc[w,'windSpeed'],val.loc[w,'windDir'])[2])   #NOT that same as observed bs at that x point b/c noise!
  mse= (np.array(errs)**2).mean()
  print('Validation Data MSE w.r.t optimal: ' + str(mse) + '\n')

  v.vizControlStrategy(knnController)
  '''

  ############################
  # RANDOM FOREST CONTROLLER #
  ############################
  '''
  f = 20
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

  def forestControl(windSpeed,windDir):
    query = pd.DataFrame(columns = x.columns.difference(['boatSpeed'])) #Empty dataframe with correct columns
    for m in range(19):       #Check every combination of angles (5 deg resolution)
      for j in range(19):
        query.loc[m*19+j,:] = {'windSpeed':windSpeed,'windDir':windDir,'main':m*5,'jib':j*5}
    pred = pd.DataFrame(rf.predict(query))
    ind = pred.idxmax()[0]
    # print((query.loc[ind,'main'],query.loc[ind,'jib']))
    return (query.loc[ind,'main'],query.loc[ind,'jib'])

  v.vizControlStrategy(forestControl)
  '''
  

  ##########   LESS USEFUL PRELIMINARY TESTS ################

  ############################
  #         KNN              #
  ############################

  '''  
  n = 300
  print('Fitting Data with KNN Regression (' + str(n) + ' neighbors)\n...\n')
  knn = KNN(n_neighbors=n).fit(x.loc[:,x.columns.difference(['boatSpeed'])],x.loc[:,'boatSpeed'])

  print('Fitting Training Data')
  yhat = knn.predict(x.loc[:,x.columns.difference(['boatSpeed'])])
  mse = ((yhat-x.loc[:,'boatSpeed'])**2).mean()
  print('Training Data MSE: ' + str(mse) + '\n')

  print('Fitting Validation Data')
  yhat = knn.predict(val.loc[:,val.columns.difference(['boatSpeed'])])
  mse = ((yhat-val.loc[:,'boatSpeed'])**2).mean()
  print('Validation Data MSE: ' + str(mse) + '\n')
  '''

  ############################
  #      RANDOM FOREST       #
  ############################

  '''
  f = 30
  print('Fitting Data with Random Forest (Forest size of : ' + str(f) + ')\n...\n')
  rf = RFR(n_estimators=f, verbose=2, oob_score=True).fit(x,y)

  print('Fitting Training Data')
  yhat = rf.predict(x)
  mse = ((yhat-y)**2).mean()
  print('Training Data MSE: ' + str(mse) + '\n')

  print('Fitting Validation Data')
  yhat = rf.predict(val)
  mse = ((yhat-yVal)**2).mean()
  print('Validation Data MSE: ' + str(mse) + '\n')
  '''

  ############################
  #   LINEAR REGRESSION      #
  ############################

  '''
  print('Fitting Data with Linear Regression\n...\n')  
  lr = LR().fit(x.loc[:,x.columns.difference(['boatSpeed'])],x.loc[:,'boatSpeed'])

  print('Fitting Training Data')
  yhat = lr.predict(x.loc[:,x.columns.difference(['boatSpeed'])])
  mse = ((yhat-x.loc[:,'boatSpeed'])**2).mean()
  print('Training Data MSE: ' + str(mse) + '\n')

  print('Fitting Validation Data')
  yhat = lr.predict(val.loc[:,val.columns.difference(['boatSpeed'])])
  mse = ((yhat-val.loc[:,'boatSpeed'])**2).mean()
  print('Validation Data MSE: ' + str(mse) + '\n')
  '''

  ############################
  #   Ridge REGRESSION       #
  ############################

  '''
  print('Fitting Data with Expanded Ridge Regression\n...\n')

  expandFactor=7
  xExpand = polyExpand(x,expandFactor)
  valExpand = polyExpand(val,expandFactor)

  # print("Sanity check, xExpand shape is: " + str(xExpand.shape))
  # print("Sanity check, valExpand shape is: " + str(valExpand.shape))

  print("Fitting RR")
  #lams = np.logspace(-2,2, 100)
  lams = np.arange(0.1,10,0.1)
  rr = RidgeCV(alphas=lams, store_cv_values=True).fit(xExpand,x.loc[:,'boatSpeed'])
  print('Info on Cross Validation:')
  print('Average error per lambda')
  av = np.mean(rr.cv_values_,axis = 0)
  print(av)

  print('selected: ' + str(rr.alpha_))
  plt.scatter(lams,av)
  #plt.gca().set_xscale('log')
  plt.show()

  print('Predicting Training Data')
  yhat = rr.predict(xExpand)
  mse = ((yhat-x.loc[:,'boatSpeed'])**2).mean()
  print('Training Data MSE: ' + str(mse))
  print('Training R2: ' + str(rr.score(xExpand,x.loc[:,'boatSpeed'])) + '\n')

  print('Predicting Validation Data')
  yhat = rr.predict(valExpand)
  mse = ((yhat-val.loc[:,'boatSpeed'])**2).mean()
  print('Validation Data MSE: ' + str(mse))
  print('Validation R2: ' + str(rr.score(valExpand,val.loc[:,'boatSpeed'])) + '\n')
  




  def rrController(wSpd,wDir):

    #SailPos needs to be a single tuple for the optimizer. Store in (main,jib) order
    def rrbs(sailPos,sign=1.0):
      a = pd.DataFrame()
      a['main'] = sailPos[:,0]
      a['jib'] = sailPos[:,1]
      a['windSpeed'] = pd.Series([wSpd for x in range(len(a.index))])
      a['windDir'] =  pd.Series([wDir for x in range(len(a.index))])
      return sign*rr.predict(polyExpand(a,expandFactor))

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


    spds = rrbs(sailPos)
    i = np.argmax(spds)
    # print("My optimization finds:")
    # print("main: " + str(sailPos[i,0]))
    # print("jib: " + str(sailPos[i,1]))
    # print("speed: " + str(spds[i]))
    return(sailPos[i,0],sailPos[i,1])


  # print(lrController(10,60))
  
  v.vizControlStrategy(rrController)
  '''

  '''
  print('Fitting Data with Expanded Ridge Regression\n...\n')

  xExpand = polyExpand(x,3)
  valExpand = polyExpand(val,3)

  print('Fitting ')
  
  #Shouldn't pass alpha = 0 -  leads to weird nan values in scores
  #lams = np.logspace(-2,2, 100)
  lams = np.arange(10,30,0.1)
  rr = RidgeCV(alphas=lams, store_cv_values=True).fit(xExpand,x.loc[:,'boatSpeed'])

  # print('Info on Cross Validation:')
  # print('Average error per lambda')
  
  av = np.mean(rr.cv_values_,axis = 0)
  print(av)

  print('selected: ' + str(rr.alpha_))
  plt.scatter(lams,av)
  #plt.gca().set_xscale('log')
  plt.show()


  print('Predicting Training Data')
  yhat = rr.predict(xExpand)
  mse = ((yhat-x.loc[:,'boatSpeed'])**2).mean()
  print('Training Data MSE: ' + str(mse) + '\n')

  print('Predicting Validation Data')
  yhat = rr.predict(valExpand)
  mse = ((yhat-val.loc[:,'boatSpeed'])**2).mean()
  print('Validation Data MSE: ' + str(mse) + '\n')
  '''

  ############################
  #        DPGMM             #
  ############################

  '''
  n = 10
  print('Fitting Data with DPGMM (' +str(n) + ' components maximum)\n...\n')
  dpgmm = Mix.DPGMM(n_components=n, covariance_type='diag',verbose=0).fit(x,y )  #Can change verbosity but it's VERY verbose

  print('Fitting Training Data')
  yhat = dpgmm.predict(x)
  print(yhat[:10])
  ##FIXME: output from predict is just what gaussian it belongs to...

  print('Fitting Validation Data')
  yhat = dpgmm.predict(val)
  #mse = ((yhat-yVal)**2).mean()
  #print('Validation Data MSE: ' + str(mse) + '\n')
  '''


if __name__ == "__main__":
  main()