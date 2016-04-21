import numpy as np 
import pandas as pd

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