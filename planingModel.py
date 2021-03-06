#!/usr/bin/env python

#planingModel.py 			Eric Anderson (4/16)

#Uses heuristics to create fake data to tune machine learning
#algorithm on.
#This generator extends speedModel to allow for drastic jumps in speed for near-optimal sail trim,
#corresponding roughly to a 'planing' mode that you can achieve.

import numpy as np

#Boat specific constants, set as globals
PointingAngle = 50    #The best angle we can point without huge speed loss


#A simple linear interpolation function
def lerp(val, currentMin, currentMax, interpMin, interpMax):
  if val < currentMin or val > currentMax:
    print("Val is out of bounds: val = " + str(val) + ", min = " + str(currentMin) + ", max = " + str(currentMax))
    return
  if currentMax <= currentMin:
    print("currentMin (" + str(currentMin) + ") >= currentMax (" + str(currentMax) + ")")
    return
  if interpMin == interpMax:
    print("interpMin equals intperpMax: " + str(interpMin))
  percentage = float((val-currentMin))/(currentMax-currentMin)
  return interpMin + percentage*(interpMax-interpMin)

#Returns max speed achievable given wind conditions
#Basically you can't sail faster than the wind, even at low speeds
#At high speeds, you only get a fraction of the windspeed
#This max speed does not take into account the planing boost, only resultant speed does that
def maxSpeed(windSpeed, windDir):
  windySpeedFactor = 0.5    #In very windy conditions, what proportion of the windspeed is beam reach?
                            #Note that if this is set too low you actually decrease in speed at high wind range!
  beatingFactor = 0.6       #How fast can you beat relative to beam reach speed?
  
  #Dependency on wind speed
  if windSpeed < 2:
    spd = windSpeed
  elif windSpeed > 15:
    spd = windySpeedFactor*windSpeed
  else:
    spd = windSpeed*lerp(windSpeed,2,15,1,windySpeedFactor) #You can sail windspeed in 2kt

  #Dependency on wind angle
  if windDir < PointingAngle-10:      #Can't pinch more than 10 degrees
    dirFactor = 0.0
  elif windDir < PointingAngle:       #Pinching
    dirFactor = lerp(windDir,PointingAngle-10,PointingAngle,0,beatingFactor)    #Speed drops off A LOT as you pinch
  elif windDir < 90:
    dirFactor = lerp(windDir,PointingAngle,90,beatingFactor,1)  #Slow more as you turn up to beat
  else:
    dirFactor = lerp(windDir,90,180,1,0.8)     #Get slightly slower as you turn away from the wind to run

  return spd*dirFactor

#Returns a factor (0 to 1) that scales the speed according to how well the sail is trimmed
#Sail Position is INPUT to this function
def sailPosFactor(pos, windSpeed,windDir, main=True):
  if pos < 0 or pos > 90:
    msg = 'Illegal main position: ' + str(pos) if main else 'Illegal jib position: ' + str(pos)
    print(msg)
    return
  if windDir < 0 or windDir > 180:
    print("Illegal windDir, must be in [0,180], was: " + str(windDir))
    return
  
  if pos > windDir:                 #Equivalent to backing the sail
    return 0
  else:                             #Code for optPos handles pinching just fine
    opt = optPos(windSpeed,windDir,main)
    diff = abs(pos-opt)
    if diff > 30:       #If you're really off, we'll say you barely move
      return 0.2
    else:
      return lerp(diff,0,30,1,0.2)  #If you're within 30 degrees of correct, it's just linear

#The secret answer to the model, i.e. the optimal posiiton I would like to get from ML
#Note that this would have you trim slightly tighter on beam reach than I would...
#But, my 'beam reach' is true wind, this beam reach is apparent? So might make sense?
#This is now dependent on windSpeed (changes if windSpeed>= 12)
def optPos(windSpeed,windDir,main=True):
  jibOffset = 5                     #At max trim, how much should jib be eased?
  if windDir < PointingAngle:       #If we're pinching, just pull in sails as much as we ever would
    if main:
      return 0
    else:
      return jibOffset

  extraEase = 0.0 
  if(windSpeed > 12):
    extraEase = lerp(windSpeed,12,18,0,7)   #Add at most 7 degrees of ease
  if(main):
    return min(lerp(windDir,PointingAngle,180,0,90) + extraEase,90)
  else:
    return min(lerp(windDir,PointingAngle,180,jibOffset,90) + extraEase,90) #Jib eased slightly more than main

#Planing code: if you are cracked off at all, and windspeed is > 8 kt, you can get a 1/8*windspeed boost for planing
#That is, 1 kt boost for eight knots go from there
def planingBoost(windSpeed,windDir,mainPos,jibPos):
  planingTol = 3.0
  #We have to be close to optimal
  optMain = optPos(windSpeed,windDir)
  optJib = optPos(windSpeed,windDir,main=False)

  #If you're not within 3 degrees, you get nothing
  if abs(optMain - mainPos) > 3.0 or abs(optJib - jibPos) > 3.0:
    return 0.0 

  #Good angle and windspeed?  If not, no planing
  if windDir >= PointingAngle + 10 and windSpeed >= 8:
    planeFactor = lerp(windDir,PointingAngle,90,0.02,0.125) if windDir < 90 else 0.125  #Upwind planing gets less benefit, beam reaching down gets a lot of benefit
    return windSpeed * planeFactor
  else:
    return 0.0

    
#The actual data creation, approximates speed based on the four inputs
def resultantSpeed(windSpeed,windDir,mainPos,jibPos):
  percentageMainDriven = 0.65           #How important the main is relative to the jib.  In this case, 65%
  speed = maxSpeed(windSpeed,windDir)
  sailC = percentageMainDriven*sailPosFactor(mainPos,windSpeed,windDir) + (1-percentageMainDriven)*sailPosFactor(jibPos,windSpeed,windDir,False)
  return speed * sailC + planingBoost(windSpeed,windDir,mainPos,jibPos)

#Returns the "optimal" sail position and assoc. speed according to the model
#You can use this to check how good your guess was
#Returns (main,jib,speed) tuples
def peekOptimal(windSpeed,windDir):
  optMain = optPos(windSpeed,windDir)
  optJib = optPos(windSpeed,windDir,False)
  return (optMain,optJib,resultantSpeed(windSpeed,windDir,optMain,optJib))


#Given a controller, returns the MSE versus optimal for a grid of sailing conditions
# And also the average percent error
def coarseErrorvOpt(controller):
  actualSpeed = []
  optSpeed = []
  for wSpd in range(1,18):
    for wDir in range(50,180):
      m,j = controller(wSpd,wDir)[:2]
      actualSpeed.append(resultantSpeed(wSpd,wDir,m,j))
      optSpeed.append(peekOptimal(wSpd,wDir)[2])
  actualSpeed = np.array(actualSpeed)
  optSpeed = np.array(optSpeed)
  mse = ((actualSpeed - optSpeed)**2).mean()
  percent = abs(np.array(actualSpeed - optSpeed)).mean()
  return (mse,percent)

#Assumptions:
#On beam reach, you can do windspeed in light wind
#In heavy wind, you can do 0.6 windspeed (lerp in between)
#Sails should be proportional from close haul to running
#Does heeling matter? Is it implicitly defined?
#The model only works for conditions up to 18 kt
def main():
  '''
  print('sailPosFactor test\n')
  for wDir in range(0,190,10):
    print('wDir = ' + str(wDir))
    for pos in range(0,100,10):
      print("pos = " + str(pos) + ", sailPosFactor = " + str(sailPosFactor(pos,wDir)))
    print("\n")
  '''

  '''
  print('MAX SPEED TEST')
  for spd in range(0,18,2):
    print("windspeed = " + str(spd))
    for angle in range(31,180,10):
      print("angle = " + str(angle) + ", MaxSpeed = " + str(maxSpeed(spd,angle)))
  '''

  '''
  print('Optimal Test')
  for spd in range(0,18,2):
    print("windspeed = " + str(spd))
    for angle in range(30,190,10):
      m,j,bs = peekOptimal(spd,angle)
      print("Wind Angle = " + str(angle))
      print("\tmain: " + str(m) + ", jib: " + str(j) + ", boatspeed: " + str(bs))
    print("\n")
  '''



if __name__ == "__main__":
  main()