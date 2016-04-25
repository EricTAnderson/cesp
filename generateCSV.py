#!/usr/bin/env python

#generateCSV.py
#uses speedModel.py to create sample data written to a csv file
#Noise is optionally added to the data
#Output format is (wind speed (0-18), relative Wind Direction(0-180), main position (0-90), jib position(0-90), boat speed)

import sys
import csv
import getopt
import random as r
import speedModel as sm
import planingModel as pm
import numpy.random as npr

#Parse input
def parseArgs(argv):
  usage = 'usage: ./generateCSV -l <file length> -s <seed> -f <fileName> --noise <noise> --model <model name>'
  if len(argv) == 0:
    print('Not enough args')
    print(usage)
    sys.exit(2)
  length = 0
  noise = 0.0
  seed = 0
  fname = "gen.csv"
  speedFunc = ("default",sm.resultantSpeed)
  try:
    opts, args = getopt.getopt(argv,"hf:l:s:m:",["noise=", "file=","model="])
  except getopt.GetoptError as e:
    print("Parse Error")
    print(e.msg)
    print(e.opt)
    print(usage)
    sys.exit(2)

  for opt, arg in opts:
    if opt == '-h':
      print(usage)
      sys.exit()
    elif opt in ("-l"):
      length = int(arg)
    elif opt in ("-s"):
      seed = int(arg)
    elif opt in ("--noise"):
      noise = float(arg)
    elif opt in ("--file", "-f"):
      fname = arg
    elif opt in ("--model","-m"):
      if arg == "planing":
        speedFunc = ("planing", pm.resultantSpeed)

  return (length,noise,seed,fname,speedFunc)


def main(argv):
  length,noise,seed,fname,speedFunc = parseArgs(argv)

  print('Using filename: ' + fname)
  print('Using length: ' + str(length))
  print('Using seed: ' + str(seed))
  print("Using Model: " + speedFunc[0])
  print('Adding Noise: ' + (str(noise) if noise > 0.0001 else "False"))

  cont = input("Continue? [Y/n]:")
  if cont != 'Y':
    print('Exiting...')
    sys.exit(1)

  #We could be smarter and do this as vector operations,
  #but we're not worried about performance here
  r.seed(seed)
  npr.seed(seed)
  with open(fname,'w') as f:
    wr = csv.writer(f)
    wr.writerow(['windSpeed', 'windDir', 'main', 'jib', 'boatSpeed']) #Nice headers
    
    #In the planning model, we are specifically adding some near-optimal points so that we can clearly see planing features
    if speedFunc[0] == 'planing':
      ptsToSteal = int(length*0.1)
      print('In planing generation, stealing ' + str(ptsToSteal) + ' pts')
      length = length - ptsToSteal
      for i in range(ptsToSteal):
        wDir = r.random() * 180  #Wind in range 0 to 180
        wSpd = r.random() * 18   #No training data above 18 knots
        opt = pm.peekOptimal(wSpd,wDir)
        msign = 1.0 if r.random() >= 0.5 else -1.0
        jsign = 1.0 if r.random() >= 0.5 else -1.0
        m = opt[0] + msign*r.random() * 5.0      #Main and jib both within 10 deg of optimal
        m = min(max(m,0),90)
        j = opt[1] + jsign*r.random() * 5.0
        j = min(max(j,0),90)
        bs = speedFunc[1](wSpd,wDir,m,j)

        #Might want to obscure the speed output with noise
        #We'll just add a gaussian
        if noise > 0.0001:
          bs = npr.normal(bs,noise)

        wr.writerow([wSpd,wDir,m,j,bs])




    for i in range(length):
      wDir = r.random() * 180  #Wind in range 0 to 180
      wSpd = r.random() * 18   #No training data above 18 knots
      m = r.random() * 90      #Main and jib both from 0 to 90
      j = r.random() * 90
      bs = speedFunc[1](wSpd,wDir,m,j)

      #Might want to obscure the speed output with noise
      #We'll just add a gaussian
      if noise > 0.0001:
        bs = npr.normal(bs,noise)

      wr.writerow([wSpd,wDir,m,j,bs])



if __name__ == "__main__":
  main(sys.argv[1:])