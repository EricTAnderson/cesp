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
import numpy.random as npr

#Parse input
def parseArgs(argv):
  usage = 'usage: ./generateCSV -l <file length> -s <seed> -f <fileName> --noise <noise>'
  if len(argv) == 0:
    print('Not enough args')
    print(usage)
    sys.exit(2)
  length = 0
  noise = 0.0
  seed = 0
  fname = "gen.csv"
  try:
    opts, args = getopt.getopt(argv,"hf:l:s:",["noise=", "file="])
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
  return (length,noise,seed,fname)


def main(argv):
  length,noise,seed,fname = parseArgs(argv)

  print('Using filename: ' + fname)
  print('Using length: ' + str(length))
  print('Using seed: ' + str(seed))
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
    for i in range(length):
      wDir = r.random() * 180  #Wind in range 0 to 180
      wSpd = r.random() * 18   #No training data above 18 knots
      m = r.random() * 90      #Main and jib both from 0 to 90
      j = r.random() * 90
      bs = sm.resultantSpeed(wSpd,wDir,m,j)

      #Might want to obscure the speed output with noise
      #We'll just add a gaussian
      if noise > 0.0001:
        bs = npr.normal(bs,noise)

      wr.writerow([wSpd,wDir,m,j,bs])



if __name__ == "__main__":
  main(sys.argv[1:])