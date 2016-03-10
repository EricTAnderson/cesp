#!/usr/bin/env python3

# visualization.py 			Eric Anderson (3/16)
# A library of visualizations of cleaned input data
# (i.e., the wind vector, the sail positions, and boatspeed)
# Because of the high dimensional space, we need to be able to
# Cross-section the data efficiently. We also need to be able to
# superimpose datasets on each other to see how predictions do versus
# actual data

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image as I, ImageDraw as D
from math import cos, sin
import speedModel as sm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

#Returns an image of the sailboat pointed upwards with sails in the correct position
def boatImage(main,jib,stbd=False):
  boat = I.open('../transBoat.png')
  mainX = 41
  mainY = 80
  mainL = 60
  jibX = 41
  jibY = 10
  jibL = 40
  draw = D.Draw(boat)
  draw.line([mainX,mainY,mainX + mainL*sin(np.deg2rad(main)),mainY + mainL*cos(np.deg2rad(main))],fill=(0,0,0,255),width=3)
  draw.line([jibX,jibY,jibX + jibL*sin(np.deg2rad(jib)),jibY + jibL*cos(np.deg2rad(jib))],fill=(0,0,0,255),width=3)
  del(draw)

  if(stbd):
    return boat.transpose(0)
  return boat

#Returns polars of optimal sail position (just a practice function)
def optimalPolar(windSpeed):
  ax = plt.subplot(111, projection='polar')
  ax.set_theta_zero_location('N')
  ax.set_theta_direction(-1)
  theta = []
  r = []
  for windDir in range(0,181,5):
    (m,j,s) = sm.peekOptimal(windSpeed,windDir)
    theta.append(windDir)
    r.append(s)
    if(windDir % 20 == 0 and windDir > 35):
      arr = boatImage(m,j)
      arr = arr.rotate(-windDir,expand=True)
      im = OffsetImage(arr, zoom=0.4)
      ab = AnnotationBbox(im, (np.deg2rad(windDir),s+1), xycoords='data', frameon=False)
      ax.add_artist(ab)
      
      if(windDir < 180):
        #Plot the symmetrical boat, too
        arr = boatImage(m,j, True)
        arr = arr.rotate(windDir,expand=True)
        im = OffsetImage(arr, zoom=0.4)
        ab = AnnotationBbox(im, (np.deg2rad(-windDir),s+1), xycoords='data', frameon=False)
        ax.add_artist(ab)

  
  ax.plot(np.deg2rad(theta), r, color='b', linewidth=3,alpha=0.5)
  ax.plot(0-np.deg2rad(theta), r, color='b', linewidth=3,alpha=0.5)   #For symmetry
  ax.set_rmax(10.0)
  ax.grid(True)
  ax.set_title('Boatspeed with perfect sail position (Windspeed = ' + str(windSpeed) + ' knots)')
  plt.show()

