#!/usr/bin/env python3

# visualization.py 			Eric Anderson (3/16)
# A library of visualizations of cleaned input data
# (i.e., the wind vector, the sail positions, and boatspeed)
# Because of the high dimensional space, we need to be able to
# Cross-section the data efficiently. We also need to be able to
# superimpose datasets on each other to see how predictions do versus
# actual data

import sys
from matplotlib.widgets import Slider
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from PIL import Image as I, ImageDraw as D
from math import cos, sin
import speedModel as sm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

#Returns an image of the sailboat pointed upwards with sails in the correct position
def boatImage(main,jib,stbd=False, newColor=(0,0,0,255)):
  boat = I.open('../transBoat.png')
  mainX = 57
  mainY = 80
  mainL = 60
  jibX = 57
  jibY = 10
  jibL = 40
  draw = D.Draw(boat)
  draw.line([mainX,mainY,mainX + mainL*sin(np.deg2rad(main)),mainY + mainL*cos(np.deg2rad(main))],fill=(0,0,0,255),width=3)
  draw.line([jibX,jibY,jibX + jibL*sin(np.deg2rad(jib)),jibY + jibL*cos(np.deg2rad(jib))],fill=(0,0,0,255),width=3)
  del(draw)

  pixdata = boat.load()
  for y in range(boat.size[1]):
    for x in range(boat.size[0]):
        if pixdata[x, y][3] == 255:#(0,0,0, 255):
            pixdata[x, y] = newColor

  # #Change color
  # boat = boat.convert('RGBA')
  
  # data = np.array(boat)   # "data" is a height x width x 4 numpy array
  # red, green, blue, alpha = data.T # Temporarily unpack the bands for readability
  
  # # Replace white with red... (leaves alpha values alone...)
  # black_areas = (red == 0) & (blue == 0) & (green == 0)
  # data[..., :-1][black_areas.T] = newColor # Transpose back needed
  
  # boat = Image.fromarray(data)
  # boat.show()

  if(stbd):
    return boat.transpose(0)
  return boat

#Returns polars of optimal sail position (just a practice function)
#Would be nice to add slider for wind speed...
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


#Returns polars of optimal sail position (just a practice function)
#Would be nice to add slider for wind speed...
def vizControlStrategy(controller=None,plotOpt=True):
  windSpeed = 10
  ax = plt.subplot(111, projection='polar')
  subplots_adjust(bottom=0.20)
  

  #RGBA values on 0 to 1 scale here
  def plotGivenWindSpeed(controller,windSpeed,color=(0,0,0,1)):
    if controller == None:
      return
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    theta = []
    r = []
    for windDir in range(0,181,5):
      m,j = controller(windSpeed,windDir)[:2]       #sm.peekOptimal(windSpeed,windDir)
      s=sm.resultantSpeed(windSpeed,windDir,m,j)    #This is the objective function always (while training on gen'ed data)
      theta.append(windDir)
      r.append(s)
      if(windDir % 20 == 0 and windDir > 35):
        arr = boatImage(m,j,newColor=tuple(255*x for x in color))      #Optional parameter newColor to change color
        arr = arr.rotate(-windDir,expand=True)
        im = OffsetImage(arr, zoom=0.4)
        ab = AnnotationBbox(im, (np.deg2rad(windDir),s+1), xycoords='data', frameon=False)
        ax.add_artist(ab)
        
        if(windDir < 180):
          #Plot the symmetrical boat, too
          arr = boatImage(m,j, True,newColor=tuple(255*x for x in color))
          arr = arr.rotate(windDir,expand=True)
          im = OffsetImage(arr, zoom=0.4)
          ab = AnnotationBbox(im, (np.deg2rad(-windDir),s+1), xycoords='data', frameon=False)
          ax.add_artist(ab)
  
      
    p1 = ax.plot(np.deg2rad(theta), r, color=color, linewidth=3,alpha=0.3)
    p2 = ax.plot(0-np.deg2rad(theta), r, color=color, linewidth=3,alpha=0.3)   #For symmetry
    ax.set_rmax(12)
    ax.grid(True)
    ax.set_title('Learned Control Strategy (Red) v Optimal (black)')

  axcolor = 'lightgoldenrodyellow'
  axSpd = axes([0.15, 0.1, 0.65, 0.03], axisbg=axcolor)
  sSpd = Slider(axSpd, 'Wind Speed', 0.01, 18.0, valinit=7)
  
  #Initial plot
  plotGivenWindSpeed(sm.peekOptimal,sSpd.val) if plotOpt else None  #Optimal control
  plotGivenWindSpeed(controller,sSpd.val,color=(1,0,0,1))           #Provided controller
  
  def update(val):
    ax.clear()
    plotGivenWindSpeed(sm.peekOptimal,sSpd.val) if plotOpt else None  #Optimal control
    plotGivenWindSpeed(controller,sSpd.val,color=(1,0,0,1))       #Provided controller
 
  sSpd.on_changed(update)

  plt.show()

