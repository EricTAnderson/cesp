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
  draw.line([mainX,mainY,mainX + mainL*sin(np.deg2rad(main)),mainY + mainL*cos(np.deg2rad(main))],fill=255,width=3)
  draw.line([jibX,jibY,jibX + jibL*sin(np.deg2rad(jib)),jibY + jibL*cos(np.deg2rad(jib))],fill=255,width=3)
  del(draw)
  if(stbd):
    return boat.transpose(0)  #Flip horizontally
  return boat

