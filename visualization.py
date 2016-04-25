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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
from matplotlib.text import TextPath
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle, PathPatch 
from matplotlib.transforms import Affine2D

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


#Plots sailing polars and sailboat icons with representative sail positions for the optimal controller
#(as defined in speedModel) if plotOpt is true and for a (presumably) learned controller function
def vizControlStrategy(controller=None,plotOpt=True,model=sm, rawData = pd.DataFrame()):
  windSpeed = 10
  ax = plt.subplot(111, projection='polar')
  subplots_adjust(bottom=0.20)
  

  #RGBA values on 0 to 1 scale here
  #Adding MSE analysis here, although it's too bad that it has to be in a visualization file
  def plotGivenWindSpeed(controller,windSpeed,color=(0,0,0,1),plotData=False):
    if controller == None:
      return
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    #Maybe add scatter plot in background?
    if plotData and not rawData.empty :
      l = rawData[rawData['windSpeed'] > windSpeed - 0.5]
      tot = l[l['windSpeed'] < windSpeed + 0.5]
      scat1 = ax.scatter(np.deg2rad(tot.loc[:,'windDir']),tot.loc[:,'boatSpeed'],c = tot.loc[:,'boatSpeed'],picker=5, cmap = plt.cm.get_cmap('YlOrRd'),zorder=-1, alpha=0.5)
      scat2 = ax.scatter(np.deg2rad(360-tot.loc[:,'windDir']),tot.loc[:,'boatSpeed'],c=tot.loc[:,'boatSpeed'], picker=5, cmap = plt.cm.get_cmap('YlOrRd'),zorder=-1,alpha=0.5)


    theta = []
    r = []
    optSpeed = []
    for windDir in range(0,181,5):
      m,j = controller(windSpeed,windDir)[:2]       
      s=model.resultantSpeed(windSpeed,windDir,m,j)    #This is the objective function always (while training on gen'ed data)
      theta.append(windDir)
      r.append(s)
      optSpeed.append(model.peekOptimal(windSpeed,windDir)[2])
      #Plot boats
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
  
    mseOpt = ((np.array(r) - np.array(optSpeed))**2).mean()
    p1 = ax.plot(np.deg2rad(theta), r, color=color, linewidth=3,alpha=0.6,zorder=1)
    p2 = ax.plot(0-np.deg2rad(theta), r, color=color, linewidth=3,alpha=0.6,zorder=1)   #For symmetry
    ax.set_rmax(12)
    ax.grid(True)
    ax.set_title('Learned Control Strategy (Red) v Optimal (black)\nMSE from Opt: ' + str(mseOpt))

    

  axcolor = 'lightgoldenrodyellow'
  axSpd = axes([0.15, 0.1, 0.65, 0.03], axisbg=axcolor)
  sSpd = Slider(axSpd, 'Wind Speed', 0.01, 18.0, valinit=7)
  
  #Initial plot
  plotGivenWindSpeed(model.peekOptimal,sSpd.val) if plotOpt else None  #Optimal control
  plotGivenWindSpeed(controller,sSpd.val,color=(1,0,0,1),plotData = True)           #Provided controller
  
  def update(val):
    ax.clear()
    plotGivenWindSpeed(model.peekOptimal,sSpd.val) if plotOpt else None  #Optimal control
    plotGivenWindSpeed(controller,sSpd.val,color=(1,0,0,1),plotData=True)       #Provided controller
 
  sSpd.on_changed(update)

  plt.show()

#A helper function to write text in 3d environment, taken from matplotlib online examples
def text3d(ax, xyz, s, zdir="z", size=None, angle=0, **kwargs):

    x, y, z = xyz
    if zdir == "y":
        xy1, z1 = (x, z), y
    elif zdir == "y":
        xy1, z1 = (y, z), x
    else:
        xy1, z1 = (x, y), z

    text_path = TextPath((0, 0), s, size=size)
    trans = Affine2D().rotate(angle).translate(xy1[0], xy1[1])

    p1 = PathPatch(trans.transform_path(text_path), **kwargs)
    ax.add_patch(p1)
    art3d.pathpatch_2d_to_3d(p1, z=z1, zdir=zdir)


#Take in a DataFrame of raw data and visualize it
def vizRawData(data):
  beatingAngle = 45 #Where do we draw the no go lines?

  fig = plt.figure()
  fig.set_size_inches(18, 12,forward=True)          #Manually resize
  gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) #Breaks up the figure according to special ratio
  ax = fig.add_subplot(gs[0], projection='3d')      #The main figure
  im = fig.add_subplot(gs[1])                       #The point inspector pane
  im.axis('equal')
  im.axis([0,1,0,1])
  plt.tight_layout(pad=4.08, h_pad=None, w_pad=5, rect=None)  #Layout the plots nicely
  
  #Formatting
  ax.set_title('Boatspeed as a function of Wind Angle (Raw Data)')
  ax.set_zlabel('Boatspeed (kts)')
  ax.set_xlabel('Wind Vector')
  ax.set_ylabel('Wind Vector')

  ax.set_xlim3d(-20,20)
  ax.set_ylim3d(-20,20)
  ax.set_zlim3d(0,12)
  ax.axes.get_xaxis().set_ticks([])
  ax.axes.get_yaxis().set_ticks([])
  im.axes.get_xaxis().set_ticks([])
  im.axes.get_yaxis().set_ticks([])


  #Plot my own axes
  ax.plot([0,0],[0,0],[0,15],'--k') #Vertical line
  ax.plot([0,0],[0,20],[0,0],'--k',linewidth=3) #Horizontal line for Irons
  ax.plot([0,20*np.sin(np.deg2rad(beatingAngle))],[0,20*np.cos(np.deg2rad(beatingAngle))],[0,0],color=(0.5,0.5,0.5),linestyle='--',linewidth=3) #Horizontal line for beating angles
  ax.plot([0,-20*np.sin(np.deg2rad(beatingAngle))],[0,20*np.cos(np.deg2rad(beatingAngle))],[0,0],color=(0.5,0.5,0.5),linestyle='--',linewidth=3) #Horizontal line for beating angles
  
  # ax.plot([0,-20],[0,20],[0,0],color=(0.5,0.5,0.5),linestyle='--',linewidth=3) #Horizontal line for beating angles

  #text3d angle is off by 90 from what I'm using
  text3d(ax, (0.5,-0.5, 0),
       " - no go zone - ",
       zdir="z", size=3,
       ec="none", fc=(0.3,0.3,0.3),angle=np.deg2rad(90-beatingAngle))
  text3d(ax, (0.5,0.5, 0),
       "   no go zone  ",
       zdir="z", size=3,
       ec="none", fc=(0.3,0.3,0.3),angle=np.deg2rad(90+beatingAngle))
  text3d(ax, (0.5,3, 0),
     " Wind Dir",
     zdir="z", size=2,
     ec="none", fc='k',angle=np.deg2rad(90))

  for r in range(20):
    p = Circle((0,0), r,fill=False,edgecolor=(0.7,0.7,0.7),linestyle=('-' if r%4 == 0 else '--'))
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")

  im.text(0.5, 0.75, 'Click on a point\n for more info',ha='center',fontsize=15,bbox={'facecolor':'blue', 'alpha':0.5, 'pad':10})

  #Map data from polar
  x = data.loc[:,'windSpeed']*np.sin(np.deg2rad(data.loc[:,'windDir']))  #So points along +y axis are head to wind
  y = data.loc[:,'windSpeed']*np.cos(np.deg2rad(data.loc[:,'windDir']))
  z = data.loc[:,'boatSpeed']
  scat1 = ax.scatter(x,y,z, c=z, marker='o',picker=5, cmap = plt.cm.get_cmap('YlOrRd'))
  scat2 = ax.scatter(-x,y,z, c=z, marker='o',picker=5, cmap = plt.cm.get_cmap('YlOrRd'))


  #What happens when we click on a point?
  def onpick(event):
    #Make sure we're clicking within the correct subplot, as only one on-click for entire figure
    if event.artist!=scat1 and event.artist!=scat2: 
      return True

    if not len(event.ind): return True
    im.clear()
    ind = event.ind[0]

    j = data.iloc[ind].loc['jib']
    m = data.iloc[ind].loc['jib']
    wSpd = data.iloc[ind].loc['windSpeed']
    wDir = data.iloc[ind].loc['windDir']
    bs = data.iloc[ind].loc['boatSpeed']

    #Assemble numerics
    dispStr = 'Wind Speed: {:.2f} kts'.format(wSpd)
    dispStr = dispStr + '\nWind Angle: {:.2f} deg'.format(wDir)
    dispStr = dispStr + '\nMain Angle: {:.2f}'.format(m)
    dispStr = dispStr + '\nJib Angle: {:.2f}'.format(j)
    dispStr = dispStr + '\nBoat Speed: {:.2f} kts'.format(bs)

    #Display numerics
    im.text(0.5, 0.75, dispStr,ha='center',fontsize=15,bbox={'facecolor':'blue', 'alpha':0.5, 'pad':10})

    #Plot the graphic
    arr = boatImage(m,j)
    offset = OffsetImage(arr,zoom = 2)
    ab = AnnotationBbox(offset, (0.5,0.3), xycoords='data', frameon=False)
    im.add_artist(ab)
    im.annotate('W', xy=(0.5,0.3),
      xytext=(0.5 - 0.2*np.sin(np.deg2rad(wDir)),0.3+ 0.2*np.cos(np.deg2rad(wDir))),
      arrowprops=dict(facecolor='blue'))
    im.axes.get_xaxis().set_ticks([])   #Turn off numbers
    im.axes.get_yaxis().set_ticks([])
    fig.canvas.draw()   #Update everything

  #These get overwritten if they aren't last
  ax.invert_xaxis()
  ax.invert_yaxis()

  fig.canvas.mpl_connect('pick_event', onpick)  #Add the picker
  plt.show()


def sliceRawData(data):
  beatingAngle = 45 #Where do we draw the no go lines?
  scat1 = None
  scat2 = None

  #Total figure
  fig = plt.figure()
  fig.set_size_inches(18, 12,forward=True)          #Manually resize
  
  #Polar Plot
  gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) #Breaks up the figure according to special ratio
  ax = fig.add_subplot(gs[0], projection='polar')      #The main figure
  subplots_adjust(bottom=0.20)
  ax.set_theta_zero_location('N')
  ax.set_theta_direction(-1)
  
  #Inspection pane
  im = fig.add_subplot(gs[1])                       #The point inspector pane
  im.axis('equal')
  im.axis([0,1,0,1])
  im.axes.get_xaxis().set_ticks([])
  im.axes.get_yaxis().set_ticks([])
  im.text(0.5, 0.75, 'Click on a point\n for more info',ha='center',fontsize=15,bbox={'facecolor':'blue', 'alpha':0.5, 'pad':10})

  #General layout
  plt.tight_layout(pad=4.08, h_pad=None, w_pad=5, rect=None)  #Layout the plots nicely

  #What happens when we click on a point?  Copied from vizRawData()
  def onpick(event):
    #Make sure we're clicking within the correct subplot, as only one on-click for entire figure
    # if event.artist!=scat1 and event.artist != scat2:
    #   print("artist is not scat") 
    #   return True

    if not len(event.ind): return True
    im.clear()
    ind = event.ind[0]
    l = data[data['windSpeed'] > sSpd.val - 0.5]
    tot = l[l['windSpeed'] < sSpd.val + 0.5]
    j = tot.iloc[ind].loc['jib']
    m = tot.iloc[ind].loc['main']
    wSpd = tot.iloc[ind].loc['windSpeed']
    wDir = tot.iloc[ind].loc['windDir']
    bs = tot.iloc[ind].loc['boatSpeed']

    #Assemble numerics
    dispStr = 'Wind Speed: {:.2f} kts'.format(wSpd)
    dispStr = dispStr + '\nWind Angle: {:.2f} deg'.format(wDir)
    dispStr = dispStr + '\nMain Angle: {:.2f}'.format(m)
    dispStr = dispStr + '\nJib Angle: {:.2f}'.format(j)
    dispStr = dispStr + '\nBoat Speed: {:.2f} kts'.format(bs)

    #Display numerics
    im.text(0.5, 0.75, dispStr,ha='center',fontsize=15,bbox={'facecolor':'blue', 'alpha':0.5, 'pad':10})

    #Plot the graphic
    arr = boatImage(m,j)
    offset = OffsetImage(arr,zoom = 2)
    ab = AnnotationBbox(offset, (0.5,0.3), xycoords='data', frameon=False)
    im.add_artist(ab)
    im.annotate('W', xy=(0.5,0.3),
      xytext=(0.5 - 0.2*np.sin(np.deg2rad(wDir)),0.3+ 0.2*np.cos(np.deg2rad(wDir))),
      arrowprops=dict(facecolor='blue'))
    im.axes.get_xaxis().set_ticks([])   #Turn off numbers
    im.axes.get_yaxis().set_ticks([])
    fig.canvas.draw()   #Update everything


  def plotWindSpeed(wSpd):
    l = data[data['windSpeed'] > wSpd - 0.5]
    tot = l[l['windSpeed'] < wSpd + 0.5]
    scat1 = ax.scatter(np.deg2rad(tot.loc[:,'windDir']),tot.loc[:,'boatSpeed'],c = tot.loc[:,'boatSpeed'],picker=5, cmap = plt.cm.get_cmap('YlOrRd'))
    scat2 = ax.scatter(np.deg2rad(360-tot.loc[:,'windDir']),tot.loc[:,'boatSpeed'],c=tot.loc[:,'boatSpeed'], picker=5, cmap = plt.cm.get_cmap('YlOrRd'))
 
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
  
    ax.set_rmax(12)
    ax.grid(True)
    ax.set_title('BoatSpeed for Given Windspeed')

  #Slider info
  axcolor = 'lightgoldenrodyellow'
  axSpd = axes([0.15, 0.05, 0.45, 0.03], axisbg=axcolor)
  sSpd = Slider(axSpd, 'Wind Speed', 0.01, 18.0, valinit=7)


  def update(val):
    ax.clear()
    plotWindSpeed(val)

  plotWindSpeed(sSpd.val)
  sSpd.on_changed(update)
  fig.canvas.mpl_connect('pick_event', onpick)  #Add the picker
  plt.show()

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