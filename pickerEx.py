#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('click on points')

ax.scatter(np.random.rand(100),np.random.rand(100),picker=True)  # 5 points tolerance

def onpick(event):
	ind = event.ind
	xpt = np.take(x,ind)
	print ('onpick points:' + str(xpt))

fig.canvas.mpl_connect('pick_event', onpick)

plt.show()