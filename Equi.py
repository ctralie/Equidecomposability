import sys
sys.path.append('G-RFLCT')
from sys import argv
from Shapes3D import *
from Primitives3D import *
from QuickHull import *
from Tkinter import *

def repaint(canvas):
	canvas.delete(ALL)
	#for P in GlobalVars.points:
	#	canvas.create_oval(P.x-4, GlobalVars.res-P.y+4, P.x+4, GlobalVars.res-P.y-4, fill="#000000")

if __name__ == '__main__':
	def mouseClicked(event):
		canvas = event.widget
		x = canvas.canvasx(event.x)
		y = canvas.canvasy(event.y)
	
	def mouse2Clicked(event):
		#Recompute hull
		canvas = event.widget
		GlobalVars.hull = []
		GlobalVars.segments = []
		QuickHull(GlobalVars.points, GlobalVars.hull, GlobalVars.segments)
		repaint(canvas)
		
	def mouse3Clicked(event):
		#Erase last point
		canvas = event.widget
		repaint(canvas)
	
	root = Tk()
	root.title('Equi')
	
	w = Canvas(root, width=GlobalVars.res, height=GlobalVars.res)
	w.pack()
	repaint(w)
	w.bind("<Button-1>", mouseClicked)
	w.bind("<Button-2>", mouse2Clicked)
	w.bind("<Button-3>", mouse3Clicked)
	root.mainloop()
