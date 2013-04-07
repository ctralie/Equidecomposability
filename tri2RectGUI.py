from Primitives3D import *
from Equidecomposition import *
from Tkinter import *
import time
from threading import Thread

class Display(object):
	def __init__(self, pw = 800, ph = 600):
		self.points = []
		self.cuts = []
		self.cutPoints = []
		self.t = 0 #Animation parameter
		[self.width, self.height] = [pw, ph]
		self.root = Tk()
		self.root.title('Cutting Triangle Into Rectangle')
	
		self.canvas = Canvas(self.root, width=pw, height=ph)
		self.canvas.pack()
		self.repaint()
		self.canvas.bind("<Button-1>", self.mouseClicked)
		self.canvas.bind("<Button-2>", self.mouse2Clicked)
		self.canvas.bind("<Button-3>", self.mouse3Clicked)
		self.root.mainloop()

	def repaint(self):
		self.canvas.delete(ALL)
		if True:
			h = self.height
			for P in self.points:
				self.canvas.create_oval(P.x-4, P.y+4, P.x+4, P.y-4, fill="#000000")
			for i in range(0, len(self.points)):
				[P1, P2] = [self.points[i], self.points[(i+1)%len(self.points)]]
				self.canvas.create_line(P1.x, P1.y, P2.x, P2.y, fill="#0000FF")
		if len(self.cuts) > 0:
			drawPolygonCuts(self.canvas, self.height, self.cuts, self.t)

	def AnimatePieces(self):
		#Now animate
		self.t = 0
		while self.t < 1:
			self.repaint()
			self.t = self.t + 0.01
			time.sleep(0.02)
		self.repaint()

	def mouseClicked(self, event):
		canvas = event.widget
		x = canvas.canvasx(event.x)
		y = canvas.canvasy(event.y)
		self.points.append(Point3D(x, y, 0))
		self.repaint()

	def mouse3Clicked(self, event):
		if len(self.points) > 0:
			self.points.pop()
			self.repaint()

	def mouse2Clicked(self, event):
		#Do the animation
		if len(self.points) < 3:
			return
		self.cuts = []
		self.cutPoints = []
		[A, B, C] = self.points[0:3]
		area = getPolygonArea([A, B, C])
		height = 50
		width = area/height
		rectCorner = Point3D(10, 10, 0)
		cutTriangleIntoRectangle(self.cuts, A, B, C, rectCorner, width, height, self.cutPoints, True)
		thread = Thread(target = self.AnimatePieces)
		thread.start()

if __name__ == '__main__':
	display = Display()
