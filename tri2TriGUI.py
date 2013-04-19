from Primitives3D import *
from Equidecomposition import *
from Tkinter import *
import time
from threading import Thread

class Display(object):
	def __init__(self, pw = 800, ph = 600):
		self.ClearAll()
		
		self.t = 0 #Animation parameter
		[self.width, self.height] = [pw, ph]
		self.root = Tk()
		self.root.title('Cutting Triangles Into Triangles')
	
		self.canvas = Canvas(self.root, width=pw, height=ph)
		self.canvas.grid(row=0, column=0, rowspan = 5)
		self.canvas.bind("<Button-1>", self.mouseClicked)
		self.canvas.bind("<Button-2>", self.mouse2Clicked)
		self.canvas.bind("<Button-3>", self.mouse3Clicked)
		
		selTri1Button = Button(self.root, text = "Select Triangle1", command = self.selectTriangle1)
		selTri1Button.grid(row=0, column=1)
		selTri2Button = Button(self.root, text = "Select Triangle2", command = self.selectTriangle2)
		selTri2Button.grid(row=1, column=1)
		cutButton = Button(self.root, text="Do Cut", command=self.doTriangleCuts)
		cutButton.grid(row=2, column=1)
		animateButton = Button(self.root, text = "Animate", command=self.doAnimation)
		animateButton.grid(row=3, column=1)
		resetButton = Button(self.root, text = "Reset", command=self.Reset)
		resetButton.grid(row=4, column=1)
		
		self.repaint()
		self.root.mainloop()

	def selectTriangle1(self):
		self.selectingTri1 = True
	
	def selectTriangle2(self):
		self.selectingTri1 = False

	def ClearAll(self):
		self.cuts = []
		self.cuts1 = []
		self.cuts2 = []
		self.selectingTri1 = True
		self.tri1Points = []
		self.tri2Points = []
		#Debugging points for checking polygon intersections
		self.selPoints1 = []
		self.selPoints2 = []
		self.intPoints = []
	
	def Reset(self):
		self.ClearAll()
		self.repaint()

	def repaint(self):
		self.canvas.delete(ALL)
		if len(self.cuts1) == 0:
			drawPolygon2DTk(self.canvas, self.tri1Points)
			drawPolygon2DTk(self.canvas, self.tri2Points)
		else:
			drawPolygonCuts(self.canvas, self.height, self.cuts, self.t)
			#drawPolygonCuts(self.canvas, self.height, self.cuts1, self.t, "#0000FF")
			#drawPolygonCuts(self.canvas, self.height, self.cuts2, self.t, "#FF0000")
			if self.t >= 1:
				drawPolygonCuts(self.canvas, self.height, self.cuts, 0)
				#drawPolygonCuts(self.canvas, self.height, self.cuts1, 0, "#0000FF")
				#drawPolygonCuts(self.canvas, self.height, self.cuts2, 0, "#FF0000")
		#Draw selected debugging polygons
		drawPolygon2DTk(self.canvas, self.selPoints1, "#00FFFF")
		drawPolygon2DTk(self.canvas, self.selPoints2, "#FF00FF")
		drawPolygon2DTk(self.canvas, self.intPoints, "#FFFF00")

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
		if self.selectingTri1:
			self.tri1Points.append(Point3D(x, y, 0))
		else:
			self.tri2Points.append(Point3D(x, y, 0))
		self.repaint()

	def mouse2Clicked(self, event):
		canvas = event.widget
		x = canvas.canvasx(event.x)
		y = canvas.canvasy(event.y)
		cut1 = None
		cut2 = None
		P = Point3D(x, y, 0)
		#if len(self.tri1Points) >= 3:
		#	[A, B, C] = self.tri1Points[0:3]
		#	print pointInsideConvexPolygon2D([A, B, C], P);
		for cut in self.cuts1:
			if pointInsideConvexPolygon2D(cut.points, P):
				cut1 = cut
				break
		if not cut1:
			print "No cut found from triangle 1 that contains mouse click point"
			return
		for cut in self.cuts2:
			if pointInsideConvexPolygon2D(cut.points, P):
				cut2 = cut
				break
		if not cut2:
			print "No cut found from triangle 2 that contains mouse click point"
			return
		self.selPoints1 = cut1.points
		self.selPoints2 = cut2.points
		self.intPoints = clipSutherlandHodgman(cut1.points, cut2.points)
		#printPointsList(self.selPoints1, "Points1")
		#printPointsList(self.selPoints2, "Points2")
		self.repaint()

	def mouse3Clicked(self, event):
		points = self.tri1Points
		if not self.selectingTri1:
			points = self.tri2Points
		if len(points) > 0:
			points.pop()
			self.repaint()

	def doTriangleCuts(self):
		self.cuts = []
		self.cuts1 = []
		self.cuts2 = []
		if len(self.tri1Points) < 3 or len(self.tri2Points) < 3:
			return
		self.cuts = []
		[A, B, C] = self.tri1Points[0:3]
		[D, E, F] = self.tri2Points[0:3]
		area1 = getPolygonArea([A, B, C])
		area2 = getPolygonArea([D, E, F])
		#Rescale triangle2 to have the same area as triangle1
		ratio = math.sqrt(area1/area2)
		DE = E - D
		DF = F - D
		E = D + ratio*DE
		F = D + ratio*DF
		(self.tri2Points[1], self.tri2Points[2]) = (E, F)
		area2 = getPolygonArea([D, E, F])
		print "Triangle Area 1 = %g\nRescaled Triangle Area 2 = %g"%(area1, area2)
		cutTriangleIntoTriangle(self.cuts1, self.cuts2, self.cuts, A, B, C, D, E, F)
		print "Area of cuts = %g"%getAreaOfCuts(self.cuts)
		print "len(cuts1) = %i, len(cuts2) = %i, len(cuts) = %i"%(len(self.cuts1), len(self.cuts2), len(self.cuts))
		#Do the animation
		self.doAnimation()
	
	def doAnimation(self):
		if len(self.cuts) > 0:
			self.t = 0
			thread = Thread(target = self.AnimatePieces)
			thread.start()

if __name__ == '__main__':
	display = Display()
