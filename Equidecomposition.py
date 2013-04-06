from Shapes3D import *
from Primitives3D import *


class BiPolygonCut(object):
	def __init__(self):
		self.transform1 = Matrix4()
		self.transform2 = Matrix4()

class PolygonCut(object):
	def __init__(self):
		self.transform = Matrix4()#Initialize to identity matrix
		self.flag = false #Flag this polygon for special drawing (for debugging purposes)
		self.minX = float('inf')
		self.maxX = float('-inf')
		self.minY = float('inf')
		self.maxY = float('-inf')
		self.points = []
	
	#Bounding boxes used to speed up intersection tests (hopefully)
	def updateBoundingBox(self):
		self.minX = float('inf')
		self.maxX = float('-inf')
		self.minY = float('inf')
		self.maxY = float('-inf')
		for P in self.points:
			if P.x < self.minX:
				self.minX = P.x
			if P.x > self.maxX:
				self.maxX = P.x
			if P.y < self.minY:
				self.minY = P.y
			if P.y > self.maxY
				self.maxY = P.y
	
	def boundingBoxIntersects(self, other):
		[x1, x2, y1, y2] = [other.minX, other.maxX, other.minY, other.maxY]
		[xmin, xmax, ymin, ymax] = [self.minX, self.maxX, self.minY, self.maxY]
		#Check to see if any four vertices of the first box are within the second box
		if (x1 >= xmin and x1 <= xmax and y1 >= ymin and y1 <= ymax):
			return True
		if (x2 >= xmin and x2 <= xmax and y1 >= ymin and y1 <= ymax):
			return True
		if (x2 >= xmin and x2 <= xmax and y2 >= ymin and y2 <= ymax):
			return True
		if (x1 >= xmin and x1 <= xmax and y2 >= ymin and y2 <= ymax):
			return True
		#Do the same for the other box
		[x1, x2, y1, y2] = [self.minX, self.maxX, self.minY, self.maxY]
		[xmin, xmax, ymin, ymax] = [other.minX, other.maxX, other.minY, other.maxY]
		if (x1 >= xmin and x1 <= xmax and y1 >= ymin and y1 <= ymax):
			return True
		if (x2 >= xmin and x2 <= xmax and y1 >= ymin and y1 <= ymax):
			return True
		if (x2 >= xmin and x2 <= xmax and y2 >= ymin and y2 <= ymax):
			return True
		if (x1 >= xmin and x1 <= xmax and y2 >= ymin and y2 <= ymax):
			return True
		return False


#Return the "score" of dropping a vertical line from point A to the
#segment BC in the triangle ABC, where the "score" is higher if
#the minimum angle formed is higher.  Return -1 if that dropped
#vertical line is not completely contained within the triangle
def triangleHalfCutScore(A, B, C):
	vBA = A - B
	vBC = C - B
	proj = vBC.proj(vBA)
	dot = proj.Dot(vBC)
	if dot > EPS and dot < vBC.squaredMag()-EPS:
		#Score the cut based on the minimum angle in the two
		#triangles that are formed
		score = 0
		D = Point3D(B.x+proj.x, B.y+proj.y, 0)
		score = min(score, 1-COSBetween(A, B, D))
		score = min(score, 1-COSBetween(B, A, D))
		score = min(score, 1-COSBetween(D, A, B))
		score = min(score, 1-COSBetween(A, D, C))
		score = min(score, 1-COSBetween(D, A, C))
		score = min(score, 1-COSBetween(C, A, D))
		#Uncomment this to do the score by minimum area instead
		#of minimum angle
		#score = float('inf')
		#score = min(score, getTriangleArea(A, C, D))
		#score = min(score, getTriangleArea(D, A, B))
		return score
	return -1#The line cut isn't within the polygon


#Cut the polygons in "cuts" with the segment AB.  This may slice 
#some of the polygons in "cuts" in half
def cutWithSegment(cuts, A, B):
	removeIndices = []#If a polygon got split remove the unsplit
	#parent polygon. This array stores the indices of those parent polygons
	parentPolygonNum = len(cuts)
	for i in range(0, parentPolygonNum):
		index1 = -1#Index of first intersection in polygon
		index2 = -1#Index of second intersection in polygon
		intersectP1 = None
		intersectP2 = None
		for k in range(0, len(cuts[i].points):
			P1 = cuts[i].points[k]
			P2 = cuts[i].points[(k+1)%len(cuts[i].points)]
			intersection = intersectSegments2D(A, B, P1, P2) #Allow endpoint intersections
			validIntersection = False
			#NOTE: This should properly handle the case where an intersection hits
			#and endpoint of an edge
			if intersection:
				if not intersectP1:
					intersectP1 = intersection
					index1 = k
					validIntersection = True
				elif not PointsEqual(intersectP1, intersection):
					intersectP2 = intersection
					index2 = k
					validIntersection = True
		if intersectP1 and intersectP2:
			#A valid cut exists splitting this polygon in half
			#Split off two new polygons
			cut1 = PolygonCut()
			cut2 = PolygonCut()
			P1 = Point3D(intersectP1.x, intersectP1.y, 0)
			P2 = Point3D(intersectP2.x, intersectP2.y, 0)
			#The new cuts inhherit the transformations from their parent
			cut1.transform = cuts[i].transform;
			cut2.transform = cuts[i].transform;
			#Traverse new polygon 1 in clockwise order
			#(this is important becuase subsequent intersection algorithms assume
			#that the polygons are clockwise)
			cut1.points.append(P1)
			index = index1
			while True:
				index = (index+1)%len(cuts[i].points)
				cut1.points.append(cuts[i]->points[index])
				if index == index2:
					break
			cut1.points.append(P2)
			
			#Traverse new polygon 2 in clockwise order
			cut2.points.append(P2)
			index = index2
			while True:
				index = (index+1)%len(cuts[i].points)
				cut2.points.append(cuts[i]->points[index])
				if index == index1:
					break
			cut22points.append(P1)

			#Add the two new polygons to the list of cuts
			cuts.append(cut1)
			cuts.append(cut2)

			#Add the parent to the deletion list
			removeIndices.append(i)
	#print "%i polygons got cut in half"%len(removeIndices)
	#Remove all of the parent polygons
	for k in removeIndices:
		cuts[k] = cuts[-1]
		cuts.pop()

def cutHorizontally(cuts, A, w, h):
	P1 = Point3D(A.x-10, A.y+h, 0)
	P2 = Point3D(A.x+w, A.y+h, 0)
	cutWithSegment(cuts, P1, P2)
	#The polygons that need to be translated
	#are the ones contained within the upper cut
	#They need to be translated to the right and down
	translation = Matrix4([1, 0, 0, w/2, 0, 1, 0, -h, 0, 0, 1, 0, 0, 0, 0, 1])
	#Now add all of the cuts that are inside the top half
	bottomY = A.y + h
	topY = A.y + h*2
	for i in range(0, len(cuts)):
		inside = True
		poly = cuts[i]
		for k in range(0, len(poly.points)):
			if poly.points[k].y > (topY + EPS) or poly.points[k].y < (bottomY - EPS):
				#print "Polygon %i point %i is at Y location %g outside of <%g, %g>\n\n"%(i, k, poly.points[k].y, bottomY, topY)
				inside = False
				break
		if inside:
			cuts[i].transform = translation * cuts[i].transform
			cuts[i].points = [translation*P for P in cuts[i].points]

