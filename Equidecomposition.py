from Primitives3D import *
from Utilities2D import *
import math

class PolygonCut(object):
	def __init__(self):
		self.transform = Matrix4()#Initialize to identity matrix
		self.flag = False #Flag this polygon for special drawing (for debugging purposes)
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
			if P.y > self.maxY:
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
	
	def getArea(self):
		return getPolygonArea(self.points)


class BiPolygonCut(PolygonCut):
	#Just like PolygonCut but with a bit of extra information for debugging
	#-transform represents the transform from the first polygon to the second polygon
	#-transform1 represents the transform from the first polgon to the intermediate polygon
	#-transform2 represents the transform from the the second polygon to the intermediate polygon
	#Therefore, transform = transform2.Inverse()*transform1
	def __init__(self):
		self.transform1 = Matrix4()
		self.transform2 = Matrix4()

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
		for k in range(0, len(cuts[i].points)):
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
				cut1.points.append(cuts[i].points[index])
				if index == index2:
					break
			cut1.points.append(P2)
			
			#Traverse new polygon 2 in clockwise order
			cut2.points.append(P2)
			index = index2
			while True:
				index = (index+1)%len(cuts[i].points)
				cut2.points.append(cuts[i].points[index])
				if index == index1:
					break
			cut2.points.append(P1)

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
	P1 = Point3D(A.x-2*w, A.y+h, 0)
	P2 = Point3D(A.x+2*w, A.y+h, 0)
	cutWithSegment(cuts, P1, P2)
	#The polygons that need to be translated
	#are the ones contained within the upper cut
	#They need to be translated to the right and down
	translation = Matrix4([ 1, 0, 0, w/2, 
							0, 1, 0, -h, 
							0, 0, 1, 0, 
							0, 0, 0, 1 ])
	#Now add all of the cuts that are inside the top half
	bottomY = A.y + h
	topY = A.y + h*2
	for poly in cuts:
		inside = True
		for P in poly.points:
			if P.y > (topY + EPS) or P.y < (bottomY - EPS):
				#print "Polygon %i point %i is at Y location %g outside of <%g, %g>\n\n"%(i, k, poly.points[k].y, bottomY, topY)
				inside = False
				break
		if inside:
			poly.transform = translation * poly.transform
			poly.points = [translation*P for P in poly.points]

def cutVertically(cuts, A, w, h):
	P1 = Point3D(A.x+w, A.y-2*h, 0)
	P2 = Point3D(A.x+w, A.y+2*h, 0)
	cutWithSegment(cuts, P1, P2)
	#The polygons that need to be translated
	#are the ones contained within the right cut
	#They need to be translated to the left and up
	translation = Matrix4([ 1, 0, 0, -w, 
							0, 1, 0, h/2, 
							0, 0, 1, 0, 
							0, 0, 0, 1 ])
	#Now add all of the cuts that are inside the right half
	leftX = A.x + w
	rightX = A.x + w*2
	for poly in cuts:
		inside = True
		for P in poly.points:
			if P.x > (rightX+EPS) or P.x < (leftX-EPS):
				#print "Polygon %i point %i is at Y location %g outside of <%g, %g>\n\n"%(i, k, poly->points[k].y, bottomY, topY)
				inside = False
				break
		if inside:
			poly.transform = translation * poly.transform
			poly.points = [translation*P for P in poly.points]

#This function cuts the rectangle in half until width < 2*height && height < 2*width
#Point A is the bottom left corner
#Returns the new width and height of the rectangle
#NOTE: This function assumes that the rectangles are axis-aligned
def cutRectangleToCorrectProportions(cuts, A, width, height):
	[w, h] = [width, height]
	while h > 2*w:
		#Make a horizontal cut to cut the polygon in half
		w = w*2.0
		h = h/2.0
		cutHorizontally(cuts, A, w, h)
	while w > 2*h:
		w = w/2.0
		h = h*2.0
		cutVertically(cuts, A, w, h)
	return [w, h]

#Fill the list "inside" with pointers to all of the cuts that are inside
#of the triangle ABC
#Allow some numerical tolerance for making this call
def getCutsInsideTriangle(cuts, A, B, C, inside):
	#Make sure points are in clockwise order
	if CCW2D(A, B, C) < 1:
		temp = A
		A = C
		C = temp
	for poly in cuts:
		isInside = True
		for P in poly.points:
			CLOSENESS_EPS = 1 #TODO: Tweak this parameter
			isInside = pointInsideTriangle2D(A, B, C, P, CLOSENESS_EPS)
			if not isInside:
				break
		if isInside:
			inside.append(poly)

def rectRotate90CCW(cuts, rectCorner, rectw):
	#Rotate the polygon so that this is the case
	translation = Matrix4([ 1, 0, 0, rectCorner.x,
							0, 1, 0, rectCorner.y,
							0, 0, 1, 0,
							0, 0, 0, 1 ])
	#Rotate it 90 degrees about the lower left point and then slide it back 
	#to the right
	R = Matrix4([0, -1, 0, 0,
				1, 0, 0, 0,
				0, 0, 1, 0,
				0, 0, 0, 1] )
	finalTrans = Matrix4([  1, 0, 0, rectw, 
							0, 1, 0, 0, 
							0, 0, 1, 0, 
							0, 0, 0, 1] )
	transformation = finalTrans*(translation*(R*translation.Inverse()))
	for i in range(0, len(cuts)):
		cuts[i].transform = transformation*cuts[i].transform
		cuts[i].points = [transformation*P for P in cuts[i].points]

#Use the "escalator method" to cut rectangles into rectangles
#This function assumes that both rectangles are axis-aligned and share the lower left corner
#at "rectCorner".  It also *very importantly* assumes that rect1w < 2*rect1h, rect1h < 2*rect1w,
#and rect2w < 2*rect2h and rect2h < 2*rect2w
def cutRectangleIntoRectangle(cuts, rectCorner, rect1w, rect1h, rect2w, rect2h, cutPoints):
	if rect1w > 2*rect1h or rect1h > 2*rect1w or rect2w > 2*rect2h or rect2h > 2*rect2w:
		print "ERROR: Rectangles do not have the correct width/height ratio to cut one into the other"
		return
	needToRotate = False #Do we need to rotate the polygon back 90 degrees CCW at the end
	if rect2h > rect2w:
		#Want the long side of the target polygon to be its width
		temp = rect2h
		rect2h = rect2w
		rect2w = temp
		needToRotate = True
	if rect1h < rect1w:
		#Want the long side of the original polygon to be its height
		temp = rect1h
		rect1h = rect1w
		rect1w = temp
		rectRotate90CCW(cuts, rectCorner, rect1w)

	#Now actually make the two cuts and move the triangles down
	#STEP 1: Come up with points that represent the boundaries of the rectangles
	#and the intersections between the rectangles
	A = Point3D(rectCorner.x, rectCorner.y, 0)
	B = Point3D(A.x, A.y+rect1h, 0)
	C = Point3D(B.x+rect1w, B.y, 0)
	D = Point3D(C.x, A.y, 0)
	E = Point3D(A.x, A.y+rect2h, 0)
	F = Point3D(E.x+rect1w, E.y, 0)
	G = Point3D(E.x+rect2w, E.y, 0)
	H = Point3D(A.x+rect2w, A.y, 0)
	intersection = intersectSegments2D(B, H, E, F, True)
	if not intersection:
		print "ERROR: Unable to find point 'I' cutting rectangle into another rectangle"
		return
	I = Point3D(intersection.x, intersection.y, 0)
	intersection = intersectSegments2D(B, H, C, D, True)
	if not intersection:
		print "ERROR: Unable to find point 'J' cutting rectangle into another rectangle"
		return
	J = Point3D(intersection.x, intersection.y, 0)
	for P in [A, B, C, D, E, F, G, H, I, J]:
		cutPoints.append(P)

	#STEP 2: Cut out the big triangle and the little triangle
	vHI = I - H;
	seg1P = Point3D(I.x+vHI.x, I.y+vHI.y, 0)
	seg2P = H
	cutWithSegment(cuts, seg1P, seg2P)

	vGE = E - G
	seg1P = Point3D(E.x+vGE.x, E.y, 0)
	seg2P = G
	cutWithSegment(cuts, seg1P, seg2P)
	
	#STEP 3: Move big triangle and little triangle
	insideBigTriangle = []
	getCutsInsideTriangle(cuts, B, C, J, insideBigTriangle)
	translation = Matrix4([ 1, 0, 0, rect2w-rect1w,
							0, 1, 0, -(J.y-D.y),
							0, 0, 1, 0,
							0, 0, 0, 1])

	for poly in insideBigTriangle:
		poly.transform = translation*poly.transform
		poly.points = [translation*P for P in poly.points]

	insideLittleTriangle = []
	getCutsInsideTriangle(cuts, E, B, I, insideLittleTriangle)
	translation = Matrix4([ 1, 0, 0, J.x-B.x,
							0, 1, 0, J.y-B.y,
							0, 0, 1, 0,
							0, 0, 0, 1])

	for poly in insideLittleTriangle:
		poly.transform = translation*poly.transform
		poly.points = [translation*P for P in poly.points]

	if needToRotate:
		#Need to flip around the width and the height of the rectangle
		rectRotate90CCW(cuts, rectCorner, rect2h)

#rectCorner is the bottom left corner of the target rectangle
#rectw and recth are the width and height, respectively, of the target rectangle
#cutToDimensions specifies whether the triangle should be cut to the dimensions of the
#target rectangle, or if only the initial triangle to "natural" rectangle cut should be made
#Returns [width, height] of the cut (will be [rectw, recth] if cutToDimensions is True
#but if cutToDimensions is false it will be the width and height of the "natural rectangle")
def cutTriangleIntoRectangle(cuts, A, B, C, rectCorner, rectw, recth, cutPoints, cutToDimensions):
	#STEP 1: Make sure the points are specified in clockwise order and
	#fix them if this is not the case (this is important for step 3)
	if (CCW2D(A, B, C) < 0):
		temp = A
		A = C
		C = temp

	#STEP 2: Cut the triangle into its "natural rectangle" by making 4 cuts
	#Preliminary step: Find a vertex whose projection onto its adjacent
	#segment is within the bounds of that segment that maximizes the minimum
	#angle formed in the resulting two triangles
	[PA, PB, PC] = [A, B, C]
	AScore = triangleHalfCutScore(A, B, C)
	BScore = triangleHalfCutScore(B, C, A)
	CScore = triangleHalfCutScore(C, A, B)
	score = -1
	if AScore > BScore and AScore > CScore:
		[PA, PB, PC] = [A, B, C]
		score = AScore
	elif BScore > AScore and BScore > CScore:
		[PA, PB, PC] = [B, C, A]
		score = BScore
	else:
		[PA, PB, PC] = [C, A, B]
		score = CScore
	if score < 0:
		print "Error: Could not find a triangle vertex that projected onto the opposite edge"
		return
	
	vBA = PA - PB
	vBC = PC - PB
	vBD = vBC.proj(vBA)
	D = Point3D(PB.x+vBD.x, PB.y+vBD.y, 0)
	vDA = PA - D
	vDE = Vector3D(vDA.x*0.5, vDA.y*0.5, 0)
	E = Point3D(D.x+vDE.x, D.y+vDE.y, 0)
	dummy = Point3D(E.x+vBC.x, E.y+vBC.y, 0)
	intersect = intersectSegments2D(PA, PC, E, dummy, False)
	if not intersect:
		print "ERROR: Could not find point 'F' in initial triangle cut"
		return
	F = Point3D(intersect.x, intersect.y, 0)
	dummy = Point3D(E.x-vBC.x, E.y-vBC.y, 0)
	intersect = intersectSegments2D(PA, PB, E, dummy, False)
	if not intersect:
		print "ERROR: Could not find point 'G' in initial triangle cut"
		return
	G = Point3D(intersect.x, intersect.y, 0)
	#Now add the four initial polygons
	#Upper left triangle
	poly1 = PolygonCut()
	for P in [F, PA, E]:
		poly1.points.append(P)
	#Rotate 180 degrees around point F
	poly1.transform = Matrix4([-1, 0, 0, 2*F.x,
								0, -1, 0, 2*F.y,
								0, 0, 1, 0,
								0, 0, 0, 1])
	poly1.points = [poly1.transform*P for P in poly1.points]
	cuts.append(poly1)

	#Upper right triangle
	poly2 = PolygonCut()
	for P in [E, PA, G]:
		poly2.points.append(P)
	#Rotate 180 degrees around point G
	poly2.transform = Matrix4([-1, 0, 0, 2*G.x,
								0, -1, 0, 2*G.y,
								0, 0, 1, 0,
								0, 0, 0, 1])
	poly2.points = [poly2.transform*P for P in poly2.points]
	cuts.append(poly2)

	#Lower trapezoid
	poly3 = PolygonCut()
	for P in [PC, F, G, PB]:
		poly3.points.append(P)
	cuts.append(poly3)

	#STEP 3: Rotate the new rectangle so that it is axis-aligned, and
	#translate it so that its lower left corner matches the lower left
	#corner of the target polygon
	vPCD = D - PC
	vDE = E - D
	vPCD.normalize()
	vDE.normalize()
	R = Matrix4( [  vPCD.x, vDE.x, 0, 0,
					vPCD.y, vDE.y, 0, 0,
					0, 0, 1, 0,
					0, 0, 0, 1 ])
	R = R.Inverse() #This matrix will rotate the proper amount around the origin
	translation = Matrix4([ 1, 0, 0, cuts[2].points[0].x,
							0, 1, 0, cuts[2].points[0].y,
							0, 0, 1, 0,
							0, 0, 0, 1])
	rotateAxisAlign = (translation * R)*translation.Inverse()
	#Finally, translate the rectangle so that its bottom left corner aligns
	#with the bottom left corner of the target rectangle
	offset = rectCorner - cuts[2].points[0]
	translation = Matrix4([ 1, 0, 0, offset.x,
							0, 1, 0, offset.y,
							0, 0, 1, 0,
							0, 0, 0, 1])
	rotateAxisAlign = translation*rotateAxisAlign
	#Now apply this matrix to every polygonal cut to align the rectangle
	#with the target rectangle at the lower left corner
	for cut in cuts:
		cut.transform = rotateAxisAlign*cut.transform
		cut.points = [rotateAxisAlign*P for P in cut.points]

	#Determine the width and height of the rectangle, and cut it in half
	#until width < 2*height AND height < 2*width
	widthVector = cuts[2].points[3] - cuts[2].points[0]
	width = widthVector.squaredMag()**0.5
	heightVector = cuts[0].points[1] - cuts[0].points[2]
	height = heightVector.squaredMag()**0.5
	
	if not cutToDimensions:
		#Don't do any further cuts; just return the natural rectangle made
		#from the first three cut pieces, along with the width and height
		#of this rectangle by reference
		return [width, height]

	#Adjust the widths and heights of the target rectangle if necessary
	[targetw, targeth] = [rectw, recth]
	#TODO: This could be dangerous if either rectw or recth is close to 0
	while targetw > 2*targeth:
		targetw /= 2.0
		targeth *= 2.0
	while (targeth > 2*targetw):
		targeth /= 2.0
		targetw *= 2.0
	
	#Adjust the proportions of the new rectangle if necessary
	[width, height] = cutRectangleToCorrectProportions(cuts, rectCorner, width, height)
	#Cut the new rectangle into the (possibly adjusted) target rectangle
	cutRectangleIntoRectangle(cuts, rectCorner, width, height, targetw, targeth, cutPoints)

	#Now put the target rectangle back to its original proportions (if it had to be cut before)
	while (targetw + EPS < rectw):
		targetw *= 2
		targeth /= 2
		cutHorizontally(cuts, rectCorner, targetw, targeth)
	while (targeth + EPS < recth):
		targetw /= 2
		targeth *= 2
		cutVertically(cuts, rectCorner, targetw, targeth)
	return [targetw, targeth]

#cuts1 and cuts2 are the PolygonCuts to get from the triangles to the intermediate square
#cuts are the BiPolygonCuts to get from the two triangles to each other
def cutTriangleIntoTriangle(cuts1, cuts2, cuts, A, B, C, D, E, F):
	area = getPolygonArea([A, B, C])
	area2 = getPolygonArea([D, E, F])
	areaDiff = abs(area - area2)
	if  areaDiff > EPS:
		print "ERROR: Cannot cut triangles into each other with a nonzero difference of area of %g"%areaDiff
		return
	squareDim = math.sqrt(area)
	squareCorner = Point3D(50, 50, 0)
	cutPoints1 = []
	cutTriangleIntoRectangle(cuts1, A, B, C, squareCorner, squareDim, squareDim, cutPoints1, True)
	cutPoints2 = []
	cutTriangleIntoRectangle(cuts2, D, E, F, squareCorner, squareDim, squareDim, cutPoints2, True)
	#Now check every cut against every other cut (TODO: Speed this up with KD trees or something?)
	for cut1 in cuts1:
		for cut2 in cuts2:
			intersection = clipSutherlandHodgman(cut1.points, cut2.points)
			if len(intersection) > 0:
				trans1 = cut1.transform
				trans2Inv = cut2.transform.Inverse()
				newCut = BiPolygonCut()
				newCut.points = intersection
				newCut.points = [trans2Inv*P for P in newCut.points]
				newCut.transform = trans2Inv*trans1
				newCut.transform1 = cut1.transform
				newCut.transform2 = cut2.transform
				cuts.append(newCut)
				print "Found intersection of length %i"%len(intersection)

def drawPolygon2DTk(canvas, poly, color = "#0000FF", drawVertices = True):
	if drawVertices:
		for P in poly:
			canvas.create_oval(P.x-4, P.y+4, P.x+4, P.y-4, fill="#000000")
	for i in range(0, len(poly)):
		[P1, P2] = [poly[i], poly[(i+1)%len(poly)]]
		canvas.create_line(P1.x, P1.y, P2.x, P2.y, fill=color)

#t is the interpolation parameter that says how far along to slide the cut
def drawPolygonCut(canvas, height, poly, t, color = "#0000FF"):
	t = (t+t**0.5)/2
	cosA = poly.transform.m[0]
	sinA = poly.transform.m[4]
	dx = poly.transform.m[3]*t
	dy = poly.transform.m[7]*t
	A = math.atan2(sinA, cosA)*t
	[cosA, sinA] = [math.cos(A), math.sin(A)]
	trans1 = Matrix4( [ cosA, -sinA, 0, dx,
						sinA, cosA, 0, dy,
						0, 0, 1, 0,
						0, 0, 0, 1] )
	trans = trans1*(poly.transform.Inverse())
	points = [trans*P for P in poly.points]
	for i in range(0, len(points)):
		[P1, P2] = [points[i], points[(i+1)%len(points)]]
		#canvas.create_oval(P1.x-4, P1.y+4, P1.x+4, P1.y-4, fill="#000000")
		canvas.create_line(P1.x, P1.y, P2.x, P2.y, fill=color)

def drawPolygonCuts(canvas, height, polygonCuts, t, color = "#0000FF"):
	for poly in polygonCuts:
		drawPolygonCut(canvas, height, poly, t, color)

def getAreaOfCuts(cuts):
	return sum([cut.getArea() for cut in cuts])

if __name__ == '__main__':
	cuts = []
	cutPoints = []
	A = Point3D(-1, 0, 0)
	B = Point3D(0, 1, 0)
	C = Point3D(1, 0, 0)
	rectCorner = Point3D(0, 0, 0)
	rectw = 1
	recth = 1
	cutTriangleIntoRectangle(cuts, A, B, C, rectCorner, rectw, recth, cutPoints, True)
