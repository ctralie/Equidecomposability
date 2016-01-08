import math
import numpy as np
from Utilities2D import *

EPS = 1e-12

class PolygonCut(object):
    def __init__(self):
        self.transform = np.eye(4) #Initialize to identity matrix
        self.transforms = [] #Used to keep track of transformations as they happen
        self.flag = False #Flag this polygon for special drawing (for debugging purposes)
        self.minX = float('inf')
        self.maxX = float('-inf')
        self.minY = float('inf')
        self.maxY = float('-inf')
        self.points = []
    
    def getPointsStack(self):
        #Convert into an N x 3 numpy array
        stack = np.zeros((len(self.points), 3))
        for i in range(stack.shape[0]):
            stack[i, :] = self.points[i]        
        return stack
    
    #Bounding boxes used to speed up intersection tests (hopefully)
    def updateBoundingBox(self):
        stack = self.getPointsStack()
        self.minX = np.min(stack[:, 0])
        self.maxX = np.max(stack[:, 0])
        self.minY = np.min(stack[:, 1])
        self.maxY = np.max(stack[:, 1])
    
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
        return getPolygonArea(self.getPointsStack())
    
    def transformPoints(self, T):
        #Put the points into homogenous coordinates
        P = np.zeros((4, len(self.points)))
        P[3, :] = 1
        P[0:2, :] = self.getPointsStack().T
        P = T.dot(P)
        for i in range(P.shape[0]):
            self.points[i][0:2] = P[0:2, i]


class BiPolygonCut(PolygonCut):
    #Just like PolygonCut but with a bit of extra information for debugging
    #-transform represents the transform from the first polygon to the second polygon
    #-transform1 represents the transform from the first polgon to the intermediate polygon
    #-transform2 represents the transform from the the second polygon to the intermediate polygon
    #Therefore, transform = transform2.Inverse()*transform1
    def __init__(self):
        self.transform1 = np.eye(4)
        self.transform2 = np.eye(4)

#Return the "score" of dropping a vertical line from point A to the
#segment BC in the triangle ABC, where the "score" is higher if
#the minimum angle formed is higher.  Return -1 if that dropped
#vertical line is not completely contained within the triangle
def triangleHalfCutScore(A, B, C):
    vBA = A - B
    vBC = C - B
    proj = vBA*(vBC.dot(vBA)/np.sum(vBA**2))
    dot = proj.dot(vBC)
    if dot > EPS and dot < np.sum(vBC**2)-EPS:
        #Score the cut based on the minimum angle in the two
        #triangles that are formed
        score = 0
        D = np.array([B[0]+proj[0], B[1]+proj[1], 0])
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
            if intersection.size > 0:
                if not intersectP1:
                    intersectP1 = intersection
                    index1 = k
                    validIntersection = True
                elif not PointsEqual(intersectP1, intersection):
                    intersectP2 = intersection
                    index2 = k
                    validIntersection = True
        if intersectP1.size > 0 and intersectP2.size > 0:
            #A valid cut exists splitting this polygon in half
            #Split off two new polygons
            cut1 = PolygonCut()
            cut2 = PolygonCut()
            P1 = np.array([intersectP1[0], intersectP1[1], 0])
            P2 = np.array([intersectP2[0], intersectP2[1], 0])
            #The new cuts inhherit the transformations from their parent
            cut1.transform = cuts[i].transform
            cut1.transforms = list(cuts[i].transforms)
            cut2.transform = cuts[i].transform
            cut2.transforms = list(cuts[i].transforms)
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
    P1 = np.array([A[0]-2*w, A[1]+h, 0])
    P2 = np.array([A[0]+2*w, A[1]+h, 0])
    cutWithSegment(cuts, P1, P2)
    #The polygons that need to be translated
    #are the ones contained within the upper cut
    #They need to be translated to the right and down
    translation = np.eye(4)
    translation[0, 3] = w/2
    translation[1, 3] = -h
    #Now add all of the cuts that are inside the top half
    bottomY = A[1] + h
    topY = A[1] + h*2
    for poly in cuts:
        inside = True
        for P in poly.points:
            if P[1] > (topY + EPS) or P[1] < (bottomY - EPS):
                #print "Polygon %i point %i is at Y location %g outside of <%g, %g>\n\n"%(i, k, poly.points[k][1], bottomY, topY)
                inside = False
                break
        if inside:
            poly.transform = translation.dot(poly.transform)
            poly.transforms = [translation] + poly.transforms
            poly.transformPoints(translation)
        else:
            #If this polygon cut is not inside of the group that needs to be
            #translated, hold it in place during the animation
            poly.transforms = [np.eye(4)] + poly.transforms

def cutVertically(cuts, A, w, h):
    P1 = np.array([A[0]+w, A[1]-2*h, 0])
    P2 = np.array([A[0]+w, A[1]+2*h, 0])
    cutWithSegment(cuts, P1, P2)
    #The polygons that need to be translated
    #are the ones contained within the right cut
    #They need to be translated to the left and up
    translation = np.eye(4)
    translation[0, 3] = -w
    translation[1, 3] = h/2
    #Now add all of the cuts that are inside the right half
    leftX = A[0] + w
    rightX = A[0] + w*2
    for poly in cuts:
        inside = True
        for P in poly.points:
            if P[0] > (rightX+EPS) or P[0] < (leftX-EPS):
                #print "Polygon %i point %i is at Y location %g outside of <%g, %g>\n\n"%(i, k, poly->points[k][1], bottomY, topY)
                inside = False
                break
        if inside:
            poly.transform = translation.dot(poly.transform)
            poly.transforms = [translation] + poly.transforms
            poly.transformPoints(translation)
        else:
            #If this polygon cut is not inside of the group that needs to be
            #translated, hold it in place during the animation
            poly.transforms = [np.eye(4)] + poly.transforms

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
    translation = np.eye(4)
    translation[0:2, 3] = rectCorner[0:2]
    translationInv = np.eye(4)
    translationInv[0:2, 3] = -rectCorner[0:2]
    #Rotate it 90 degrees about the lower left point and then slide it back 
    #to the right
    R = np.array([0, -1, 0, 0,
                1, 0, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1])
    R = np.reshape(R, [4, 4])
    finalTrans = np.eye(4)
    finalTrans[0, 3] = rectw
    transformation = finalTrans.dot(translation.dot(R.dot(translationInv)))
    for i in range(0, len(cuts)):
        cuts[i].transform = transformation.dot(cuts[i].transform)
        cuts[i].transforms = [transformation] + cuts[i].transforms
        cuts[i].transformPoints(transformation)

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
    A = np.array([rectCorner[0], rectCorner[1], 0])
    B = np.array([A[0], A[1]+rect1h, 0])
    C = np.array([B[0]+rect1w, B[1], 0])
    D = np.array([C[0], A[1], 0])
    E = np.array([A[0], A[1]+rect2h, 0])
    F = np.array([E[0]+rect1w, E[1], 0])
    G = np.array([E[0]+rect2w, E[1], 0])
    H = np.array([A[0]+rect2w, A[1], 0])
    intersection = intersectSegments2D(B, H, E, F, True)
    if intersection.size == 0:
        print "ERROR: Unable to find point 'I' cutting rectangle into another rectangle"
        return
    I = Point3D(intersection[0], intersection[1], 0)
    intersection = intersectSegments2D(B, H, C, D, True)
    if intersection.size == 0:
        print "ERROR: Unable to find point 'J' cutting rectangle into another rectangle"
        return
    J = np.array([intersection[0], intersection[1], 0])
    for P in [A, B, C, D, E, F, G, H, I, J]:
        cutPoints.append(P)

    #STEP 2: Cut out the big triangle and the little triangle
    vHI = I - H
    seg1P = I + vHI
    seg2P = H
    cutWithSegment(cuts, seg1P, seg2P)

    vGE = E - G
    seg1P = np.array([E[0]+vGE[0], E[1], 0])
    seg2P = G
    cutWithSegment(cuts, seg1P, seg2P)
    
    #STEP 3: Move big triangle and little triangle
    insideBigTriangle = []
    getCutsInsideTriangle(cuts, B, C, J, insideBigTriangle)
    translation = np.eye(4)
    translation[0, 3] = rect2w-rect1w
    translation[1, e] = -(J[1]-D[1])
    NTransforms = len(cuts[0].transforms)
    for poly in insideBigTriangle:
        poly.transform = translation.dot(poly.transform)
        poly.transforms = [translation] + poly.transforms
        NTransforms = len(poly.transforms)
        poly.transformPoints(translation)
    #Add the identity transform to all points that weren't inside the big triangle
    for cut in cuts:
        if len(cut.transforms) < NTransforms:
            cut.transforms = [np.eye(4)] + cut.transforms

    insideLittleTriangle = []
    getCutsInsideTriangle(cuts, E, B, I, insideLittleTriangle)
    translation = np.eye(4)
    translation[0:2, 3] = (J-B)[0:2]

    NTransforms = len(cuts[0].transforms)
    for poly in insideLittleTriangle:
        poly.transform = translation.dot(poly.transform)
        poly.transforms = [translation] + poly.transforms
        NTransforms = len(cuts[0].transforms)
        poly.transformPoints(translation)
    #Add the identity transform to all points that weren't inside the little triangle
    for cut in cuts:
        if len(cut.transforms) < NTransforms:
            cut.transforms = [np.eye(4)] + cut.transforms

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
    vBD = (vBC.dot(vBA)/np.sum(vBA**2))*vBA
    D = PB + vBD
    vDA = PA - D
    vDE = 0.5*vDA
    E = D + vDE
    dummy = E + vBC
    intersect = intersectSegments2D(PA, PC, E, dummy, False)
    if intersect.size == 0:
        print "ERROR: Could not find point 'F' in initial triangle cut"
        return
    F = np.array(intersect)
    dummy = E - vBC
    intersect = intersectSegments2D(PA, PB, E, dummy, False)
    if intersect.size == 0:
        print "ERROR: Could not find point 'G' in initial triangle cut"
        return
    G = np.array(intersect)
    #Now add the four initial polygons
    #Upper left triangle
    poly1 = PolygonCut()
    for P in [F, PA, E]:
        poly1.points.append(P)
    #Rotate 180 degrees around point F
    poly1.transform = np.array([-1, 0, 0, 2*F[0],
                                0, -1, 0, 2*F[1],
                                0, 0, 1, 0,
                                0, 0, 0, 1])
    poly1.transform = np.reshape(poly1.transform, (4, 4))
    poly1.transforms = [poly1.transform]
    poly1.transformPoints(poly1.transform)
    cuts.append(poly1)

    #Upper right triangle
    poly2 = PolygonCut()
    for P in [E, PA, G]:
        poly2.points.append(P)
    #Rotate 180 degrees around point G
    poly2.transform = np.array([-1, 0, 0, 2*G[0],
                                0, -1, 0, 2*G[1],
                                0, 0, 1, 0,
                                0, 0, 0, 1])
    poly2.transform = np.reshape(poly2.transform, (4, 4))
    poly2.transforms = [poly2.transform]
    poly2.transformPoints(poly2.transform)
    cuts.append(poly2)

    #Lower trapezoid
    poly3 = PolygonCut()
    for P in [PC, F, G, PB]:
        poly3.points.append(P)
        poly3.transforms = [np.eye(4)] #Doesn't move on the first cut
    cuts.append(poly3)

    #STEP 3: Rotate the new rectangle so that it is axis-aligned, and
    #translate it so that its lower left corner matches the lower left
    #corner of the target polygon
    vPCD = D - PC
    vDE = E - D
    vPCD = vPCD/np.sqrt(np.sum(vPCD**2)) #Normalize
    vDE = vDE/np.sqrt(np.sum(vDE**2)) #Normalize
    R = np.array( [  vPCD[0], vDE[0], 0, 0,
                    vPCD[1], vDE[1], 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1 ])
    R = np.reshape(R, (4, 4))
    R = R.T #This matrix will rotate the proper amount around the origin
    translation = np.eye(4)
    translation[0:2, 3] = cuts2.points[0][0:2]
    translationInv = np.eye(4)
    translationInv[0:2, 3] = -translation[0:2, 3]
    rotateAxisAlign = (translation.dot(R)).dot(translationInv)
    #Finally, translate the rectangle so that its bottom left corner aligns
    #with the bottom left corner of the target rectangle
    offset = rectCorner - cuts[2].points[0]
    translation = np.eye(4)
    translation[0:2, 3] = offset[0:2]
    rotateAxisAlign = translation.dot(rotateAxisAlign)
    #Now apply this matrix to every polygonal cut to align the rectangle
    #with the target rectangle at the lower left corner
    for cut in cuts:
        cut.transform = rotateAxisAlign.dot(cut.transform)
        cut.transforms = [rotateAxisAlign] + cut.transforms
        cut.transformPoints(rotateAxisAlign)

    #Determine the width and height of the rectangle, and cut it in half
    #until width < 2*height AND height < 2*width
    widthVector = cuts[2].points[3] - cuts[2].points[0]
    width = np.sqrt(np.sum(widthVector**2))
    heightVector = cuts[0].points[1] - cuts[0].points[2]
    height = np.sqrt(np.sum(heightVector**2))
    
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

def drawPolygon2DTk(canvas, poly, outlineColor = "#0000FF", fillColor = "#FFFFFF", drawVertices = True):
    if len(poly) < 3:
        for i in range(0, len(poly)):
            [P1, P2] = [poly[i], poly[(i+1)%len(poly)]]
            canvas.create_line(P1[0], P1[1], P2[0], P2[1], fill=outlineColor)
    else:
        coordsList = []
        for P in poly:
            coordsList.append(P[0])
            coordsList.append(P[1])
        canvas.create_polygon(coordsList, outline = outlineColor, fill = fillColor)
    if drawVertices:
        for P in poly:
            canvas.create_oval(P[0]-4, P[1]+4, P[0]+4, P[1]-4, fill=outlineColor)

#t is the interpolation parameter that says how far along to slide the cut
def drawPolygonCut(canvas, height, poly, t, color = "#0000FF"):
    t = (t+t**0.5)/2 #Have it slow down towards the end
    cosA = poly.transform[0, 0]
    sinA = poly.transform[1, 0]
    dx = poly.transform[0, 3]*t
    dy = poly.transform[1, 3]*t
    A = math.atan2(sinA, cosA)*t
    [cosA, sinA] = [math.cos(A), math.sin(A)]
    trans1 = np.array( [ cosA, -sinA, 0, dx,
                        sinA, cosA, 0, dy,
                        0, 0, 1, 0,
                        0, 0, 0, 1] )
    trans1 = np.reshape(trans1, (4, 4))
    trans = trans1
    trans = trans1.dot(np.linalg.inv(poly.transform))
    points = [trans*P for P in poly.points]
    drawPolygon2DTk(canvas, points, color, drawVertices = False)

def drawPolygonCuts(canvas, height, polygonCuts, t, color = "#0000FF"):
    for poly in polygonCuts:
        drawPolygonCut(canvas, height, poly, t, color)

def getAreaOfCuts(cuts):
    return sum([cut.getArea() for cut in cuts])

if __name__ == '__main__':
    cuts = []
    cutPoints = []
    A = np.array([-1, 0, 0])
    B = np.array([0, 1, 0])
    C = np.array([1, 0, 0])
    rectCorner = np.array([0, 0, 0])
    rectw = 1
    recth = 1
    cutTriangleIntoRectangle(cuts, A, B, C, rectCorner, rectw, recth, cutPoints, True)
