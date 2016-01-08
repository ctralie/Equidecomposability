import numpy as np

EPS = 1e-11

## Utility Functions for 2D geometry
#This function assumes the polygon is convex
def getPolygonArea(verts):
    if len(verts) < 3:
        return 0.0
    v1 = verts[1, :] - verts[0, :]
    v2 = verts[1, :] - verts[0, :]
    area = 0.0
    #Triangulate and add area of each triangle
    for i in range(2, len(verts)):
        v1 = v2
        v2 = verts[i, :] - verts[0, :]
        area = area + 0.5*np.sqrt(np.sum(np.cross(v1, v2)**2))
    return area

#Return the cosine of the angle between P1 and P2 with respect
#to "Vertex" as their common, shared vertex
def COSBetween(Vertex, P1, P2):
    V1 = P1 - Vertex
    V2 = P2 - Vertex
    dot = V1.dot(V2)
    magProduct = np.sqrt(np.sum(V1*V2)*np.sum(V2*V2))
    if (magProduct < EPS):
        return 0
    return float(dot) / float(magProduct)

#Find the intersection of two lines segments in a numerically stable
#way by looking at them parametrically
def intersectSegments2D(A, B, C, D, countEndpoints = True):
    denomDet = (D[0]-C[0])*(A[1]-B[1]) - (D[1]-C[1])*(A[0]-B[0])
    if (denomDet == 0): #Segments are parallel
        return np.array([])
    num_t = (A[0]-C[0])*(A[1]-B[1]) - (A[1]-C[1])*(A[0]-B[0]);
    num_s = (D[0]-C[0])*(A[1]-C[1]) - (D[1]-C[1])*(A[0]-C[0]);
    t = float(num_t) / float(denomDet)
    s = float(num_s) / float(denomDet)
    if (s < 0 or s > 1):
        return np.array([]) #Intersection not within the bounds of segment 1
    if (t < 0 or t > 1):
        return np.array([]) #Intersection not within the bounds of segment 2

    #Don't count intersections that occur at the endpoints of both segments
    #if the user so chooses
    if ((t == 0 or t == 1) and (s == 0 or s == 1) and (not countEndpoints)):
        return np.array([])

    ret = np.array([A[0], A[1], 0])
    ret[0] = ret[0] + (B[0]-A[0])*s;
    ret[1] = ret[1] + (B[1]-A[1])*s;
    return ret

def PointsEqual2D(P1, P2):
    if (abs(P1[0]-P2[0]) < EPS and abs(P1[1]-P2[1]) < EPS):
        return True
    #print "P1 = %s, P2 = %s, abs(P1[0]-P2[0]) = %g, abs(P1[1] - P2[1]) = %g"%(P1, P2, abs(P1[0]-P2[0]), abs(P1[1]-P2[1]))
    return False

def CCW2D(A, B, C):
    det = B[0]*C[1] - B[1]*C[0] - A[0]*C[1] + A[1]*C[0] + A[0]*B[1] - A[1]*B[0]
    if (det > EPS):
        return -1
    elif (det < -EPS):
        return 1
    #Are the points all equal?
    if (PointsEqual2D(A, B) and PointsEqual2D(B, C)):
        return 0
    if (PointsEqual2D(A, B)):
        return 2
    #Is C in the closure of A and B?
    #Vectors must be in opposite directions or one of the vectors
    #must be zero (C is on one of the endpoints of A and B)
    vAC = C - A
    vBC = C - B
    vAC[2] = 0
    vBC[2] = 0
    if (vAC.dot(vBC) < EPS):
        return 0;#This fires for C in the closure of A and B (including endpoints)
    vBA = A - B
    vBA[2] = 0
    #C to the left of AB
    if (vBA.dot(vBC) > EPS):
        return -2
    #C to the right of AB
    else:
        return 2

#Helper function for "pointInsideTriangle()"
def pointOnRightSideOfEdge2D(A, B, P, CLOSENESS_EPS = EPS):
    CCWABP = CCW2D(A, B, P)
    if CCWABP != 1 and CCWABP != 0:
        if CCWABP == -1:
            #Do a perpendicular projection onto the segment
            #to make sure it isn't a super close call
            vAB = B - A
            vAP = P - A
            proj = vAB*(1-vAB.dot(vAP)/np.sum(vAP**2))
            if np.sum(proj**2) < CLOSENESS_EPS:
                return True
            return False
        #Check endpoints
        elif CCWABP == -2:
            vPA = A - P
            if np.sum(vPA**2) < CLOSENESS_EPS:
                return True
            return False
        elif CCWABP == 2:
            vPB = B - P
            if np.sum(vPA**2) < CLOSENESS_EPS:
                return True
            return False
        else:
            print "ERROR in pointOnRightSideOfEdge2D: Shouldn't have gotten here"
    return True

#This is a helper function for "getCutsInsideTriangle()" in the Equidecomposability project
#and also a helper function for ear cutting triangulation
def pointInsideTriangle2D(A, B, C, P, CLOSENESS_EPS = EPS):
    [AP, BP, CP] = [A, B, C]
    if CCW2D(A, B, C) == -1:
        [AP, BP, CP] = [C, B, A]
    isInside = pointOnRightSideOfEdge2D(AP, BP, P, CLOSENESS_EPS)
    isInside = isInside and (pointOnRightSideOfEdge2D(BP, CP, P, CLOSENESS_EPS))
    isInside = isInside and (pointOnRightSideOfEdge2D(CP, AP, P, CLOSENESS_EPS))
    return isInside
