import visualID as vID
from visualID import  fg, hl, bg
import numpy as np

from ase.atoms import Atoms
from ase.io import write
from ase.geometry import cellpar_to_cell
from ase.spacegroup import get_spacegroup
import os

from scipy import linalg

from pyNanoMatBuilder import data

#######################################################################
######################################## time
from datetime import datetime
import datetime, time
class timer:

    def __init__(self):
        _start_time   = None
        _end_time     = None
        _chrono_start = None
        _chrono_stop  = None

    # Return human delay like 01:14:28 543ms
    # delay can be timedelta or seconds
    def hdelay_ms(self,delay):
        if type(delay) is not datetime.timedelta:
            delay=datetime.timedelta(seconds=delay)
        sec = delay.total_seconds()
        hh = sec // 3600
        mm = (sec // 60) - (hh * 60)
        ss = sec - hh*3600 - mm*60
        ms = (sec - int(sec))*1000
        return f'{hh:02.0f}:{mm:02.0f}:{ss:02.0f} {ms:03.0f}ms'
    
    def chrono_start(self):
        global _chrono_start, _chrono_stop
        _chrono_start=time.time()
    
    # return delay in seconds or in humain format
    def chrono_stop(self, hdelay=False):
        global _chrono_start, _chrono_stop
        _chrono_stop = time.time()
        sec = _chrono_stop - _chrono_start
        if hdelay : return self.hdelay_ms(sec)
        return sec

    def hdelay_ms(self,delay):
        if type(delay) is not datetime.timedelta:
            delay=datetime.timedelta(seconds=delay)
        sec = delay.total_seconds()
        hh = sec // 3600
        mm = (sec // 60) - (hh * 60)
        ss = sec - hh*3600 - mm*60
        ms = (sec - int(sec))*1000
        return f'{hh:02.0f}:{mm:02.0f}:{ss:02.0f} {ms:03.0f}ms'
    
    def chrono_show(self):
        print(f'{fg.BLUE}Duration : {self.hdelay_ms(time.time() - _chrono_start)}{fg.OFF}')

#######################################################################
######################################## ase unitcells and symmetry analyzis
def returnUnitcellData(system):
    '''
    system is an instance of the Crystal class
    '''
    system.ucUnitcell = system.cif.cell.cellpar()
    system.ucV = cellpar_to_cell(system.ucUnitcell)
    system.ucBL = system.cif.cell.get_bravais_lattice()
    system.ucSG = get_spacegroup(system.cif,symprec=system.aseSymPrec)
    system.ucVolume = system.cif.cell.volume
    system.ucReciprocal = np.array(system.cif.cell.reciprocal())
    system.ucFormula = system.cif.get_chemical_formula()
    system.G = G(system)
    system.Gstar = Gstar(system)

def print_ase_unitcell(system):
    '''
    system is an instance of the Crystal class
    '''
    unitcell = system.ucUnitcell
    bl = system.ucBL
    formula = system.ucFormula
    volume = system.ucVolume
    sg = system.ucSG
    print(f"Bravais lattice: {bl}")
    print(f"Chemical formula: {formula}")
    print(f"Crystal family = {bl.crystal_family} (lattice system = {bl.lattice_system})")
    print(f"Name = {bl.longname} (Pearson symbol = {bl.pearson_symbol})")
    print(f"Variant names = {bl.variant_names}")
    print()
    print(f"From ase: space group number = {sg.no}      Hermann-Mauguin symbol for the space group = {sg.symbol}")
    print()
    print(f"a: {unitcell[0]:.3f} Å, b: {unitcell[1]:.3f} Å, c: {unitcell[2]:.3f} Å. (c/a = {unitcell[2]/unitcell[0]:.3f})")
    print(f"α: {unitcell[3]:.3f} °, β: {unitcell[4]:.3f} °, γ: {unitcell[5]:.3f} °")
    print()
    print(f"Volume: {volume:.3f} Å3")

def listCifsOfTheDatabase():
    from ase import io
    import pathlib
    import glob
    from ase.spacegroup import get_spacegroup
    import re
    
    path2cifFolder = os.path.join(pNMB_location(),'cif_database')
    print(f"path to cif database = {path2cifFolder}")
    
    sgITField = "_space_group_IT_number"
    sgHMField = "_symmetry_space_group_name_H-M"
    
    class Crystal:
        pass
        
    for cif in glob.glob(f'{path2cifFolder}/*.cif'):
        path2cifFile = pathlib.Path(cif)
        cifName = pathlib.Path(*path2cifFile.parts[-1:])
        vID.centertxt(f"{cifName}",size=14,weight='bold')
        cifContent = io.read(cif)
        cifFile =  open(cif, 'r')
        cifFileLines = cifFile.readlines()
        re_sgIT = re.compile(sgITField)
        re_sgHM = re.compile(sgHMField)
        for line in cifFileLines: 
            if re_sgIT.search(line): sgIT = ' '.join(line.split()[1:])
            if re_sgHM.search(line): sgHM = ' '.join(line.split()[1:])
        cifFile.close()
        c = Crystal()
        c.cif = cifContent
        c.aseSymPrec = 1e-4
        returnUnitcellData(c)
        print_ase_unitcell(c)
        color="vID.fg.RED"
        print()
        if int(sgIT) == c.ucSG.no:
            print(f"{vID.fg.GREEN}Symmetry in the cif file = {sgIT}   {sgHM}{vID.hl.BOLD} in agreement with the ase symmetry analyzis{vID.fg.OFF}")
        else:
            print(f"{vID.fg.RED}Symmetry in the cif file = {sgIT}   {sgHM}{vID.hl.BOLD} disagrees with the ase symmetry analyzis{vID.fg.OFF}")

#######################################################################
######################################## coupling with pymatgen in order to find the symmetry
def MolSym(aseobject: Atoms,
           getEquivalentAtoms: bool=False):
    import pymatgen.core as pmg
    from pymatgen.io.ase import AseAtomsAdaptor as aaa
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer
    
    chrono = timer(); chrono.chrono_start()
    vID.centertxt("Symmetry analysis",bgc='#007a7a',size='14',weight='bold')
    print(f"Currently using the PointGroupAnalyzer class of pymatgen\nThe analyzis can take a while for large compounds")
    print()
    pmgmol = pmg.Molecule(aseobject.get_chemical_symbols(),aseobject.get_positions())
    pga = PointGroupAnalyzer(pmgmol, tolerance=0.6, eigen_tolerance=0.02, matrix_tolerance=0.2)
    pg = pga.get_pointgroup()
    print(f"Point Group: {pg}")
    print(f"Rotational Symmetry Number = {pga.get_rotational_symmetry_number()}")
    chrono.chrono_stop(hdelay=False); chrono.chrono_show()
    if getEquivalentAtoms:
        return pg, pga.get_equivalent_atoms()
    else:
        return pg, []

#######################################################################
######################################## Folder pathways
def ciflist(dbFolder=data.pNMBvar.dbFolder):
    import os
    path2cif = os.path.join(pNMB_location(),dbFolder)
    print(os.listdir(path2cif))
        
def pNMB_location():
    import pyNanoMatBuilder, pathlib, os
    path = pathlib.Path(pyNanoMatBuilder.__file__)
    return pathlib.Path(*path.parts[0:-2])

#######################################################################
######################################## Vectors and distances
def RAB(coord,a,b):
    import numpy as np
    """calculate the interatomic distance between two atoms a and b"""
    r = np.sqrt((coord[a][0]-coord[b][0])**2 + (coord[a][1]-coord[b][1])**2 + (coord[a][2]-coord[b][2])**2)
    return r

def Rbetween2Points(p1,p2):
    import numpy as np
    r = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
    return r

def vector(coord,a,b):
    import numpy as np
    v = [coord[b][0]-coord[a][0],coord[b][1]-coord[a][1],coord[b][2]-coord[a][2]]
    v = np.array(v)
    return v

def vectorBetween2Points(p1,p2):
    import numpy as np
    v = [p2[0]-p1[0],p2[1]-p1[1],p2[2]-p1[2]]
    v = np.array(v)
    return v

def coord2xyz(coord):
    import numpy as np
    x = np.array(coord)[:,0]
    y = np.array(coord)[:,1]
    z = np.array(coord)[:,2]
    return x,y,z

def vertex(x, y, z, scale):
    import numpy as np
    """ Return vertex coordinates fixed to the unit sphere """
    length = np.sqrt(x**2 + y**2 + z**2)
    return [(i * scale) / length for i in (x,y,z)]

def vertexScaled(x, y, z, scale):
    import numpy as np
    """ Return vertex coordinates multiplied by the scale factor """
    return [i * scale for i in (x,y,z)]

    
def RadiusSphereAfterV(V):
    import numpy as np
    return (3*V/(4*np.pi))**(1/3)

def centerOfGravity(c: np.ndarray,
                    select=None):
    import numpy as np
    if select is None:
        select = np.array((range(len(c))))
    nselect = len(select)
    xg = 0
    yg = 0
    zg = 0
    for at in select:
        xg += c[at][0]
        yg += c[at][1]
        zg += c[at][2]
    cog = [xg/nselect, yg/nselect, zg/nselect]
    return np.array(cog)

def center2cog(c: np.ndarray):
    import numpy as np
    cog = centerOfGravity(c)
    c2cog = []
    for at in c:
        at = at - cog
        c2cog.append(at)
    return np.array(c2cog)

def normOfV(V):
    '''
    returns the norm of a vector V, [V0,V1,V2]
    '''
    import numpy as np
    return np.sqrt(V[0]**2+V[1]**2+V[2]**2)

def normV(V):
    '''
    normalizes V and returns the result as an array
    '''
    import numpy as np
    N = normOfV(V)
    return np.array([V[0]/N,V[1]/N,V[2]/N])

def centerToVertices(coordVertices: np.ndarray,
                     cog: np.ndarray):
    '''
    returns the vector and distance between the center of gravity (cog) and each vertex of a polyhedron
    input:
        - coordVertices = coordinates of the vertices ((nvertices,3) numpy array)
        - cog = center of gravity of the NP
    returns:
        - (cog-nvertices)x3 coordinates of the vectors (np.array)
        - nvertices-cog distances (np.array)
    '''
    import numpy as np
    directions = []
    distances = []
    for v in coordVertices:
        distances.append(Rbetween2Points(v,cog))
        directions.append(v - cog)
    return np.array(directions), np.array(distances)

#######################################################################
######################################## Fill edges and facets
def MakeFaceCoord(Rnn,f,coord,nAtomsOnFaces,coordFaceAt):
    import numpy as np
    # the idea here is to interpolate between edge atoms of two relevant edges
    # (for example two opposite edges of a squared face)
    # be careful of the vectors orientation of the edges!
    if (len(f) == 3):  #triangular facet
        edge1 = [f[1],f[0]]
        edge2 = [f[1],f[2]]
    if (len(f) == 4):  #square facet 0-1-2-3-4-0
        edge1 = [f[3],f[0]]
        edge2 = [f[2],f[1]]
    if (len(f) == 5):  #pentagonal facet #not working
        edge1 = [f[1],f[0]]
        edge2 = [f[1],f[2]]
    if (len(f) == 6):  #hexagonal facet #not working
        edge1 = [f[0],f[1]]
        edge2 = [f[5],f[4]]
    nAtomsOnEdges = int((RAB(coord,f[1],f[0])+1e-6)/Rnn) - 1
    nIntervalsE = nAtomsOnEdges + 1
    for n in range(nAtomsOnEdges):
        CoordAtomOnEdge1 = coord[edge1[0]]+vector(coord,edge1[0],edge1[1])*(n+1) / nIntervalsE
        CoordAtomOnEdge2 = coord[edge2[0]]+vector(coord,edge2[0],edge2[1])*(n+1) / nIntervalsE
        distBetween2EdgeAtoms = Rbetween2Points(CoordAtomOnEdge1,CoordAtomOnEdge2)
        nAtomsBetweenEdges = int((distBetween2EdgeAtoms+1e-6)/Rnn) - 1
        nIntervalsF = nAtomsBetweenEdges + 1
        for m in range(nAtomsBetweenEdges):
            coordFaceAt.append(CoordAtomOnEdge1 + vectorBetween2Points(CoordAtomOnEdge1,CoordAtomOnEdge2)*(m+1) / nIntervalsF)
            nAtomsOnFaces += 1
    return nAtomsOnFaces,coordFaceAt

#######################################################################
######################################## Momenta of inertia
def moi(model: Atoms):
    import numpy as np
    vID.centertxt("Moments of inertia",bgc='#007a7a',size='14',weight='bold')
    model.moi = model.get_moments_of_inertia() # in amu*angstrom**2
    print(f"Moments of inertia = {model.moi[0]:.2f} {model.moi[1]:.2f} {model.moi[2]:.2f} amu.Å2")
    model.masses = model.get_masses()
    model.M = model.masses.sum()
    model.moiM = model.moi/model.M
    print(f"Moments of inertia / M = {model.moiM[0]:.2f} {model.moiM[1]:.2f} {model.moiM[2]:.2f} amu.Å2")
    model.dim = 2*np.sqrt(5*model.moiM)
    print(f"Size of the ellipsoid = {model.dim[0]*0.1:.2f} {model.dim[1]*0.1:.2f} {model.dim[2]*0.1:.2f} nm")

#######################################################################
######################################## Geometry optimization
def optimizeEMT(model: Atoms, pathway="./coords/model", fthreshold=0.05):
    from varname import nameof, argname
    import numpy as np
    from ase.io import write
    from ase import Atoms
    from ase.calculators.emt import EMT
    chrono = timer(); chrono.chrono_start()
    vID.centerTitle(f"ase EMT calculator & Quasi Newton algorithm for geometry optimization")
    model.calc=EMT()
    model.get_potential_energy()
    from ase.optimize import QuasiNewton
    dyn = QuasiNewton(model, trajectory=pathway+'.opt')
    dyn.run(fmax=fthreshold)
    write(pathway+"_opt.xyz", model)
    print(f"{fg.BLUE}Optimization steps saved in {pathway+'_.opt'} (binary file){fg.OFF}")
    print(f"{fg.RED}Optimized geometry saved in {pathway+'_opt.xyz'}{fg.OFF}")
    chrono.chrono_stop(hdelay=False); chrono.chrono_show()
    return model

#######################################################################
######################################## Planes & Directions
def planeFittingLSF(coords: np.float64,
                    printErrors: bool=False,
                    printEq: bool=True):
    '''
    least-square fitting of the equation of a plane ux + vy + wz + h = 0
    that passes as close as possible to a set of 3D points
    - input
        - coords: numpy array with shape (N,3) that contains the 3 coordinates for each of the N points
        - printErrors: if True, prints the absolute error between the actual z coordinate of each points
        and the corresponding z-value calculated from the plane equation. The residue is also printed 
    - returns a numpy array [u v w h]
    '''
    import numpy as np
    from numpy import linalg as la
    xs = coords[:,0]
    ys = coords[:,1]
    zs = coords[:,2]
    nat = len(xs)
    select=[i for i in range(nat)]
    cog = centerOfGravity(coords, select)
    mat = np.zeros((3,3))
    for i in range(nat):
        mat[0,0]=mat[0,0]+(xs[i]-cog[0])**2
        mat[1,1]=mat[1,1]+(ys[i]-cog[1])**2
        mat[2,2]=mat[2,2]+(zs[i]-cog[2])**2
        mat[0,1]=mat[0,1]+(xs[i]-cog[0])*(ys[i]-cog[1])
        mat[0,2]=mat[0,2]+(xs[i]-cog[0])*(zs[i]-cog[2])
        mat[1,2]=mat[1,2]+(ys[i]-cog[1])*(zs[i]-cog[2])    
    mat[1,0]=mat[0,1]
    mat[2,0]=mat[0,2]
    mat[2,1]=mat[1,2]
    eigenvalues, eigenvectors = la.eig(mat)
    # the eigenvector associated with the smallest eigenvalue is the vector normal to the plane
    # print(eigenvalues)
    # print(eigenvectors)
    indexMinEigenvalue = np.argmin(eigenvalues)
    # print(indexMinEigenvalue)
    # print(la.norm(eigenvectors[:,indexMinEigenvalue]))
    u,v,w = eigenvectors[:,indexMinEigenvalue]
    h = -u*cog[0] - v*cog[1] - w*cog[2]
    if printEq:
        print(f"bare solution: {u:.5f} x + {v:.5f} y + {w:.5f} z + {h:.5f} = 0")
        # print("     or")
        # print(f"bare solution: {-u/w:.5f} x + {-v/w:.5f} y + {-h/w:.5f} = z")
    tmp = coords.copy
    ones = np.ones(nat)
    tmp = np.column_stack((coords,ones))
    fit = np.array([u,v,w,h])
    fit = fit.reshape(4,1)
    errors = tmp@fit
    residual = la.norm(errors)
    if printErrors:
        print(f"errors:")
        print(errors)
        print(f"residual: {residual}")
    return np.array([u,v,w,h])

def convertuvwh2hkld(plane: np.float64,
                     prthkld: bool=True):
    '''
    converts an ux + vy + wz + h = 0 equation of a plane, where u, v, w and h can be real numbers as 
    an hx + ky + lz + d = 0 equation, where h, k, and l are all integer numbers
    - input:
        - [u v w h] numpy array 
        - prthkld: does print the result by default 
    - returns an [h k l d] numpy array 
    '''
    import numpy as np
    from fractions import Fraction
    # apply only on non-zero uvw values
    planeEff = []
    acc = 1e-8
    for x in plane[0:3]: # u,v,w only
        if (np.abs(x) >= acc):
            planeEff.append(x)
    planeEff = np.array(planeEff)
    F = np.array([Fraction(x).limit_denominator() for x in np.abs(planeEff)]) # don't change the signe of hkl
    Fmin = np.min(F)
    hkld = plane/Fmin
    
    if prthkld:
        print(f"hkl solution: {hkld[0]:.5f} x + {hkld[1]:.5f} y + {hkld[2]:.5f} z + {hkld[3]:.5f} = 0")
        # print("     or")
        # print(f"hkl solution: {-hkld[0]/hkld[2]:.5f} x + {-hkld[1]/hkld[2]:.5f} y + {-hkld[3]/hkld[2]:.5f} = z")
    return hkld

def hklPlaneFitting(coords: np.float64,
                    printEq: bool=True,
                    printErrors: bool=False):
    '''
    Context: finding the Miller indices of a plane, if relevant
    Consists in a least-square fitting of the equation of a plane hx + ky + lz + d = 0
    that passes as close as possible to a set of 3D points
    - input
        - coords: numpy array with shape (N,3) that contains the 3 coordinates for each of the N points
        - printErrors: if True, prints the absolute error between the actual z coordinate of each points
        and the corresponding z-value calculated from the plane equation. The residue is also printed
    - returns a numpy array [h k l d], where h, k, and l are as close as possible to integers
    '''
    plane = planeFittingLSF(coords,printErrors)
    plane = convertuvwh2hkld(plane, printEq)
    return plane

def shortestPoint2PlaneVectorDistance(plane:np.ndarray,
                                      point:np.ndarray):
    '''
    returns the shortest distance, d, from a point X0 to a plane p (projection of X0 on p = P), as well as the PX0 vector 
    - input:
        - plane = [u v w h] definition of the p plane (numpy array)
        - point = [x0 y0 z0] coordinates of the X0 point (numpy array)
    returns the PX0 vector and ||PX0||
    '''
    t = (plane[3] + np.dot(plane[0:3],point))/(plane[0]**2+plane[1]**2+plane[2]**2)
    v = -t*plane[0:3]
    d = np.sqrt(v[0]**2+v[1]**2+v[2]**2)
    return v, d

def Pt2planeSignedDistance(plane,point):
    '''
    returns the orthogonal distance of a given point X0 to the plane p in a metric space (projection of X0 on p = P), 
    with the sign determined by whether or not X0 is in the interior of p with respect to the center of gravity [0 0 0]
    - input:
        - plane = [u v w h] definition of the P plane (numpy array)
        - point = [x0 y0 z0] coordinates of the X0 point (numpy array)
    returns the signed modulus ±||PX0||
    '''
    sd = (plane[3] + np.dot(plane[0:3],point))/np.sqrt(plane[0]**2+plane[1]**2+plane[2]**2)
    return sd

def planeAtVertices(coordVertices: np.ndarray,
                    cog: np.ndarray):
    '''
    returns the equation of the plane defined by vectors between the center of gravity (cog) and each vertex of a polyhedron
    and that is located at the vertex
    input:
        - coordVertices = coordinates of the vertices ((nvertices,3) numpy array)
        - cog = center of gravity of the NP
    returns the (cog-nvertices)x3 coordinates of the plane (np.array)
    '''
    import numpy as np
    planes = []
    for vx in coordVertices:
        vector = vx - cog
        d = -np.dot(vx,vector)
        vector = np.append(vector,d)
        planes.append(vector)
    return np.array(planes)

def planeAtPoint(plane: np.ndarray,
                 P0: np.ndarray):
    '''
    given a former [a,b,c,d] plane as input, d is recalculated so that the plane passes through P0,
    a known point on the plane
    input:
        -plane: numpy array [a b c d]
        -P0: numpy array with P0 coordinates [x0 y0 z0]
    returns [a b c -(ax0+by0+cz0)]
    '''
    d = np.dot(plane[0:3],P0)
    planeAtP = plane.copy()
    planeAtP[3] = -d
    return planeAtP

def normalizePlane(p):
    import numpy as np
    '''
    normalizes the [a,b,c,d] coordinates of a plane
    - input: plane [a,b,c,d]
    returns [a/norm,b/norm,c/norm,d/norm] where norm=dsqrt(a**2+b**2+c**2)
    '''
    return p/normOfV(p[0:3])

def point2PlaneDistance(point: np.float64,
                              plane: np.float64):
    import numpy as np
    from numpy.linalg import norm
    distance = abs(np.dot(point,plane[0:3]) + plane[3]) / norm(plane[0:3])
    return distance

def AngleBetweenVV(lineDV,planeNV):
    '''
    returns the angle, in degrees, between two vectors
    '''
    import numpy as np
    ldv = np.array(lineDV)
    pnv = np.array(planeNV)
    numerator = np.dot(ldv,pnv)
    denominator = normOfV(lineDV)*normOfV(planeNV)
    if denominator == 0:
        alpha = np.NaN
    else:
        alpha = 180*np.arccos(np.clip(numerator/denominator,-1,1))/np.pi
    return alpha

def signedAngleBetweenVV(v1,v2):
    '''
    returns, between [0°,360°] the signed angle, in degrees, between two vectors
    '''
    import numpy as np
    n = [1,1,1]
    cosTh = np.dot(v1,v2)
    sinTh = np.dot(np.cross(v1,v2),n)
    angle = np.rad2deg(np.arctan2(sinTh,cosTh))
    if angle >= 0: return angle
    else: return 360+angle
    return angle

def normal2MillerPlane(Crystal,MillerIndexes,printN=True):
    '''
    returns the normal direction to the plane defined by h,k,l Miller indices is defined as [n1 n2 n3] = (hkl) x G*,
    where G* is the reciprocal metric tensor (G* = G-1)

    the convertuvwh2hkld() function applied here converts real plane indexes to integers
    '''
    normal = MillerIndexes@Crystal.Gstar
    normal = np.append(normal,0.0) #trick because convertuvwh2hkld() converts (u v w h) planes
    normalI = convertuvwh2hkld(normal,False)[0:3]
    if printN: 
        print(f"Normal to the ({MillerIndexes[0]:2} {MillerIndexes[1]:2} {MillerIndexes[2]:2}) user-defined plane > [{normal[0]: .3e} {normal[1]: .3e} {normal[2]: .3e}]",\
              f" = [{normalI[0]: .2f} {normalI[1]: .2f} {normalI[2]: .2f}]")
    return normalI

def isPlaneParrallel2Line(v1,v2,tol=1e-5):
    '''
    returns a boolean
    a line direction vector and a plane are parallel if the |angle| between the line and the normal vector of the plane is 90°
    '''
    return np.abs(np.abs(AngleBetweenVV(v1,v2)) - 90) < tol or np.abs(np.abs(AngleBetweenVV(v1,v2)) - 270) < tol 

def isPlaneOrthogonal2Line(v1,v2,tol=1e-5):
    '''
    returns a boolean
    a line direction vector and a plane are orthogonal if the |angle| between the line and the normal vector of the plane is 0° or 180°
    '''
    return np.abs(AngleBetweenVV(v1,v2)) < tol or np.abs(np.abs(AngleBetweenVV(v1,v2)) - 180) < tol

def areDirectionsOrthogonal(v1,v2,tol=1e-6):
    '''
    returns a boolean
    lines are orthogonal if the |angle| between their direction vector is 90°
    '''
    return np.abs(np.abs(AngleBetweenVV(v1,v2)) - 90) < tol or np.abs(np.abs(AngleBetweenVV(v1,v2)) - 270) < tol

def areDirectionsParallel(v1,v2,tol=1e-6):
    '''
    returns a boolean
    lines are orthogonal if the |angle| between their direction vector is 0° or 180°
    '''
    return np.abs(AngleBetweenVV(v1,v2)) < tol or np.abs(np.abs(AngleBetweenVV(v1,v2)) - 180) < tol

def returnPlaneParallel2Line(V, shift=[1,0,0], debug = False):
    '''
    returns the [a b c] parameters for a plane parallel to the input direction
    (d must be found separately)

    algorithm:
        - choose any arbitrary vector not parallel to V[i,j,k] such as V[i+1,j,k]
        - calculate the vector perpendicular to both of these, i.e. the cross product
        - this is the normal to the plane, i.e. you directly obtain the equation of the plane ax+by+cz+d = 0, d being indeterminate
        (to find d, it would be necessary to provide an (x0,y0,z0) point that does not belong to the line, hence d = -ax0-by0-cz0)
    '''
    arbV = np.array(V.copy())
    arbV = arbV + np.array(shift)
    plane = np.cross(V,arbV)
    if areDirectionsParallel(V,arbV): sys.exit(f"Error in returnPlaneParallel2Line(): plane {V} is parallel to {arbV}. "\
                                               f"Are you sure of your data?\n(this function wants to return an equation for a plane parallel to the direction V = {V}.\n"\
                                               f" Play with the shift variable - current problematic value = {shift})")
    if debug: print(areDirectionsParallel(V,arbV), V, arbV, "cross product = ",plane)
    return plane

def planeRotation(Crystal, refPlane, rotAxis, nRot=6, debug=False):
    '''
    returns an array with planes obtained by rotating the reference plane around the input axis
    - input: 
        - Crystal = Crystal object
        - refPlane = plane to rotate
        - nRot = rotation angle is 360°/nRot
        - rotAxis = rotation axis
        - debug = normalized planes are printed
    '''
    pRef = np.array([refPlane])
    aRot = np.array([rotAxis])
    msg = f"Projection of the ({pRef[0][0]: .2f} {pRef[0][1]: .2f} {pRef[0][2]: .2f}) reference truncation plane around the "\
          f"[{rotAxis[0]: .2f}  {rotAxis[1]: .2f}  {rotAxis[2]: .2f}] axis, after projection in the cartesian coordinate system"
    vID.centertxt(msg,bgc='#cbcbcb',size='12',fgc='b',weight='bold')
    pRefCart = lattice_cart(Crystal,pRef,True,True)
    rotAxisCart = lattice_cart(Crystal,aRot,True,True)
    msg = f"{nRot}th order rotation around {rotAxisCart[0][0]: .2f} {rotAxisCart[0][1]: .2f} {rotAxisCart[0][2]: .2f}"\
          f"of the ({pRefCart[0][0]: .2f} {pRefCart[0][1]: .2f} {pRefCart[0][2]: .2f}) truncation plane"
    vID.centertxt(msg,bgc='#cbcbcb',size='12',fgc='b',weight='bold')
    planesCart = []
    for i in range(0,nRot):
        angle = i*360/nRot
        # print("rot around z    = ",RotationMol(pRefCart[0],angle,'z'))
        x = rotationMolAroundAxis(pRefCart[0],angle,rotAxisCart[0])
        # print("rot around axis = ",x)
        planesCart.append(x)
    if (debug): print(np.array(planesCart))
    vID.centertxt(f"Just for your knowledge: indexes of the {nRot} normal directions to the truncation planes after projection to the {Crystal.cif.cell.get_bravais_lattice()} unitcell",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
    planesHCP = lattice_cart(Crystal,np.array(planesCart),False,True)
    if (debug):
        vID.centertxt(f"Normalized HCP planes",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        for i,p in enumerate(planesHCP):
            print(i,normV(p))
        print()
        vID.centertxt(f"Normalized cartesian planes",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
    return np.array(planesCart)

def alignV1WithV2_returnR(v1,v2=np.array([0, 0, 1])):
    """
    returns the rotation matrix [rMat] between two vectors using scipy, so that rMat@v1 is aligned with v2
    uses the align_vectors function of scipy.spatial.transform, align_vectors(a, b)
    in which when a single vector is given for a and b, the shortest distance rotation that aligns b to a is returned
    - input:
        - two vectors given as numpy arrays, in the order v1, v2
    - returns the (3,3) rotation matrix that aligns v1 with v2
    """
    from scipy.spatial.transform import Rotation
    import numpy as np
    import sys
    v1 = np.reshape(v1, (1, -1))
    v2 = np.reshape(v2, (1, -1))
    rMat = Rotation.align_vectors(v2, v1)
    rMat = rMat[0].as_matrix()
    v1_rot = rMat@v1[0]
    aligned = np.allclose(v1_rot / np.linalg.norm(v1_rot), v2 / np.linalg.norm(v2))
    if not aligned: sys.exit(f"Was unable to align {v1} with {v2}. Check your data")
    return rMat

def rotateMoltoAlignItWithAxis(coords,axis,targetAxis=np.array([0, 0, 1])):
    '''
    returns coordinates after rotation made to align axis with targetAxis
    - input:
        - coords = natoms x 3 numpy array
        - axis, targetAxis = directions given under the form [u,v,w]
    - returns a (natoms,3) numpy array
    '''
    import numpy as np
    if isinstance(axis, list):
        axis = np.array(axis)
    if isinstance(targetAxis, list):
        targetAxis = np.array(targetAxis)
    rMat = alignV1WithV2_returnR(axis,targetAxis)
    return np.array(rMat@coords.transpose()).transpose()

#######################################################################
######################################## cut above planes
def calculateTruncationPlanesFromVertices(planes, cutFromVertexAt, nAtomsPerEdge, debug=False):
    n = int(round(1/cutFromVertexAt))
    print(f"factor = {cutFromVertexAt:.3f} ▶ {round(nAtomsPerEdge/n)} layer(s) will be removed, starting from each vertex")

    trPlanes = []
    for p in planes:
        pNormalized =normalizePlane(p.copy())
        pNormalized[3] =  pNormalized[3] - pNormalized[3]*cutFromVertexAt
        trPlanes.append(pNormalized)
        if (debug):
            print("normalized original plane = ",normalizePlane(p))
            print("cut plane = ",pNormalized,"... norm = ",normOfV(pNormalized[0:3]))
            print("signed distance between original plane and origin = ",Pt2planeSignedDistance(p,[0,0,0]))
            print("signed distance between cut plane and origin = ",Pt2planeSignedDistance(pNormalized,[0,0,0]))
            print("pcut/pRef = ",Pt2planeSignedDistance(pNormalized,[0,0,0])\
                                /Pt2planeSignedDistance(p,[0,0,0]))
        print(f"Will remove atoms just above plane "\
              f"{pNormalized[0]:.2f} {pNormalized[1]:.2f} {pNormalized[2]:.2f} d:{pNormalized[3]:.3f}")
    return np.array(trPlanes)    

def truncateAboveEachPlane(planes: np.ndarray,
                           coords,
                           debug=False,
                           delAbove: bool = True):
    '''
    - input: 
        - planes = numpy array with all [u v w d] plane definitions
        - coords = (N,3) numpy array will all coordinates
        - delAbove = if True (default) delete atoms that lie above the planes + eps = 1e-4. Delete atoms below the
                     planes otherwise (use with precaution, could return no atoms as a function of their definition)
    - returns the indexes of the atoms that are above each input planes
    '''

    delAtoms = []

    eps =1e-3
    for p in planes:
        for i,c in enumerate(coords):
            signedDistance = Pt2planeSignedDistance(p,c)
            if delAbove and signedDistance > eps:
                delAtoms.append(i)
            elif not delAbove and signedDistance < eps:
                delAtoms.append(i)
        # print(keptAtoms)
        if debug:
            for a in delAtoms:
                print(f"@{a+1}",end=',')
            print("",end='\n')
    delAtoms = np.array(delAtoms)
    delAtoms = np.unique(delAtoms)
    # if (debug): plot3D()
    return delAtoms

def truncateAbovePlanes(planes: np.ndarray,
                        coords: np.ndarray,
                        allP: bool=False,
                        delAbove: bool = True,
                        debug: bool=False,
                        eps: float=1e-3):
    '''
    - input: 
        - planes = numpy array with all [u v w d] plane definitions
        - coords = (N,3) numpy array will all coordinates
        - allP = deleted atoms must lie above ALL planes (default: False)
        - delAbove = if True (default) delete atoms that lie above the planes + eps = 1e-3 (default). Delete atoms below the
                     planes otherwise (use with precaution, could return no atoms as a function of their definition)
        - debug: if True (default is False) print atoms that match the allP/delAbove planes conditions
        - eps: atom-to-plane signed distance threshold (default 1e-3)
    - returns an N-dimension boolean array that tells which atoms above each input planes (allP = False) 
      or above all input planes at the same time (allP=True) (opposite if delAbove is False)
    '''

    import numpy as np
    vID.centertxt(f"Plane truncation (all planes condition: {allP}, delete above planes: {delAbove}, initial number of atoms = {len(coords)})",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
    
    if allP:
        delAtoms = np.ones(len(coords), dtype=bool)
    else:
        delAtoms = np.zeros(len(coords), dtype=bool)
    nOfDeletedAtoms = 0
    for p in planes:
        if allP:
            delAtomsP = np.ones(len(coords), dtype=bool)
        else:
            delAtomsP = np.zeros(len(coords), dtype=bool)
        for i,c in enumerate(coords):
            signedDistance = Pt2planeSignedDistance(p,c)
            if delAbove and allP:
                delAtoms[i] = delAtoms[i] and signedDistance > eps
            elif delAbove and not allP:
                delAtoms[i] = delAtoms[i] or signedDistance > eps
                delAtomsP[i] = signedDistance > eps
            elif not delAbove and allP:
                delAtoms[i] = delAtoms[i] and signedDistance < -eps
            elif not delAbove and not allP:
                delAtoms[i] = delAtoms[i] or signedDistance < -eps
                delAtomsP[i] = signedDistance < -eps
        nOfDeletedAtoms = np.count_nonzero(delAtoms) - nOfDeletedAtoms
        nOfDeletedAtomsP = np.count_nonzero(delAtomsP)
        if debug and not allP:
            print(f"- plane", [f"{x: .2f}" for x in p],f"> {nOfDeletedAtomsP} atoms deleted")
            for i,a in enumerate(delAtomsP):
                if a: print(f"@{i+1}",end=',')
            print("",end='\n')
        if debug and allP:
            print("allP")
            print(f"- plane", [f"{x: .2f}" for x in p],f"> {nOfDeletedAtoms} atoms deleted")
            for i,a in enumerate(delAtoms):
                if a: print(f"@{i+1}",end=',')
            print("",end='\n')
    delAtoms = np.array(delAtoms)
    if delAbove:
        print(f"{len(coords)-np.count_nonzero(delAtoms)} atoms lie below the plane(s)")
    else:
        print(f"{np.count_nonzero(delAtoms)} atoms lie below the plane(s)")
    return delAtoms

def returnPointsThatLieInPlanes(planes: np.ndarray,
                                coords: np.ndarray,
                                debug: bool=False,
                                threshold: float=1e-3):
    import numpy as np
    vID.centertxt(f"Find all points that lie in the given planes",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
    AtomsInPlane = np.zeros(len(coords), dtype=bool)
    for p in planes:
        for i,c in enumerate(coords):
            signedDistance = Pt2planeSignedDistance(p,c)
            AtomsInPlane[i] = AtomsInPlane[i] or np.abs(signedDistance) < threshold
        nOfAtomsInPlane = np.count_nonzero(AtomsInPlane)
        if debug:
            print(f"- plane", [f"{x: .2f}" for x in p],f"> {nOfAtomsInPlane} atoms lie in the planes")
            for i,a in enumerate(delAtoms):
                if a: print(f"@{i+1}",end=',')
            print("",end='\n')
    AtomsInPlane = np.array(AtomsInPlane)
    print(f"{np.count_nonzero(AtomsInPlane)} atoms lie in the plane(s)")
    return AtomsInPlane

def deleteElementsOfAList(t,
                          list2Delete: bool):
    '''
    returns a new list
    input:
        - t: list or table
        - list2Delete = list of booleans. list2Delete[i] = True ==> t[i] is deleted 
    '''
    import numpy as np
    if len(t) != len(list2Delete): sys.exit("the input list and the array of booleans must have the same dimension. Check your code")
    if type(t) == list: 
        tloc = np.array(t.copy())
    else:
        tloc = t.copy()
    tloc = np.delete(tloc,list2Delete,axis=0)
    return list(tloc)

#######################################################################
######################################## coupling with Jmol & DebyeCalculator
def saveCoords_DrawJmol(asemol,prefix,scriptJ=""):
    path2Jmol = '/usr/local/src/jmol-14.32.50'
    fxyz = "./figs/"+prefix+".xyz"
    write(fxyz, asemol)
    # jmolscript = 'cpk 0; wireframe 0.025; script "./figs/script-facettes-3-4RuLight.spt"; facettes34rulight; draw * opaque; color atoms black; set zShadePower 1; set specularPower 80; pngon; write image 1024 1024 ./figs/'
    jmolscript = scriptJ + '; frank off; cpk 0; wireframe 0.05; script "./figs/script-facettes-345PtLight.spt"; facettes345ptlight; draw * opaque;'
    jmolscript = jmolscript + 'set specularPower 80; set antialiasdisplay; set background [xf1f2f3]; set zShade ON;set zShadePower 1; write image pngt 1024 1024 ./figs/'
    jmolcmd="java -Xmx512m -jar " + path2Jmol + "/JmolData.jar " + fxyz + " -ij '" + jmolscript + prefix + ".png'" + " >/dev/null "
    print(jmolcmd)
    os.system(jmolcmd)

def DrawJmol(mol,prefix,scriptJ=""):
    path2Jmol = '/usr/local/src/jmol-14.32.50'
    fxyz = "./figs/"+mol+".xyz"
    jmolscript = scriptJ + '; frank off; set specularPower 80; set antialiasdisplay; set background [xf1f2f3]; set zShade ON;set zShadePower 1; write image pngt 1024 1024 ./figs/'
    jmolcmd="java -Xmx512m -jar " + path2Jmol + "/JmolData.jar " + fxyz + " -ij '" + jmolscript + prefix + ".png'" + " >/dev/null "
    print(jmolcmd)
    os.system(jmolcmd)

def writexyz(filename,atoms):
    '''
    simple xyz writing, with atomic symbols/x/y/z and no other information sometimes misunderstood by some utilities, such as DebyeCalculator
    '''
    element_array=atoms.get_chemical_symbols()
    # extract composition in dict form
    composition={}
    for element in element_array:
        if element in composition:
            composition[element]+=1
        else:
            composition[element]=1
       
    coord=atoms.get_positions()
    natoms=len(element_array)  
    line2write='%d \n'%natoms
    line2write+='%s\n'%str(composition)
    for i in range(natoms):
        line2write+='%s'%str(element_array[i])+'\t %.8f'%float(coord[i,0])+'\t %.8f'%float(coord[i,1])+'\t %.8f'%float(coord[i,2])+'\n'
    with open(filename,'w') as file:
        file.write(line2write)

def defWulffShapeForJMol(Crystal):
    from scipy.spatial import HalfspaceIntersection
    from scipy.spatial import ConvexHull
    import networkx as nx
    import scipy as sp
    
    cog = Crystal.cog
    feasible_point = np.array([0., 0., 0.])
    hs = HalfspaceIntersection(Crystal.trPlanes, feasible_point)
    vertices = hs.intersections + cog
    hull = ConvexHull(vertices)
    faces = hull.simplices
    neighbours = hull.neighbors
    
    def sortVCW(V,C):
        '''
        sort the vertices of a planar polygon clockwise
        - input:
            - V = list of vertices of a given polygon
            - C = coordinates of all vertices
        '''
        coords = []
        for v in V: coords.append(C[v])
        cog = np.mean(coords,axis=0)
        radialV = coords-cog
        angle = []
        V = list(V)
        for i in range(len(radialV)):
            angle.append(signedAngleBetweenVV(radialV[0],radialV[i]))
        ind = np.argsort(angle)
        Vs = np.array(list(V))
        return Vs[ind]
    
    def isCoplanar(p1,p2,tolAngle=0.1):
        angle = AngleBetweenVV(p1[0:3],p2[0:3])
        return (abs(angle) < tolAngle or abs(angle-180) <= tolAngle)
        
    def reduceFaces(F,coordsVertices):
    
        flatten = lambda l: [item for sublist in l for item in sublist]
    
        # create a graph in which nodes represent triangles
        # nodes are connected if the corresponding triangles are adjacent and coplanar
        G = nx.Graph()
        G.add_nodes_from(range(len(F)))
        pList = []
        for i,f in enumerate(F):
            planeDef = []
            for v in f:
                planeDef.append(coordsVertices[v])
            planeDef = np.array(planeDef)
            pList.append(planeFittingLSF(planeDef,printEq=False))
    
        for i,p1 in enumerate(pList):
            for n in neighbours[i]:
                p2 = pList[n]
                if isCoplanar(p1,p2):
                    G.add_edge(i,n)
        components = list(nx.connected_components(G))
        simplified = [set(flatten(F[index] for index in component)) for component in components]
    
        return simplified
    vID.centertxt("copy/paste this command line in jmol",bgc='#007a7a',size='14',weight='bold')
        
    new_faces = reduceFaces(faces,vertices)
    new_facesS = []
    for i,nf in enumerate(new_faces):
        new_facesS.append(sortVCW(nf,vertices).tolist())
    cmd = ""
    for i,nf in enumerate(new_facesS):
        cmd += "draw facet" + str(i) + " polygon "
        cmd += '['
        for at in nf:
            cmd+=f"{{{vertices[at][0]:.4f},{vertices[at][1]:.4f},{vertices[at][2]:.4f}}},"
        cmd+="]; "
    cmd += "color $facet* translucent 70 [x828282]" 
    cmde = ""
    index = 0
    for nf in new_facesS:
        nfcycle = np.append(nf,nf[0])
        for i, at in enumerate(nfcycle[:-1]):
            cmde += "draw line" + str(index) + " ["
            cmde += f"{{{vertices[at][0]:.4f},{vertices[at][1]:.4f},{vertices[at][2]:.4f}}},"
            cmde += f"{{{vertices[nfcycle[i+1]][0]:.4f},{vertices[nfcycle[i+1]][1]:.4f},{vertices[nfcycle[i+1]][2]:.4f}}},"
            cmde += "] width 0.2; "
            index += 1
    cmde += "color $line* [x828282]; "
    cmd = cmde + cmd
    return cmd

#######################################################################
######################################## coordination numbers
def calculateCN(coords,Rmax):
    '''
    returns the coordination number of each atom, where CN is calculated after threshold Rmax
    - input:
        - coords: numpy array with shape (N,3) that contains the 3 coordinates for each of the N points
        - Rmax: threshold to calculate CN
    returns an array that contains CN for each atom
    '''
    CN = np.zeros(len(coords))
    for i,ci in enumerate(coords):
        for j in range(0,i):
            Rij = Rbetween2Points(ci,coords[j])
            if Rij <= Rmax:
                CN[i]+=1
                CN[j]+=1
    return CN

def delAtomsWithCN(coords: np.ndarray,
                   Rmax: np.float64,
                   targetCN: int=12):
    '''
    identifies atoms that have a coordination number (CN) == targetCN and returns them in an array
    - input:
        - coords: numpy array with shape (N,3) that contains the 3 coordinates for each of the N points
        - CN: array of integers with the coordination number of each atom
        - targetCN (default=12)
    returns an array that contains the indexes of atoms with CN == targetCN
    '''
    CN = calculateCN(coords,Rmax)
    tabDelAtoms = []
    for i,cn in enumerate(CN):
        if cn == targetCN: tabDelAtoms.append(i)
    tabDelAtoms = np.array(tabDelAtoms)
    return tabDelAtoms

def findNeighbours(coords,Rmax):
    '''
    for all atoms i, returns the list of all atoms j within an arbitrarily determined cutoff distance Rmax from atom i
    - input:
        - coords = numpy array with the N-atoms cartesian coordinates
        - Rmax = cutoff distance
    - returns:
        - list of lists (len(list[i]) = number of nearest neighbours of atom i)
    '''
    vID.centertxt(f"Building a table of nearest neighbours",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
    chrono = timer(); chrono.chrono_start()
    nAtoms = len(coords)
    nn = [ [] for _ in range(nAtoms)]
    for i in range(nAtoms):
        for j in range(i):
            if RAB(coords,i,j) < Rmax:
                nn[i].append(j)
                nn[j].append(i)
    chrono.chrono_stop(hdelay=False); chrono.chrono_show()
    return nn

def printNeighbours(nn):
    '''
    prints the list of nearest neighbours of each atom
    - input:
        - nn = nearest neighbours given as a list of list - such as the nn provided by the neighbours() function
    '''
    for i,nni in enumerate(nn):
        print(f"Atom {i:6} has {len(nni):2} NN: {nni}")

#######################################################################
######################################## symmetry
def reflection(plane,points,doItForAtomsThatLieInTheReflectionPlane=False):
    '''
    applies a mirror-image symmetry operation of an array of points w.r.t. a plane of symmetry
    - input:
        - plane = [u,v,w,d] parameters that define a plane
        - point = (N, 3) array of points
        - doItForAtomsThatLieInTheReflectionPlane = slef-explanatory
    - returns an (N, 3) array of mirror-image points
    '''
    import numpy as np
    pr = []
    eps = 1.e-4
    for p in points:
        vp2plane, dp2plane = shortestPoint2PlaneVectorDistance(plane,p)
        if dp2plane >= eps and not doItForAtomsThatLieInTheReflectionPlane: # otherwise the point belongs to the reflection plane
            # print(dp2plane, vp2plane, p)
            ptmp = p+2*vp2plane
            pr.append(ptmp)
        else:
            ptmp = p+2*vp2plane
            pr.append(ptmp)
    return np.array(pr)

#######################################################################
######################################## rotation
def Rx(a):
    ''' returns the R/x rotation matrix'''
    import math as m
    return np.matrix([[ 1, 0           , 0   ],
                    [ 0, m.cos(a),-m.sin(a)],
                    [ 0, m.sin(a), m.cos(a)]])
  
def Ry(a):
    ''' returns the R/y rotation matrix'''
    import math as m
    return np.matrix([[ m.cos(a), 0, m.sin(a)],
                   [ 0           , 1, 0           ],
                   [-m.sin(a), 0, m.cos(a)]])
  
def Rz(a):
    ''' returns the R/z rotation matrix'''
    import math as m
    return np.matrix([[ m.cos(a), -m.sin(a), 0 ],
                   [ m.sin(a), m.cos(a) , 0 ],
                   [ 0           , 0            , 1 ]])

def EulerRotationMatrix(gamma,beta,alpha,order="zyx"):
    """
    - input:
        - gamma: Rot/x (°)
        - beta: Rot/y (°)
        - alpha: Rot/z (°)
        - if (order="zyx"): returns Rz(alpha) * Ry(beta) * Rx(gamma)
    returns a 3x3 Euler matrix as a numpy array
    """
    import math as m
    #REuler is a 3x3 matrix
    R = 1.
    gammarad = gamma*m.pi/180
    betarad = beta*m.pi/180
    alpharad = alpha*m.pi/180
    for i in range(3):
        if order[i] == "x":
            R = R*Rx(gammarad)
        if order[i] == "y":
            R = R*Ry(betarad)
        if order[i] == "z":
            R = R*Rz(alpharad)
    return R

def RotationMol(coords, angle, axis="z"):
    import math as m
    angler = angle*m.pi/180
    if axis == 'x':
        R =  np.array(Rx(angler)@coords.transpose())
    elif axis == 'y':
        R =  np.array(Ry(angler)@coords.transpose())
    elif axis == 'z':
        R =  np.array(Rz(angler)@coords.transpose())
    return R[0]

def EulerRotationMol(coords, gamma, beta, alpha, order="zyx"):
    return np.array(EulerRotationMatrix(gamma,beta,alpha,order)@coords.transpose()).transpose()

def RotationMatrixFromAxisAngle(u,angle):
    import math as m
    a = angle*m.pi/180
    ux = u[0]
    uy = u[1]
    uz = u[2]
    return np.array([[m.cos(a)+ux**2*(1-m.cos(a))   , ux*uy*(1-m.cos(a))-uz*m.sin(a), ux*uz*(1-m.cos(a))+uy*m.sin(a)],
                      [uy*ux*(1-m.cos(a))+uz*m.sin(a), m.cos(a)+uy**2*(1-m.cos(a))   , uy*uz*(1-m.cos(a))-ux*m.sin(a)],
                      [uz*ux*(1-m.cos(a))-uy*m.sin(a), uz*uy*(1-m.cos(a))+ux*m.sin(a), m.cos(a)+uz**2*(1-m.cos(a))   ]])

def rotationMolAroundAxis(coords, angle, axis):
    '''
    returns coordinates after rotation by a given angle around an [u,v,w] axis
    - input:
        - coords = natoms x 3 numpy array
        - angle = angle of rotation
        - axis = directions given under the form [u,v,w]
    - returns a numpy array
    '''
    normalizedAxis = normV(axis)
    return np.array(RotationMatrixFromAxisAngle(normalizedAxis,angle)@coords.transpose()).transpose()

#######################################################################
######################################## magic numbers
def magicNumbers(cluster,i):
    match cluster:
        case 'regfccOh':
            mn = np.round((2/3)*i**3 + 2*i**2 + (7/3)*i + 1)
            return mn
        case 'regIco':
            mn = (10*i**3 + 11*i)//3 + 5*i**2 + 1
            return mn
        case 'regfccTd':
            mn = np.round(i**3/6 + i**2 + 11*i/6 + 1)
            return mn
        case 'regDD':
            mn = 10*i**3 + 15*i**2 + 7*i + 1
            return mn
        case 'fccCube':
            mn = 4*i**3 + 6*i*2 + 3*i + 1
            return mn
        case 'bccCube':
            mn = 2*i**3 + 3*i*2 + 3*i
            return mn
        case 'fccCubo':
            mn = np.round((10*i**3 + 11*i)/3 + 5*i**2 + 1)
            return mn
        case 'fccTrOh':
            mn = np.round(16*i**3 + 15*i**2 + 6*i + 1)
            return mn
        case 'fccTrCube':
            mn = np.round(4*i**3 + 6*i**2 + 3*i - 7)
            return mn
        case 'bccrDD':
            mn = 4*i**3 + 6*i**2 + 4*i + 1
            return mn
        case 'fccdrDD':
            mn = 8*i**3 + 6*i**2 + 2*i + 3
            return mn
        case 'pbpy':
            mn = 5*i**3/6 + 5*i**2/2 + 8*i/3 + 1
            return mn
        case _:
            sys.exit(f"The {cluster} cluster is unknown")

#######################################################################
######################################## Bravais
def interPlanarSpacing(plane: np.ndarray,
                       unitcell: np.ndarray,
                       CrystalSystem: str='CUB'):
    '''
    - input:
        - plane = numpy array that the contains the [h k l d] parameters of the plane of equation
                hx + ky +lz + d = 0
        - unitcell = numpy array with [a b c alpha beta gamma]
        - CrystalSystem = name of the crystal system, string among:
          ['CUB', 'HEX', 'TRH', 'TET', 'ORC', 'MCL', 'TRI'] = cubic, hexagonal, trigonal-rhombohedral, tetragonal, orthorombic, monoclinic, tricilinic
    returns the interplanar spacing (float value)
    '''
    import sys
    h = plane[0]
    k = plane[1]
    l = plane[2]
    a = unitcell[0]
    match CrystalSystem.upper():
        case 'CUB':
            d2 = a**2 / (h**2+k**2+l**2)
        case 'HEX':
            c = unitcell[2]
            d2inv = (4/3)*(h**2 + k**2 + h*k)/a**2 + l**2/c**2
            d2 = 1/d2inv
        case 'TRH':
            alpha = (np.pi/180) * unitcell[3]
            d2inv = ((h**2 + k**2 + l**2)*np.sin(alpha)**2 + 2*(h*k + k*l + h*l)*(np.cos(alpha)**2-np.cos(alpha)))/(a**2*(1-3*np.cos(alpha)**2+2*np.cos(alpha)**3))
            d2 = 1/d2inv
        case 'TET':
            c = unitcell[2]
            d2inv = (h**2+k**2)/a**2 + l**2/c**2
            d2 = 1/d2inv    
        case 'ORC':
            b = unitcell[1]
            c = unitcell[2]
            d2inv = h**2/a**2 + k**2/b**2 + l**2/c**2
            d2 = 1/d2inv    
        case 'MCL':
            b = unitcell[1]
            c = unitcell[2]
            # beta = np.pi - (np.pi/180) * unitcell[4]
            # d2inv = h**2/(a**2*np.sin(beta)) + k**2/b**2 + l**2/(c**2*np.sin(beta)) + 2*h*l*np.cos(beta)/(a*c*np.sin(beta)**2)
            beta = (np.pi/180) * unitcell[4]
            d2inv = ((h/a)**2 + (k*np.sin(beta)/b)**2 + (l/c)**2 - 2*h*l*np.cos(beta)/(a*c))/np.sin(beta)**2
            d2 = 1/d2inv    
        case 'TRI':
            b = unitcell[1]
            c = unitcell[2]
            alpha = (np.pi/180) * unitcell[3]
            beta = (np.pi/180) * unitcell[4]
            gamma = (np.pi/180) * unitcell[5]
            V = (a*b*c) * np.sqrt(1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 + 2*np.cos(alpha)*np.cos(beta)*np.cos(gamma))
            astar = b*c*np.sin(alpha)/V
            bstar = a*c*np.sin(beta)/V
            cstar = a*b*np.sin(gamma)/V
            cosalphastar = (np.cos(gamma)*np.cos(beta) - np.cos(alpha))/(np.sin(gamma)*np.sin(beta))
            cosbetastar  = (np.cos(alpha)*np.cos(gamma) - np.cos(beta))/(np.sin(alpha)*np.sin(gamma))
            cosgammastar = (np.cos(beta)*np.cos(alpha) - np.cos(gamma))/(np.sin(beta)*np.sin(alpha))
            d2inv = (h*astar)**2 + (k*bstar)**2 + (l*cstar)**2 + 2*k*l*bstar*cstar*cosalphastar\
                                                               + 2*l*h*cstar*astar*cosbetastar\
                                                               + 2*h*k*astar*bstar*cosgammastar
            d2 = 1/d2inv    
        case _:
            sys.exit(f"{CrystalSystem} crystal system is unknown. Check your data.\n"\
                      "Or do not try to calculate interplanar distances on this system with interPlanarSpacing()")
    d = np.sqrt(d2)
    return d

def lattice_cart(Crystal,vectors,Bravais2cart=True,printV=False):
    '''
    - input:
        - Crystal = Crystal object
        - vectors = vectors to project from the Bravais basis to the cartesian coordinate system (if Bravais2cart is True)
                         or to project from the cartesian coordinate system to the Bravais basis  (if Bravais2cart is False)
        - printV = boolean (default: False), prints the resulting vectors if True
    - returns an array of projected vectors
    '''
    import numpy as np
    unitcell = Crystal.ucUnitcell
    Vuc = Crystal.ucV
    if Bravais2cart:
        Vproj = (vectors@Vuc)
        B = 'B'
        E = 'C'
    else:
        VucInv = np.linalg.inv(Vuc)
        Vproj = (vectors@VucInv)
        B = 'C'
        E = 'B'
    if printV:
        print("Change of basis")
        for i,V in enumerate(vectors):
            Bstr = f"{V[0]: .2f} {V[1]: .2f} {V[2]: .2f}"
            Vp = Vproj[i]
            Estr = f"{Vp[0]: .2f} {Vp[1]: .2f} {Vp[2]: .2f}"
            print(f"({Bstr}){B} > ({Estr}){E}")
    return Vproj 

def G(Crystal):
    a = Crystal.ucUnitcell[0]
    b = Crystal.ucUnitcell[1]
    c = Crystal.ucUnitcell[2]
    alpha = Crystal.ucUnitcell[3]*np.pi/180.
    beta = Crystal.ucUnitcell[4]*np.pi/180.
    gamma = Crystal.ucUnitcell[5]*np.pi/180.
    ca = np.cos(alpha)
    cb = np.cos(beta)
    cg = np.cos(gamma)
    GG = np.array([[      a**2, a * b * cg, a * c * cb],
                  [a * b * cg,       b**2, b * c * ca],
                  [a * c * cb, b * c * ca,       c**2]])
    return GG

def Gstar(Crystal):
    Gmat = G(Crystal)
    return linalg.inv(Gmat)

#######################################################################
######################################## Misc for plots
def imageNameWithPathway(imgName):
    path2image= os.path.join(pNMB_location(),'figs')
    imgNameWithPathway = os.path.join(path2image,imgName)
    return imgNameWithPathway

def plotImageInPropFunction(imageFile):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    image = mpimg.imread(imageFile)
    plt.figure(figsize=(2, 10))
    plt.imshow(image,interpolation='nearest')
    plt.axis('off')
    plt.show()

#######################################################################
######################################## Core/surface identification / Convex Hull analysis
def coreSurface(coords,threshold):
    from scipy.spatial import ConvexHull
    vID.centertxt("Core/Surface analyzis",bgc='#007a7a',size='14',weight='bold')
    chrono = timer(); chrono.chrono_start()
    vID.centertxt(f"Convex Hull analyzis",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
    hull = ConvexHull(coords)
    print("Found:")
    print(f"  - {len(hull.vertices)} vertices")
    print(f"  - {len(hull.simplices)} simplices")
    chrono.chrono_stop(hdelay=False); chrono.chrono_show()
    chrono = timer(); chrono.chrono_start()
    surfaceAtoms= returnPointsThatLieInPlanes(hull.equations,coords,threshold=threshold)
    chrono.chrono_stop(hdelay=False); chrono.chrono_show()
    return [hull.vertices,hull.simplices,hull.neighbors,hull.equations],surfaceAtoms
