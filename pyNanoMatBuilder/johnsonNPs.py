import visualID as vID
from visualID import  fg, hl, bg

import sys
import numpy as np
import pyNanoMatBuilder.utils as pNMBu
import ase
from ase.build import bulk, make_supercell, cut
from ase.visualize import view
from ase.cluster.cubic import FaceCenteredCubic

from pyNanoMatBuilder import platonicNPs as pNP

###########################################################################################################
class fcctbp:
    nFaces = 6
    nEdges = 9
    nVertices = 5
    edgeLengthF = 2
    heightOfPyramidF = edgeLengthF * np.sqrt(2/3)
    heightOfBiPyramidF = 2*heightOfPyramidF

    def __init__(self,
                 element: str='Au',
                 Rnn: float=2.7,
                 nLayerTd: int=1,
                 postAnalyzis=True,
                 aseView=True,
                 thresholdCoreSurface = 1.,
                 skipSymmetryAnalyzis = False,
                 noOutput = False,
                 calcPropOnly = False,
                ):
        self.element = element
        self.Rnn = Rnn
        self.nLayerTd = int(nLayerTd)
        self.Tdprop = pNP.regfccTd(self.element,self.Rnn,self.nLayerTd,noOutput=True,calcPropOnly=True)
        self.nLayer = 2*self.nLayerTd - 1
        self.nAtoms = 0
        self.nAtomsPerLayer = []
        self.interLayerDistance = self.Tdprop.interLayerDistance()
        self.nAtomsPerEdge = self.nLayerTd+1
        self.cog = np.array([0., 0., 0.])
        self.fveAngle = self.Tdprop.fveAngle
        self.fefAngle = self.Tdprop.fefAngle
        self.vcvAngle = self.Tdprop.vcvAngle
        self.heightOfBiPyramid = 2*self.Tdprop.heightOfPyramid()
        self.imageFile = pNMBu.imageNameWithPathway("tbp-C.png")

        if not noOutput: vID.centerTitle(f"fcc trigonal bipyramid with {nLayerTd} shells per pyramid")

        if not noOutput: self.prop()
        if not calcPropOnly:
            self.coords(noOutput)
            if aseView: view(self.NP)
            if postAnalyzis:
                self.propPostMake(skipSymmetryAnalyzis,thresholdCoreSurface)
                if aseView: view(self.NPcs)

    def __str__(self):
        return(f"Regular fcc double tetrahedron of {self.element} with {self.nLayer+1} layer(s) and Rnn = {self.Rnn}")

    def edgeLength(self):
        return self.Tdprop.edgeLength()

    def coords(self,noOutput):
        if not noOutput: vID.centertxt("Generation of coordinates",bgc='#007a7a',size='14',weight='bold')
        chrono = pNMBu.timer(); chrono.chrono_start()
        vID.centertxt("Generation of the coordinates of the tetrahedron",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        Td = pNP.regfccTd(self.element,self.Rnn,self.nLayerTd+1,postAnalyzis=False,noOutput=True)
        aseTd = Td.NP
        self.NP0 = aseTd.copy()
        c = aseTd.get_positions()
        vID.centertxt("Applying mirror reflection w.r.t. facet defined by atoms (0,1,2) ",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        mirrorPlane = [0,1,2]
        cMirrorPlane = []
        for at in mirrorPlane:
            cMirrorPlane.append(aseTd.get_positions()[at])
        cMirrorPlane=np.array(cMirrorPlane)
        mirrorPlane = pNMBu.planeFittingLSF(cMirrorPlane, False, False)
        pNMBu.convertuvwh2hkld(mirrorPlane)
        cr = pNMBu.reflection(mirrorPlane,aseTd.get_positions())
        nMirroredAtoms = len(cr)
        aseMirror = ase.Atoms(self.element*nMirroredAtoms, positions=cr)
        aseObject = aseTd + aseMirror
        c = pNMBu.center2cog(aseObject.get_positions())
        nAtoms = aseObject.get_global_number_of_atoms()
        aseObject = ase.Atoms(self.element*nAtoms, positions=c)
        print(f"Total number of atoms = {nAtoms}")
        chrono.chrono_stop(hdelay=False); chrono.chrono_show()
        self.NP = aseObject
        
    def prop(self):
        print(self)
        pNMBu.plotImageInPropFunction(self.imageFile)
        print("element = ",self.element)
        print("number of vertices = ",self.nVertices)
        print("number of edges = ",self.nEdges)
        print("number of faces = ",self.nFaces)
        print(f"nearest neighbour distance = {self.Rnn:.2f} Å")
        print(f"edge length = {self.edgeLength()*0.1:.2f} nm")
        print(f"inter-layer distance = {self.interLayerDistance:.2f} Å")
        print(f"number of atoms per edge = {self.nAtomsPerEdge}")
        print(f"height of bipyramid = {self.heightOfBiPyramid*0.1:.2f} nm")
        print(f"area = {6*self.Tdprop.area()/4*1e-2:.1f} nm2")
        print(f"volume = {2*self.Tdprop.volume()*1e-3:.1f} nm3")
        print(f"face-vertex-edge angle = {self.Tdprop.fveAngle:.1f}°")
        print(f"face-edge-face (dihedral) angle = {self.Tdprop.fefAngle:.1f}°")
        print(f"vertex-center-vertex (tetrahedral bond) angle = {self.Tdprop.vcvAngle:.1f}°")
        # print("number of atoms per layer = ",self.Tdprop.nAtomsPerLayerAnalytic())
        # print("total number of atoms = ",self.nAtomsAnalytic())
        print("Dual polyhedron: triangular prism")
        print("Indexes of vertex atoms = [0,1,2,3] by construction")
        print(f"coordinates of the center of gravity = {self.cog}")

    def propPostMake(self,skipSymmetryAnalyzis,thresholdCoreSurface):
        pNMBu.moi(self.NP)
        if not skipSymmetryAnalyzis: pNMBu.MolSym(self.NP)
        [self.vertices,self.simplices,self.neighbors,self.equations],surfaceAtoms =\
            pNMBu.coreSurface(self.NP.get_positions(),thresholdCoreSurface)
        self.NPcs = self.NP.copy()
        self.NPcs.numbers[np.invert(surfaceAtoms)] = 102 #Nobelium, because it has a nice pinkish color in jmol

###########################################################################################################
class epbpyM:
    nFaces3 = 10
    nFaces4 = 5
    nEdgesPbpy = 15
    nEdgesEpbpy = 20
    nVerticesPbpy = 7
    nVerticesEpbpy = 12
    phi = (1 + np.sqrt(5))/2 # golden ratio
    edgeLengthF = 1
    heightOfPyramidF = np.sqrt((5-np.sqrt(5))/10)*edgeLengthF
    interCompactPlanesF = (1+np.sqrt(5))/4
    magicFactorF = 2*heightOfPyramidF/edgeLengthF

    def __init__(self,
                 element: str='Au',
                 Rnn: float=2.7,
                 sizeP: int=1,
                 sizeE: int=0,
                 Marks: int=0,
                 postAnalyzis=True,
                 aseView=True,
                 thresholdCoreSurface = 1.,
                 skipSymmetryAnalyzis = False,
                 noOutput = False,
                 calcPropOnly = False,
                ):
        self.element=element
        self.Rnn = Rnn
        self.sizeP = sizeP
        self.sizeE = sizeE
        self.Marks = Marks
        self.nAtoms = 0
        self.nAtomsPerPentagonalCap = 0
        self.nAtomsPerElongatedPart = 0
        self.nAtomsPerEdgeOfPC = self.sizeP+1
        self.nAtomsPerEdgeOfEP = self.sizeE+1
        self.cog = np.array([0., 0., 0.])
        self.interCompactPlanesDistance = self.interCompactPlanesF * self.Rnn
        if self.Marks == 0 and self.sizeE == 0: #pentagonal bpy
            self.imageFile = pNMBu.imageNameWithPathway("pbpy-C.png")
        elif self.Marks != 0 and self.sizeE == 0: #Marks decahedron
            self.imageFile = pNMBu.imageNameWithPathway("MarksD-C.png")
        elif self.Marks == 0 and self.sizeE != 0: #Ino decahedron
            self.imageFile = pNMBu.imageNameWithPathway("InoD-C.png")
        else: #Elongated MArks decahedron
            self.imageFile = pNMBu.imageNameWithPathway("MarksD-C.png")
        if not noOutput: vID.centerTitle(f"Pentagonal bipyramid with {sizeP} atoms/edge, a x{sizeE} elongation (Ino) and a x{Marks} edge truncation (Marks)")

        if not noOutput: self.prop()
        if not calcPropOnly:
            self.coords(noOutput)
            if aseView: view(self.NP)
            if postAnalyzis:
                self.propPostMake(skipSymmetryAnalyzis,thresholdCoreSurface)
                if aseView: view(self.NPcs)

    def __str__(self):
        if self.Marks == 0:
            msg = f"Pentagonal pyramid with {self.sizeP+1} atoms per edge on the pentagonal cap, {self.sizeE} layer(s) in the elongated part, "\
                  f"no Marks truncation and Rnn = {self.Rnn}"
        else:
            msg = f"Pentagonal pyramid with {self.sizeP+1} atoms per edge on the pentagonal cap, {self.sizeE} layer(s) in the elongated part, "\
                  f"a Marks truncation by {self.Marks} atom(s) and Rnn = {self.Rnn}"
        return(msg)

    def edgeLength(self,whichEdge):
        if whichEdge == 'PC': #pentagonal cap
            return self.Rnn*(self.nAtomsPerEdgeOfPC-1)
        elif whichEdge == 'EP': #elongated part of an elongated bipyramid
            return self.Rnn*self.magicFactorF*(self.nAtomsPerEdgeOfEP-1)

    def area(self):
        el = self.edgeLength('PC')
        return el**2 * (5*np.sqrt(3)/2)
    
    def volume(self):
        el = self.edgeLength('PC')
        return el**3 * (5+np.sqrt(5))/12

    def heightOfPyramid(self):
        return self.heightOfPyramidF*self.Rnn*(2+self.nAtomsPerEdgeOfPC)

    def MakeVertices(self):
        """
        returns:
            - CoordVertices = the 7 vertex coordinates of a pentagonal dipyramid
            - edges = indexes of the 2x5 "vertical" edges of the pentagonal cap of an elongated pentagonal bipyramid 
            - faces3 = indexes of the 10 triangular faces
        """
        phi = self.phi
        scale = self.Rnn/2
        CoordVertices = [pNMBu.vertexScaled(0,0,self.sizeE*self.magicFactorF+self.sizeP*np.sqrt((10-2*np.sqrt(5))/5),scale),
                         pNMBu.vertexScaled(self.sizeP*np.sqrt((10+2*np.sqrt(5))/5),0,self.sizeE*self.magicFactorF,scale),
                         pNMBu.vertexScaled(self.sizeP*np.sqrt((5-np.sqrt(5))/10),self.sizeP*phi,self.sizeE*self.magicFactorF,scale),
                         pNMBu.vertexScaled(self.sizeP*-np.sqrt((5+2*np.sqrt(5))/5),self.sizeP,self.sizeE*self.magicFactorF,scale),
                         pNMBu.vertexScaled(self.sizeP*-np.sqrt((5+2*np.sqrt(5))/5),-self.sizeP,self.sizeE*self.magicFactorF,scale),
                         pNMBu.vertexScaled(self.sizeP*np.sqrt((5-np.sqrt(5))/10),-self.sizeP*phi,self.sizeE*self.magicFactorF,scale),
                        ]
        edgesPentagonalCap = [( 0, 1), ( 0, 2), ( 0, 3), ( 0, 4), ( 0, 5), (1, 2), (2, 3), (3, 4), (4, 5), (5, 1)]
        faces3 = [( 0, 1, 2), ( 0, 2, 3), ( 0, 3, 4), ( 0, 4, 5), ( 0, 5, 1)]

        CoordVertices = np.array(CoordVertices)
        edgesPentagonalCap = np.array(edgesPentagonalCap)
        faces3 = np.array(faces3)
        return CoordVertices, edgesPentagonalCap, faces3

    def truncationPlaneTuples4MarksDecahedron(self,refPlaneAtoms,debug=False):
        '''
        refPlaneAtoms contains [coordinates of the summit, coordinates of one apex atom, coordinatesof the origin]
        '''
        pRef = pNMBu.planeFittingLSF(refPlaneAtoms,printEq=False,printErrors=False)
        pRef = pRef[0:3]
        O = refPlaneAtoms[2]
        apexC = refPlaneAtoms[1]
        interPlanarDistance = self.interCompactPlanesDistance
        d = -interPlanarDistance * (self.nAtomsPerEdgeOfPC-self.Marks-1)
        plane0 = np.append(pRef,[d])
        planes = [plane0]
        
        for i in range(1,5):
            angle = i*72
            x = pNMBu.RotationMol(pRef,angle,'z')
            x = np.append(x,[d])
            planes.append(x)
            norm = pNMBu.normV(x)
            if (debug): print("angle = ",angle,"  plane = ",x,"   norm = ",norm)
        planes = np.array(planes)
        if (debug): print("\nplanes:\n",planes)
                    
        indices = [0, 1, 2, 3, 4, 0, 1]
        indicesOfTruncationPlanes = []
        for i in range(0,5):
            tuple = (indices[i],indices[i+2])
            indicesOfTruncationPlanes.append(tuple)
        if (debug): print("\nIndices of couples of truncation planes:\n",indicesOfTruncationPlanes)
        return planes, indicesOfTruncationPlanes

    def coords(self,noOutput):
        if not noOutput: vID.centertxt("Generation of coordinates",bgc='#007a7a',size='14',weight='bold')
        c = []
        # print(self.nAtomsPerLayer)
        indexVertexAtoms = []
        indexEdgePCAtoms = []
        indexEdgeEPAtoms = []
        indexFace3Atoms = []
        indexCoreAtoms = []

        # vertices
        nAtoms0 = 0
        self.nAtoms = 6
        cVertices, E, F3 = self.MakeVertices()
        c.extend(cVertices.tolist())
        indexVertexAtoms.extend(range(nAtoms0,self.nAtoms))
        # intermediate atoms on edges e
        nAtoms0 = self.nAtoms
        Rvv = pNMBu.RAB(cVertices,E[0,0],E[0,1]) #distance between two vertex atoms
        nAtomsOnEdges = int((Rvv+1e-6) / self.Rnn) - 1
        nIntervals = nAtomsOnEdges + 1
        coordEdgeAt = []
        for n in range(nAtomsOnEdges):
            for e in E:
                a = e[0]
                b = e[1]
                tmp = cVertices[a]+pNMBu.vector(cVertices,a,b)*(n+1) / nIntervals
                coordEdgeAt.append(tmp)
        self.nAtoms += nAtomsOnEdges * len(E)
        c.extend(coordEdgeAt)
        # CAtoms.extend(range(nAtoms0,self.nAtoms))
        self.nAtomsPerEdgeOfPC = nAtomsOnEdges  + 2 #2 vertices
        
        # now, triangular facets atoms
        coordFace3At = []
        nAtomsOnFaces3 = 0
        nAtoms0 = self.nAtoms
        for f in F3:
            nAtomsOnFaces3,coordFace3At = pNMBu.MakeFaceCoord(self.Rnn,f,c,nAtomsOnFaces3,coordFace3At)
        self.nAtoms += nAtomsOnFaces3
        c.extend(coordFace3At)
        indexFace3Atoms.extend(range(nAtoms0,self.nAtoms))
        
        # Marks decahedron?
        if self.Marks != 0:
            pMarks, indexCouples = self.truncationPlaneTuples4MarksDecahedron(np.array([c[0],c[1],[0,0,0]]))
            for ic in indexCouples:
                p0 = pMarks[ic[0]].copy()
                p1 = pMarks[ic[1]].copy()
                p1[0:3] = -p1[0:3] # don't forget to change the sign, see scheme in Sandbox 
                planes = [p0, p1]
                AtomsAboveAllPlanes = pNMBu.truncateAbovePlanes(planes,c,allP=True,delAbove=True,debug=False)
                c = pNMBu.deleteElementsOfAList(c,AtomsAboveAllPlanes)
                self.nAtoms = len(c)
        
        #reflection of the upper pyramid w.r.t. the (0,0,1) plane
        symPlane = np.array([0,0,1,0])
        ReflectionAtoms = pNMBu.reflection(symPlane,c,True)
        c.extend(ReflectionAtoms)
        self.nAtoms += len(ReflectionAtoms)

        # now internal atoms
        nAtomsHalfPy = int(self.nAtoms/2)
        coordCoreAt = []
        for i in range(nAtomsHalfPy):
            Rvv = pNMBu.RAB(c,i,i+nAtomsHalfPy) #distance between two mirror atoms
            nAtomsInCore = int((Rvv+1e-6) / self.Rnn*self.magicFactorF) - 1
            nIntervals = nAtomsInCore + 1
            for n in range(nAtomsInCore):
                tmp = c[i]+pNMBu.vector(c,i,i+nAtomsHalfPy)*(n+1) / nIntervals
                coordCoreAt.append(tmp)
        self.nAtoms += len(coordCoreAt)
        c.extend(coordCoreAt)
        # if self.sizeE = 0, it's a pentagonal bipyramid or a Marks decahedron without side faces
        # given the doItForAtomsThatLieInTheReflectionPlane trick in pNMBu.reflection, it is necessary to remove duplicate atoms
        if self.sizeE == 0:
            c = np.unique(np.array(c),axis=0)
            self.nAtoms = len(c)
 
        aseObject = ase.Atoms(self.element*self.nAtoms, positions=c)

        self.cog = pNMBu.centerOfGravity(c)
        print(f"Total number of atoms = {self.nAtoms}")
        self.NP = aseObject

    def prop(self):
        print(self)
        pNMBu.plotImageInPropFunction(self.imageFile)
        print("element = ",self.element)
        if self.sizeE == 0 and self.Marks == 0:
            print("number of vertices = ",self.nVerticesPbpy)
            print("number of edges = ",self.nEdgesPbpy)
            print("number of faces = ",self.nFaces3)
        elif self.sizeE != 0 and self.Marks == 0:
            print("number of vertices = ",self.nVerticesEpbpy)
            print("number of edges = ",self.nEdgesEpbpy)
            print("number of faces = ",self.nFaces3+self.nFaces4)
        print(f"magic factor = {self.magicFactorF:.3f} (ratio  between the height of the vertical interatomic distance and the nearest-neighbour distance)")
        print(f"nearest neighbour distance = {self.Rnn:.2f} Å")
        print(f"edge length of the pentagonal cap = {self.edgeLength('PC')*0.1:.2f} nm")
        print(f"edge length of the elongated part = {self.edgeLength('EP')*0.1:.2f} nm")
        print(f"inter compact planes factor = {self.interCompactPlanesF:.3f}")
        print(f"inter compact planes distance = {self.interCompactPlanesDistance:.2f} Å")
        # print(f"inter-layer distance = {self.interLayerDistance:.2f} Å")
        print(f"number of atoms per edge on the pentagonal cap = {self.nAtomsPerEdgeOfPC}")
        if self.Marks == 0:
            structure = "bipyramid"
        else:
            structure = "Marks decahedron"
        if self.sizeE == 0:
            print(f"height of the {structure} = {self.heightOfPyramid():.2f} Å")
            if self.Marks == 0:
                print(f"area = {self.area()*1e-2:.1f} nm2")
                print(f"volume = {self.volume()*1e-3:.1f} nm3")
        elif self.sizeE != 0:
            print(f"height of the elongated {structure} = {(self.heightOfPyramid() + self.edgeLength('EP'))*0.1:.2f} nm")
        # print("number of atoms per layer = ",self.Td.nAtomsPerLayerAnalytic())
        # print("total number of atoms = ",self.nAtomsAnalytic())
        if self.sizeE == 0 and self.Marks == 0:
            print("Dual polyhedron: triangular prism")
        elif self.sizeE != 0 and self.Marks == 0:
            print("Dual polyhedron: pentagonal bifrustum")
        print(f"coordinates of the center of gravity = {self.cog}")

    def propPostMake(self,skipSymmetryAnalyzis,thresholdCoreSurface):
        pNMBu.moi(self.NP)
        if not skipSymmetryAnalyzis: pNMBu.MolSym(self.NP)
        [self.vertices,self.simplices,self.neighbors,self.equations],surfaceAtoms =\
            pNMBu.coreSurface(self.NP.get_positions(),thresholdCoreSurface)
        self.NPcs = self.NP.copy()
        self.NPcs.numbers[np.invert(surfaceAtoms)] = 102 #Nobelium, because it has a nice pinkish color in jmol
