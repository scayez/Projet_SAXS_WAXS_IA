from visualID import  fg, hl, bg
import visualID as vID

import sys
import numpy as np
import pyNanoMatBuilder.utils as pNMBu
import ase
from ase.build import bulk, make_supercell, cut
from ase.visualize import view
from ase.cluster.cubic import FaceCenteredCubic
import os

###########################################################################################################
class regfccOh:
    nFaces = 8
    nEdges = 12
    nVertices = 6
    edgeLengthF = 1 # length of an edge
    radiusCSF = edgeLengthF * np.sqrt(2)/2 #Centroid to vertex distance = Radius of circumsphere
    radiusISF = edgeLengthF * np.sqrt(6)/6 #Radius of insphere that is tangent to faces
    radiusMSF = edgeLengthF / 2 #Radius of midsphere that is tangent to edges
    dihedralAngle = np.rad2deg(np.arccos(-1/3))
    interShellF = 1/radiusCSF
  
    def __init__(self,
                 element: str='Au',
                 Rnn: float = 2.7,
                 nOrder: int = 1,
                 postAnalyzis=True,
                 aseView=True,
                 thresholdCoreSurface = 1.,
                 skipSymmetryAnalyzis = False,
                 noOutput = False,
                 calcPropOnly = False,
                ):
        self.element = element
        self.Rnn = Rnn
        self.nOrder = nOrder
        self.nAtoms = 0
        self.nAtomsPerLayer = []
        self.nAtomsPerEdge = self.nOrder+1
        self.interLayerDistance = self.Rnn/self.interShellF
        self.cog = np.array([0., 0., 0.])
        self.imageFile = pNMBu.imageNameWithPathway("fccOh-C.png")
        if not noOutput: vID.centerTitle(f"{nOrder}th order regular fcc Octahedron")

        if not noOutput: self.prop()
        if not calcPropOnly:
            self.coords(noOutput)
            if aseView: view(self.NP)
            if postAnalyzis:
                self.propPostMake(skipSymmetryAnalyzis,thresholdCoreSurface)
                if aseView: view(self.NPcs)
          
    def __str__(self):
        return(f"Regular octahedron of order {self.nOrder} (i.e. {self.nOrder+1} atoms lie on an edge) and Rnn = {self.Rnn}")
    
    def nAtomsF(self,i):
        """ returns the number of atoms of an octahedron of size i"""
        return round((2/3)*i**3 + 2*i**2 + (7/3)*i + 1)
    
    def nAtomsPerShellAnalytic(self):
        n = []
        Sum = 0
        for i in range(1,self.nOrder+1):
            Sum = sum(n)
            ni = self.nAtomsF(i)
            n.append(ni-Sum)
        return n

    def nAtomsPerShellCumulativeAnalytic(self):
        n = []
        Sum = 0
        for i in range(1,self.nOrder+1):
            Sum = sum(n)
            ni = self.nAtomsF(i)
            n.append(ni)
        return n    

    def nAtomsAnalytic(self):
        n = self.nAtomsF(self.nOrder)
        return n
    
    def edgeLength(self):
        return self.Rnn**self.nOrder

    def radiusCircumscribedSphere(self):
        return self.radiusCSF*self.edgeLength()

    def radiusInscribedSphere(self):
        return self.radiusISF*self.edgeLength()

    def radiusMidSphere(self):
        return self.radiusMSF*self.edgeLength()

    def area(self):
        el = self.edgeLength()
        return el**2*np.sqrt(3)
    
    def volume(self):
        el = self.edgeLength()
        return np.sqrt(2)*el**3/3 

    def MakeVertices(self,i):
        """
        input:
            - i = index of the shell
        returns:
            - CoordVertices = the 6 vertex coordinates of the ith shell of an octahedron
            - edges = indexes of the 30 edges
            - faces = indexes of the 20 faces 
        """
        if (i == 0):
            CoordVertices = [0., 0., 0.]
            edges = []
            faces = []
        elif (i > self.nOrder):
            sys.exit(f"regfccOh.MakeVertices(i) is called with i = {i} > nOrder = {self.nOrder}")
        else:
            scale = self.interLayerDistance * i
            CoordVertices = [ pNMBu.vertex(-1, 0, 0, scale),\
                              pNMBu.vertex( 1, 0, 0, scale),\
                              pNMBu.vertex( 0,-1, 0, scale),\
                              pNMBu.vertex( 0, 1, 0, scale),\
                              pNMBu.vertex( 0, 0,-1, scale),\
                              pNMBu.vertex( 0, 0, 1, scale)]
            edges = [( 2, 0), ( 2, 1), ( 3, 0), ( 3, 1), ( 4, 0), ( 4, 1), ( 4, 2), ( 4, 3), ( 5, 0), ( 5, 1), ( 5, 2), ( 5, 3)]
            faces = [( 2, 0, 4), ( 2, 0, 5), ( 2, 1, 4), ( 2, 1, 5), ( 3, 0, 4), ( 3, 0, 5), ( 3, 1, 4), ( 3, 1, 5)]
            CoordVertices = np.array(CoordVertices)
            edges = np.array(edges)
            faces = np.array(faces)
        return CoordVertices, edges, faces

    def coords(self,noOutput):
        if not noOutput: vID.centertxt("Generation of coordinates",bgc='#007a7a',size='14',weight='bold')
        chrono = pNMBu.timer(); chrono.chrono_start()
        c = []
        # print(self.nAtomsPerLayer)
        indexVertexAtoms = []
        indexEdgeAtoms = []
        indexFaceAtoms = []
        indexCoreAtoms = []

        # vertices
        nAtoms0 = 0
        self.nAtoms += self.nVertices
        cVertices, E, F = self.MakeVertices(self.nOrder)
        c.extend(cVertices.tolist())
        indexVertexAtoms.extend(range(nAtoms0,self.nAtoms))

        # intermediate atoms on edges e
        nAtoms0 = self.nAtoms
        Rvv = pNMBu.RAB(cVertices,E[0,0],E[0,1]) #distance between two vertex atoms
        nAtomsOnEdges = int((Rvv+1e-6) / self.Rnn)-1
        nIntervals = nAtomsOnEdges + 1
        #print("nAtomsOnEdges = ",nAtomsOnEdges)
        coordEdgeAt = []
        for n in range(nAtomsOnEdges):
            for e in E:
                a = e[0]
                b = e[1]
                coordEdgeAt.append(cVertices[a]+pNMBu.vector(cVertices,a,b)*(n+1) / nIntervals)
        self.nAtoms += nAtomsOnEdges * len(E)
        c.extend(coordEdgeAt)
        indexEdgeAtoms.extend(range(nAtoms0,self.nAtoms))
        # print(indexEdgeAtoms)
        
        # now, facet atoms
        coordFaceAt = []
        nAtomsOnFaces = 0
        nAtoms0 = self.nAtoms
        for f in F:
            nAtomsOnFaces,coordFaceAt = pNMBu.MakeFaceCoord(self.Rnn,f,c,nAtomsOnFaces,coordFaceAt)
        self.nAtoms += nAtomsOnFaces
        c.extend(coordFaceAt)
        indexFaceAtoms.extend(range(nAtoms0,self.nAtoms))

        # now, core atoms. Layer by layer strategy, starting from bottom to top
        # when identified, just use MakeFaceCoord and define, for each layer, the four atoms on the edge as a facet
        coordCoreAt = []
        nAtomsInCore = 0
        nAtoms0 = self.nAtoms
        # first apply it to atoms 0, 1, 2, 3
        # f = [a,b,c,d] must be given in the order a--b
        #                                          |  |
        #                                          d--c
        f = np.array([0,3,1,2])
        nAtomsInCore,coordCoreAt = pNMBu.MakeFaceCoord(self.Rnn,f,c,nAtomsInCore,coordCoreAt)
        # don't ask... it is the algorithm to find the indexes of the square
        # corners of each layer along z
        def layerup(ilayer,f):
            return 12*ilayer+f-2
        def layerdown(ilayer,f):
            return 12*ilayer+f+2
        for i in range(1,nAtomsOnEdges+1): 
            f = layerup(i,np.array([0,3,1,2]))
            # print(i,"  fup",f)
            nAtomsInCore,coordCoreAt = pNMBu.MakeFaceCoord(self.Rnn,f,c,nAtomsInCore,coordCoreAt)
            f = layerdown(i,np.array([0,3,1,2]))
            # print(i,"fdown",f)
            nAtomsInCore,coordCoreAt = pNMBu.MakeFaceCoord(self.Rnn,f,c,nAtomsInCore,coordCoreAt)
        self.nAtoms += nAtomsInCore
        c.extend(coordCoreAt)
        indexCoreAtoms.extend(range(nAtoms0,self.nAtoms))

        print(f"Total number of atoms = {self.nAtoms}")
        aseObject = ase.Atoms(self.element*self.nAtoms, positions=c)

        self.cog = pNMBu.centerOfGravity(c)
      
        chrono.chrono_stop(hdelay=False); chrono.chrono_show()
        self.NP = aseObject
            
    def prop(self):
        vID.centertxt("Properties",bgc='#007a7a',size='14',weight='bold')
        print(self)
        pNMBu.plotImageInPropFunction(self.imageFile)
        print("element = ",self.element)
        print("number of vertices = ",self.nVertices)
        print("number of edges = ",self.nEdges)
        print("number of faces = ",self.nFaces)
        print(f"intershell factor = {self.interShellF:.2f}")
        print(f"nearest neighbour distance = {self.Rnn:.2f} Å")
        print(f"interlayer distance = {self.interLayerDistance:.2f} Å")
        print(f"edge length = {self.edgeLength()*0.1:.2f} nm")
        print(f"number of atoms per edge = {self.nAtomsPerEdge}")
        print(f"radius after volume = {pNMBu.RadiusSphereAfterV(self.volume()*1e-3):.2f} nm")
        print(f"radius of the circumscribed sphere = {self.radiusCircumscribedSphere()*0.1:.2f} nm")
        print(f"radius of the inscribed sphere = {self.radiusInscribedSphere()*0.1:.2f} nm")
        print(f"area = {self.area()*1e-2:.1f} nm2")
        print(f"volume = {self.volume()*1e-3:.1f} nm3")
        print(f"dihedral angle = {self.dihedralAngle:.1f}°")
        # print("number of atoms per shell = ",self.nAtomsPerShellAnalytic())
        print("intermediate magic numbers = ",self.nAtomsPerShellCumulativeAnalytic())
        print("total number of atoms = ",self.nAtomsAnalytic())
        print("Dual polyhedron: cube")
        print("Indexes of vertex atoms = [0,1,2,3,4,5] by construction")
        print(f"coordinates of the center of gravity = {self.cog}")
        return

    def propPostMake(self,skipSymmetryAnalyzis,thresholdCoreSurface):
        pNMBu.moi(self.NP)
        if not skipSymmetryAnalyzis: pNMBu.MolSym(self.NP)
        [self.vertices,self.simplices,self.neighbors,self.equations],surfaceAtoms =\
            pNMBu.coreSurface(self.NP.get_positions(),thresholdCoreSurface)
        self.NPcs = self.NP.copy()
        self.NPcs.numbers[np.invert(surfaceAtoms)] = 102 #Nobelium, because it has a nice pinkish color in jmol

###########################################################################################################
class regIco:
    nFaces = 20
    nEdges = 30
    nVertices = 12
    phi = (1 + np.sqrt(5))/2 # golden ratio
    edgeLengthF = 1
    radiusCSF = np.sqrt(10 + 2*np.sqrt(5))/4
    interShellF = 1/radiusCSF
#    interShellF = np.sqrt(2*(1-1/np.sqrt(5)))
    radiusISF = np.sqrt(3) * (3 + np.sqrt(5))/12
  
    def __init__(self,
                 element: str='Au',
                 Rnn: float=2.7,
                 nShell: int=1,
                 postAnalyzis=True,
                 aseView=True,
                 thresholdCoreSurface = 1.,
                 skipSymmetryAnalyzis = False,
                 noOutput = False,
                 calcPropOnly = False,
                ):
        self.element=element
        self.Rnn = Rnn
        self.nShell = nShell
        self.nAtoms = 0
        self.nAtomsPerShell = [0]
        self.interShellDistance = self.Rnn / self.interShellF
        self.imageFile = pNMBu.imageNameWithPathway("ico-C.png")
        if not noOutput: vID.centerTitle(f"{nShell} shells icosahedron")
          
        if not noOutput: self.prop()
        if not calcPropOnly:
            self.coords(noOutput)
            if aseView: view(self.NP)
            if postAnalyzis:
                self.propPostMake(skipSymmetryAnalyzis,thresholdCoreSurface)
                if aseView: view(self.NPcs)
          
    def __str__(self):
        return(f"Regular icosahedron with {self.nShell} shell(s) and Rnn = {self.Rnn}")
    
    def nAtomsF(self,i):
        """ returns the number of atoms of an icosahedron of size i"""
        return (10*i**3 + 11*i)//3 + 5*i**2 + 1
    
    def nAtomsPerShellAnalytic(self):
        n = []
        Sum = 0
        for i in range(self.nShell+1):
            Sum = sum(n)
            ni = self.nAtomsF(i)
            n.append(ni-Sum)
        return n
    
    def nAtomsPerShellCumulativeAnalytic(self):
        n = []
        Sum = 0
        for i in range(self.nShell+1):
            Sum = sum(n)
            ni = self.nAtomsF(i)
            n.append(ni)
        return n
    
    def nAtomsAnalytic(self):
        n = self.nAtomsF(self.nShell)
        return n
    
    def edgeLength(self):
        return self.Rnn*self.nShell

    def radiusCircumscribedSphere(self):
        return self.radiusCSF*self.edgeLength()

    def radiusInscribedSphere(self):
        return self.radiusISF*self.edgeLength()

    def area(self):
        el = self.edgeLength()
        return 5 * el**2 * np.sqrt(3)
    
    def volume(self):
        el = self.edgeLength()
        return 5 * el**3 * (3 + np.sqrt(5))/12
    
    def MakeVertices(self,i):
        """
        input:
            - i = index of the shell
        returns:
            - CoordVertices = the 12 vertex coordinates of the ith shell of an icosahedron
            - edges = indexes of the 30 edges
            - faces = indexes of the 20 faces 
        """
        if (i == 0):
            CoordVertices = [0., 0., 0.]
            edges = []
            faces = []
        elif (i > self.nShell):
            sys.exit(f"icoreg.MakeVertices(i) is called with i = {i} > nShell = {self.nShell}")
        else:
            phi = self.phi
            scale = self.interShellDistance * i
            CoordVertices = [ pNMBu.vertex(-1, phi, 0, scale),\
                              pNMBu.vertex( 1, phi, 0, scale),\
                              pNMBu.vertex(-1, -phi, 0, scale),\
                              pNMBu.vertex( 1, -phi, 0, scale),\
                              pNMBu.vertex(0, -1, phi, scale),\
                              pNMBu.vertex(0, 1, phi, scale),\
                              pNMBu.vertex(0, -1, -phi, scale),\
                              pNMBu.vertex(0, 1, -phi, scale),\
                              pNMBu.vertex( phi, 0, -1, scale),\
                              pNMBu.vertex( phi, 0, 1, scale),\
                              pNMBu.vertex(-phi, 0, -1, scale),\
                              pNMBu.vertex(-phi, 0, 1, scale) ]
            edges = [( 1, 0), ( 3, 2), ( 4, 2), ( 4, 3), ( 5, 0), ( 5, 1), ( 5, 4), ( 6, 2), ( 6, 3), ( 7, 0),\
                  ( 7, 1), ( 7, 6), ( 8, 1), ( 8, 3), ( 8, 6), ( 8, 7), ( 9, 1), ( 9, 3), ( 9, 4), ( 9, 5),\
                  ( 9, 8), (10, 0), (10, 2), (10, 6), (10, 7), (11, 0), (11, 2), (11, 4), (11, 5), (11,10),]
            faces = [(7,0,1),(7,1,8),(7,8,6),(7,6,10),(7,10,0),\
                     (0,11,5),(0,5,1),(1,5,9),(1,8,9),(8,9,3),(8,3,6),(6,3,2),(6,10,2),(10,2,11),(10,0,11),\
                     (4,2,3),(4,3,9),(4,9,5),(4,5,11),(4,11,2)]
            edges = np.array(edges)
            CoordVertices = np.array(CoordVertices)
            faces = np.array(faces)
        return CoordVertices, edges, faces

    def coords(self,noOutput):
        if not noOutput: vID.centertxt("Generation of coordinates",bgc='#007a7a',size='14',weight='bold')
        chrono = pNMBu.timer(); chrono.chrono_start()
        # central atom = "1st shell"
        c = [[0., 0., 0.]]
        self.nAtoms = 1
        self.nAtomsPerShell = [0]
        self.nAtomsPerShell[0] = 1
        indexVertexAtoms = []
        indexEdgeAtoms = []
        indexFaceAtoms = []
        for i in range(1,self.nShell+1):
            # vertices
            nAtoms0 = self.nAtoms
            cshell, E, F = self.MakeVertices(i)
            self.nAtoms += self.nVertices
            self.nAtomsPerShell.append(self.nVertices)
            c.extend(cshell.tolist())
            indexVertexAtoms.extend(range(nAtoms0,self.nAtoms))

            # intermediate atoms on edges e
            nAtoms0 = self.nAtoms
            Rvv = pNMBu.RAB(cshell,E[0,0],E[0,1]) #distance between two vertex atoms
            nAtomsOnEdges = int((Rvv+1e-6) / self.Rnn)-1
            nIntervals = nAtomsOnEdges + 1
            # print("nAtomsOnEdges = ",nAtomsOnEdges)
            coordEdgeAt = []
            for n in range(nAtomsOnEdges):
                for e in E:
                    a = e[0]
                    b = e[1]
                    coordEdgeAt.append(cshell[a]+pNMBu.vector(cshell,a,b)*(n+1) / nIntervals)
            self.nAtomsPerShell[i] += nAtomsOnEdges * len(E) # number of edges x nAtomsOnEdges
            self.nAtoms += nAtomsOnEdges * len(E)
            c.extend(coordEdgeAt)
            indexEdgeAtoms.extend(range(nAtoms0,self.nAtoms))
            
            # now, facet atoms
            coordFaceAt = []
            nAtomsOnFaces = 0
            nAtoms0 = self.nAtoms
            for f in F:
                nAtomsOnFaces,coordFaceAt = pNMBu.MakeFaceCoord(self.Rnn,f,cshell,nAtomsOnFaces,coordFaceAt)
            self.nAtomsPerShell[i] += nAtomsOnFaces
            self.nAtoms += nAtomsOnFaces
            c.extend(coordFaceAt)
            indexFaceAtoms.extend(range(nAtoms0,self.nAtoms))

        print(f"Total number of atoms = {self.nAtoms}")
        print(self.nAtomsPerShell)
        aseObject = ase.Atoms(self.element*self.nAtoms, positions=c)
            
        # print(indexVertexAtoms)
        # print(indexEdgeAtoms)
        # print(indexFaceAtoms)
        chrono.chrono_stop(hdelay=False); chrono.chrono_show()
        self.NP=aseObject
    
    def prop(self):
        vID.centertxt("Properties",bgc='#007a7a',size='14',weight='bold')
        print(self)
        pNMBu.plotImageInPropFunction(self.imageFile)
        print("element = ",self.element)
        print("number of vertices = ",self.nVertices)
        print("number of edges = ",self.nEdges)
        print("number of faces = ",self.nFaces)
        print("phi = ",self.phi)
        print(f"intershell factor = {self.interShellF:.2f}")
        print(f"nearest neighbour distance = {self.Rnn:.2f} Å")
        print(f"intershell distance = {self.interShellDistance:.2f} Å")
        print(f"edge length = {self.edgeLength()*0.1:.2f} nm")
        print(f"radius after volume = {pNMBu.RadiusSphereAfterV(self.volume()*1e-3):.2f} nm")
        print(f"radius of the circumscribed sphere = {self.radiusCircumscribedSphere()*0.1:.2f} nm")
        print(f"radius of the inscribed sphere = {self.radiusInscribedSphere()*0.1:.2f} nm")
        print(f"area = {self.area()*1e-2:.1f} nm2")
        print(f"volume = {self.volume()*1e-3:.1f} nm3")
        print("number of atoms per shell = ",self.nAtomsPerShellAnalytic())
        print("cumulative number of atoms per shell = ",self.nAtomsPerShellCumulativeAnalytic())
        print("total number of atoms = ",self.nAtomsAnalytic())
        print("Dual polyhedron: dodecahedron")

    def propPostMake(self,skipSymmetryAnalyzis,thresholdCoreSurface):
        pNMBu.moi(self.NP)
        if not skipSymmetryAnalyzis: pNMBu.MolSym(self.NP)
        [self.vertices,self.simplices,self.neighbors,self.equations],surfaceAtoms =\
            pNMBu.coreSurface(self.NP.get_positions(),thresholdCoreSurface)
        self.NPcs = self.NP.copy()
        self.NPcs.numbers[np.invert(surfaceAtoms)] = 102 #Nobelium, because it has a nice pinkish color in jmol

###########################################################################################################
class regfccTd:
    nFaces = 4
    nEdges = 6
    nVertices = 4
    edgeLengthF = 1 # length of an edge
    heightOfPyramidF = edgeLengthF * np.sqrt(2/3)
    radiusCSF = edgeLengthF * np.sqrt(3/8) #Centroid to vertex distance = Radius of circumsphere
    radiusISF = edgeLengthF/np.sqrt(24) #Radius of insphere that is tangent to faces
    radiusMSF = edgeLengthF/np.sqrt(8) #Radius of midsphere that is tangent to edges
    fveAngle = np.rad2deg(np.arccos(1/np.sqrt(3))) #Face-vertex-edge angle
    fefAngle = np.rad2deg(np.arccos(1/3)) #Face-edge-face angle
    vcvAngle = np.rad2deg(np.arccos(-1/3)) #Vertex-Center-Vertex angle
  
    def __init__(self,
                 element: str='Au',
                 Rnn: float=2.7,
                 nLayer: int=1,
                 postAnalyzis=True,
                 aseView=True,
                 thresholdCoreSurface = 1.,
                 skipSymmetryAnalyzis = False,
                 noOutput = False,
                 calcPropOnly = False,
                ):
        self.element = element
        self.Rnn = Rnn
        self.nLayer = nLayer
        self.nAtoms = 0
        self.nAtomsPerLayer = []
        self.nAtomsPerEdge = self.nLayer
        self.cog = np.array([0., 0., 0.])
        self.imageFile = pNMBu.imageNameWithPathway("fccTd-C.png")
        self.NP = None
        if not noOutput: vID.centerTitle(f"fcc tetrahedron: {nLayer} atoms/edge = number of layers")
          
        if not noOutput: self.prop()

        if not calcPropOnly:
           self.coords(noOutput)
           if aseView: view(self.NP)
           if postAnalyzis:
               self.propPostMake(skipSymmetryAnalyzis,thresholdCoreSurface)
               if aseView: view(self.NPcs)
          
    def __str__(self):
        return(f"Regular tetrahedron with {self.nLayer} layer(s) and Rnn = {self.Rnn}")
    
    def nAtomsF(self,i):
        return round(i**3/6 + i**2 + 11*i/6 + 1)
    
    def nAtomsPerLayerAnalytic(self):
        n = []
        Sum = 0
        for i in range(self.nLayer):
            Sum = sum(n)
            ni = int(self.nAtomsF(i))
            n.append(ni-Sum)
            # print(i,ni,Sum,n)
        return n
    
    def nAtomsAnalytic(self):
        n = self.nAtomsF(self.nLayer-1)
        return n
    
    def edgeLength(self):
        return self.Rnn*(self.nLayer-1)

    def heightOfPyramid(self):
        return self.heightOfPyramidF*self.edgeLength()
    
    def interLayerDistance(self):
        return self.heightOfPyramid()/(self.nLayer-1)
    
    def radiusCircumscribedSphere(self):
        return self.radiusCSF*self.edgeLength()

    def radiusInscribedSphere(self):
        return self.radiusISF*self.edgeLength()

    def radiusMidSphere(self):
        return self.radiusMSF*self.edgeLength()

    def area(self):
        el = self.edgeLength()
        return el**2*np.sqrt(3)
    
    def volume(self):
        el = self.edgeLength()
        return el**3/(6*np.sqrt(2)) 

    def MakeVertices(self,nL):
        """
        input:
            - nL = number of layers = number of atoms per edge
        returns:
            - CoordVertices = the 4 vertex coordinates of a tetrahedron
            - edges = indexes of the 6 edges
            - faces = indexes of the 4 faces 
        """
        if (nL > self.nLayer):
            sys.exit(f"regTd.MakeVertices(nL) is called with nL = {nL} > nLayer = {self.nLayer}")
        else:
            scale = self.radiusCircumscribedSphere()
            c = 1/(2*np.sqrt(2)) # edge length 1
            CoordVertices = [pNMBu.vertex(c, c, c, scale),\
                             pNMBu.vertex(c, -c, -c, scale),\
                             pNMBu.vertex(-c, c, -c, scale),\
                             pNMBu.vertex(-c, -c, c, scale)]
            edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
            faces = [(0,2,1),(0,1,3),(0,3,2),(1,2,3)]
            edges = np.array(edges)
            CoordVertices = np.array(CoordVertices)
            faces = np.array(faces)
        return CoordVertices, edges, faces

    def coords(self,noOutput):
        if not noOutput: vID.centertxt("Generation of coordinates",bgc='#007a7a',size='14',weight='bold')
        chrono = pNMBu.timer(); chrono.chrono_start()
        c = []
        # print(self.nAtomsPerLayer)
        indexVertexAtoms = []
        indexEdgeAtoms = []
        indexFaceAtoms = []
        indexCoreAtoms = []

        # vertices
        nAtoms0 = 0
        self.nAtoms += self.nVertices
        cVertices, E, F = self.MakeVertices(self.nLayer-1)
        c.extend(cVertices.tolist())
        indexVertexAtoms.extend(range(nAtoms0,self.nAtoms))

        # intermediate atoms on edges e
        nAtoms0 = self.nAtoms
        Rvv = pNMBu.RAB(cVertices,E[0,0],E[0,1]) #distance between two vertex atoms
        nAtomsOnEdges = int((Rvv+1e-6) / self.Rnn)-1
        nIntervals = nAtomsOnEdges + 1
        #print("nAtomsOnEdges = ",nAtomsOnEdges)
        coordEdgeAt = []
        for n in range(nAtomsOnEdges):
            for e in E:
                a = e[0]
                b = e[1]
                coordEdgeAt.append(cVertices[a]+pNMBu.vector(cVertices,a,b)*(n+1) / nIntervals)
        self.nAtoms += nAtomsOnEdges * len(E)
        c.extend(coordEdgeAt)
        indexEdgeAtoms.extend(range(nAtoms0,self.nAtoms))
        self.nAtomsPerEdge = nAtomsOnEdges  + 2 #2 vertices
        # print(indexEdgeAtoms)
        
        # now, facet atoms
        coordFaceAt = []
        nAtomsOnFaces = 0
        nAtoms0 = self.nAtoms
        for f in F:
            nAtomsOnFaces,coordFaceAt = pNMBu.MakeFaceCoord(self.Rnn,f,c,nAtomsOnFaces,coordFaceAt)
        self.nAtoms += nAtomsOnFaces
        c.extend(coordFaceAt)
        indexFaceAtoms.extend(range(nAtoms0,self.nAtoms))

        # now, core atoms. Layer by layer strategy, using atoms on edges [0-1],[0-2],[0-3]
        # when identified, just use MakeFaceCoord and define, for each layer, the three atoms on the edge as a facet
        # just start from 4th layer
        coordCoreAt = []
        nAtomsInCore = 0
        nAtoms0 = self.nAtoms
        for ilayer in range(4,self.nLayer+1):
            FirstAtom = 4 + (ilayer-2)*6
            f = np.array([FirstAtom, FirstAtom+1, FirstAtom+2])
            # print("layer ",ilayer,f)
            nAtomsInCore,coordCoreAt = pNMBu.MakeFaceCoord(self.Rnn,f,c,nAtomsInCore,coordCoreAt)
        self.nAtoms += nAtomsInCore
        c.extend(coordCoreAt)
        indexCoreAtoms.extend(range(nAtoms0,self.nAtoms))

        print(f"Total number of atoms = {self.nAtoms}")
        print(self.nAtomsPerLayer)
        aseObject = ase.Atoms(self.element*self.nAtoms, positions=c)

        self.cog = pNMBu.centerOfGravity(c)
        
        chrono.chrono_stop(hdelay=False); chrono.chrono_show()
        self.NP = aseObject
    
    def prop(self):
        vID.centertxt("Properties",bgc='#007a7a',size='14',weight='bold')
        print(self)
        pNMBu.plotImageInPropFunction(self.imageFile)
        print("element = ",self.element)
        print("number of vertices = ",self.nVertices)
        print("number of edges = ",self.nEdges)
        print("number of faces = ",self.nFaces)
        print(f"nearest neighbour distance = {self.Rnn:.2f} Å")
        print(f"edge length = {self.edgeLength()*0.1:.2f} nm")
        print(f"number of atoms per edge = {self.nAtomsPerEdge}")
        print(f"inter-layer distance = {self.interLayerDistance():.2f} Å")
        print(f"height of pyramid = {self.heightOfPyramid()*0.1:.2f} nm")
        print(f"radius after volume = {pNMBu.RadiusSphereAfterV(self.volume()*1e-3):.2f} nm")
        print(f"radius of the circumscribed sphere = {self.radiusCircumscribedSphere()*0.1:.2f} nm")
        print(f"radius of the inscribed sphere = {self.radiusInscribedSphere()*0.1:.2f} nm")
        print(f"radius of the midsphere that is tangent to edges = {self.radiusMidSphere()*0.1:.2f} nm")
        print(f"area = {self.area()*1e-2:.1f} nm2")
        print(f"volume = {self.volume()*1e-3:.1f} nm3")
        print(f"face-vertex-edge angle = {self.fveAngle:.1f}°")
        print(f"face-edge-face (dihedral) angle = {self.fefAngle:.1f}°")
        print(f"vertex-center-vertex (tetrahedral bond) angle = {self.vcvAngle:.1f}°")
        print("number of atoms per layer = ",self.nAtomsPerLayerAnalytic())
        print("total number of atoms = ",self.nAtomsAnalytic())
        print("Dual polyhedron: tetrahedron")
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
class regDD:
    nFaces = 12
    nEdges = 30
    nVertices = 20
    phi = (1 + np.sqrt(5))/2 # golden ratio
    edgeLengthF = 1
    radiusCSF = np.sqrt(3) * (1 + np.sqrt(5))/4
    interShellF = 1/radiusCSF
    radiusISF = np.sqrt((5/2) + (11/10)*np.sqrt(5))/2
  
    def __init__(self,
                 element: str='Au',
                 Rnn: float=2.7,
                 nShell: int=1,
                 postAnalyzis=True,
                 aseView=True,
                 thresholdCoreSurface = 1.,
                 skipSymmetryAnalyzis = False,
                 noOutput = False,
                 calcPropOnly = False,
                ):
        self.element = element
        self.Rnn = Rnn
        self.nShell = nShell
        self.nAtoms = 0
        self.nAtomsPerShell = [0]
        self.interShellDistance = self.Rnn / self.interShellF
        self.imageFile = pNMBu.imageNameWithPathway("rDD-C.png")
        vID.centerTitle(f"{nShell} shells regular dodecahedron")
          
        if not noOutput: self.prop()
        if not calcPropOnly:
            self.coords(noOutput)
            if aseView: view(self.NP)
            if postAnalyzis:
                self.propPostMake(skipSymmetryAnalyzis,thresholdCoreSurface)
                if aseView: view(self.NPcs)
          
    def __str__(self):
        return(f"Regular dodecahedron with {self.nShell} shell(s) and Rnn = {self.Rnn}")
    
    def nAtomsF(self,i):
        """ returns the number of atoms of a dodecahedron of size i"""
        return 10*i**3 + 15*i**2 + 7*i + 1
    
    def nAtomsPerShellAnalytic(self):
        n = []
        Sum = 0
        for i in range(self.nShell+1):
            Sum = sum(n)
            ni = self.nAtomsF(i)
            n.append(ni-Sum)
        return n
    
    def nAtomsAnalytic(self):
        n = self.nAtomsF(self.nShell)
        return n
    
    def edgeLength(self):
        return self.Rnn*self.nShell

    def radiusCircumscribedSphere(self):
        return self.radiusCSF*self.edgeLength()

    def radiusInscribedSphere(self):
        return self.radiusISF*self.edgeLength()

    def area(self):
        el = self.edgeLength()
        return 3 * el**2 * np.sqrt(25 + 10*np.sqrt(5))
    
    def volume(self):
        el = self.edgeLength()
        return (15 + 7*np.sqrt(5)) * el**2/4 

    def MakeVertices(self,i):
        """
        input:
            - i = index of the shell
        returns:
            - CoordVertices = the 20 vertex coordinates of the ith shell of a dodecahedron
            - edges = indexes of the 30 edges
            - faces = indexes of the 12 faces 
        """
        if (i == 0):
            CoordVertices = [0., 0., 0.]
            edges = []
            faces = []
        elif (i > self.nShell):
            sys.exit(f"icoreg.MakeVertices(i) is called with i = {i} > nShell = {self.nShell}")
        else:
            phi = self.phi
            scale = self.interShellDistance * i
            CoordVertices = [pNMBu.vertex(1, 1, 1, scale),\
                             pNMBu.vertex(-1, 1, 1, scale),\
                             pNMBu.vertex(1, -1, 1, scale),\
                             pNMBu.vertex(1, 1, -1, scale),\
                             pNMBu.vertex(-1, -1, 1, scale),\
                             pNMBu.vertex(-1, 1, -1, scale),\
                             pNMBu.vertex(1, -1, -1, scale),\
                             pNMBu.vertex(-1, -1, -1, scale),\
                             pNMBu.vertex(0, phi, 1/phi, scale),\
                             pNMBu.vertex(0, -phi, 1/phi, scale),\
                             pNMBu.vertex(0, phi, -1/phi, scale),\
                             pNMBu.vertex(0, -phi, -1/phi, scale),\
                             pNMBu.vertex(1/phi, 0, phi, scale),\
                             pNMBu.vertex(-1/phi, 0, phi, scale),\
                             pNMBu.vertex(1/phi, 0, -phi, scale),\
                             pNMBu.vertex(-1/phi, 0, -phi, scale),\
                             pNMBu.vertex(phi, 1/phi, 0, scale),\
                             pNMBu.vertex(phi, -1/phi, 0, scale),\
                             pNMBu.vertex(-phi, 1/phi, 0, scale),\
                             pNMBu.vertex(-phi, -1/phi, 0, scale)]
            edges = [(8,0), (8,1), (9,2), (9,4), (10,3), (10,5), (10,8), (11,6), (11,7),\
                     (11,9), (12,0), (12,2), (13,1), (13,4), (13,12), (14,3), (14,6), (15,5),\
                     (15,7), (15,14), (16,0), (16,3), (17,2), (17,6), (17,16), (18,1), (18,5),\
                     (19,4), (19,7), (19,18)]
            faces = [(0,8,10,3,16),(0,12,13,1,8),(8,1,18,5,10),(10,5,15,14,3),(3,14,6,17,16),(16,17,2,12,0),\
                     (4,9,11,7,19),(4,13,12,2,9),(9,2,17,6,11),(11,6,14,15,7),(7,15,5,18,19),(19,18,1,13,4)]
            edges = np.array(edges)
            CoordVertices = np.array(CoordVertices)
            faces = np.array(faces)
        return CoordVertices, edges, faces

    def coords(self,noOutput):
        vID.centertxt("Generation of coordinates",bgc='#007a7a',size='14',weight='bold')
        chrono = pNMBu.timer(); chrono.chrono_start()
        # central atom = "1st shell"
        c = [[0., 0., 0.]]
        self.nAtoms = 1
        self.nAtomsPerShell = [0]
        self.nAtomsPerShell[0] = 1
        indexVertexAtoms = []
        indexEdgeAtoms = []
        indexFaceAtoms = []
        for i in range(1,self.nShell+1):
            # vertices
            nAtoms0 = self.nAtoms
            cshell, E, F = self.MakeVertices(i)
            self.nAtoms += self.nVertices
            self.nAtomsPerShell.append(self.nVertices)
            c.extend(cshell.tolist())
            indexVertexAtoms.extend(range(nAtoms0,self.nAtoms))

            # intermediate atoms on edges e
            nAtoms0 = self.nAtoms
            Rvv = pNMBu.RAB(cshell,E[0,0],E[0,1]) #distance between two vertex atoms
            nAtomsOnEdges = int((Rvv+1e-6) / self.Rnn)-1
            nIntervals = nAtomsOnEdges + 1
            #print("nAtomsOnEdges = ",nAtomsOnEdges)
            coordEdgeAt = []
            for n in range(nAtomsOnEdges):
                for e in E:
                    a = e[0]
                    b = e[1]
                    coordEdgeAt.append(cshell[a]+pNMBu.vector(cshell,a,b)*(n+1) / nIntervals)
            self.nAtomsPerShell[i] += nAtomsOnEdges * len(E) # number of edges x nAtomsOnEdges
            self.nAtoms += nAtomsOnEdges * len(E)
            c.extend(coordEdgeAt)
            indexEdgeAtoms.extend(range(nAtoms0,self.nAtoms))
            #print(c)
            
            # center of each pentagonal facet
            nAtomsOnFaces = 0
            nAtoms0 = self.nAtoms
            coordFaceAt = []
            for f in F:
                nAtomsOnFaces += 1
                coordCenterFace = pNMBu.centerOfGravity(cshell,f)
                #print("coordCenterFace",coordCenterFace)
                self.nAtomsPerShell[i] += 1
                coordFaceAt.append(coordCenterFace)
                # atoms from the center of each pentagonal facet to each of its apex 
                nAtomsOnInternalRadius = i-1
                nIntervals = nAtomsOnInternalRadius+1
                # print(f)
                for indexApex,apex in enumerate(f):
                    if (indexApex == len(f)-1):
                        indexApexPlus1 = 0
                    else:
                        indexApexPlus1 = indexApex+1
                    apexPlus1 = f[indexApexPlus1]
                    for n in range(nAtomsOnInternalRadius):
                        nAtomsOnFaces += 1
                        coordAtomOnApex = coordCenterFace+pNMBu.vectorBetween2Points(coordCenterFace,cshell[apex])*(n+1) / nIntervals
                        coordAtomOnApexPlus1 = coordCenterFace+pNMBu.vectorBetween2Points(coordCenterFace,cshell[apexPlus1])*(n+1) / nIntervals
                        coordFaceAt.append(coordAtomOnApex)
                        RbetweenRadialAtoms = pNMBu.Rbetween2Points(coordAtomOnApex,coordAtomOnApexPlus1)
                        nAtomsBetweenRadialAtoms = int((RbetweenRadialAtoms+1e-6) / self.Rnn)-1
                        nIntervalsRadial = nAtomsBetweenRadialAtoms + 1
                        for k in range(nAtomsBetweenRadialAtoms):
                            nAtomsOnFaces += 1
                            coordFaceAt.append(coordAtomOnApex+pNMBu.vectorBetween2Points(coordAtomOnApex,coordAtomOnApexPlus1)*(k+1) / nIntervalsRadial)
            self.nAtoms += nAtomsOnFaces
            c.extend(coordFaceAt)
            indexFaceAtoms.extend(range(nAtoms0,self.nAtoms))

        print(f"Total number of atoms = {self.nAtoms}")
        print(self.nAtomsPerShell)
        aseObject = ase.Atoms(self.element*self.nAtoms, positions=c)
                
        chrono.chrono_stop(hdelay=False); chrono.chrono_show()
        self.NP = aseObject
    
    def prop(self):
        vID.centertxt("Properties",bgc='#007a7a',size='14',weight='bold')
        print(self)
        pNMBu.plotImageInPropFunction(self.imageFile)
        print("element = ",self.element)
        print("number of vertices = ",self.nVertices)
        print("number of edges = ",self.nEdges)
        print("number of faces = ",self.nFaces)
        print("phi = ",self.phi)
        print(f"intershell factor = {self.interShellF:.2f}")
        print(f"nearest neighbour distance = {self.Rnn:.2f} Å")
        print(f"intershell distance = {self.interShellDistance:.2f} Å")
        print(f"edge length = {self.edgeLength()*0.1:.2f} nm")
        print(f"radius after volume = {pNMBu.RadiusSphereAfterV(self.volume()*1e-3):.2f} nm")
        print(f"radius of the circumscribed sphere = {self.radiusCircumscribedSphere()*0.1:.2f} nm")
        print(f"radius of the inscribed sphere = {self.radiusInscribedSphere()*0.1:.2f} nm")
        print(f"area = {self.area()*1e-2:.1f} nm2")
        print(f"volume = {self.volume()*1e-3:.1f} nm3")
        print("number of atoms per shell = ",self.nAtomsPerShellAnalytic())
        print("total number of atoms = ",self.nAtomsAnalytic())
        print("Dual polyhedron: icosahedron")

    def propPostMake(self,skipSymmetryAnalyzis,thresholdCoreSurface):
        pNMBu.moi(self.NP)
        if not skipSymmetryAnalyzis: pNMBu.MolSym(self.NP)
        [self.vertices,self.simplices,self.neighbors,self.equations],surfaceAtoms =\
            pNMBu.coreSurface(self.NP.get_positions(),thresholdCoreSurface)
        self.NPcs = self.NP.copy()
        self.NPcs.numbers[np.invert(surfaceAtoms)] = 102 #Nobelium, because it has a nice pinkish color in jmol

###########################################################################################################
class cube:
    nFaces = 6
    nEdges = 12
    nVertices = 8
    edgeLengthFfcc = np.sqrt(2)
    edgeLengthFbcc = 2/np.sqrt(3)
    radiusCSF = np.sqrt(3)/2
    radiusISF = 1/2
  
    def __init__(self,
                 crystalStructure='fcc',
                 element='Au',
                 Rnn: float=2.7,
                 nOrder: int=1,
                 postAnalyzis=True,
                 aseView=True,
                 thresholdCoreSurface = 1.,
                 skipSymmetryAnalyzis = False,
                 noOutput = False,
                 calcPropOnly = False,
                ):
        self.crystalStructure = crystalStructure
        self.element = element
        self.Rnn = Rnn
        self.nOrder = nOrder
        self.nAtomsPerEdge = nOrder+1
        self.nAtoms = 0
        self.nAtomsPerShell = [0]
        self.cog = np.array([0., 0., 0.])
        self.imageFile = pNMBu.imageNameWithPathway("cube-C.png")
        if not noOutput: vID.centerTitle(f"{nOrder}x{nOrder}x{nOrder} {self.crystalStructure} cube")
          
        if not noOutput: self.prop()
        if not calcPropOnly:
            self.coords(noOutput)
            if aseView: view(self.NP)
            if postAnalyzis:
                self.propPostMake(skipSymmetryAnalyzis,thresholdCoreSurface)
                if aseView: view(self.NPcs)
          
    def __str__(self):
        return(f"{self.nOrder}x{self.nOrder}x{self.nOrder} fcc cube with Rnn = {self.Rnn}")
    
    def nAtomsfccF(self,i):
        """ returns the number of atoms of an fcc cube of size i x i x i"""
        return 4*i**3 + 6*i*2 + 3*i + 1

    def nAtomsbccF(self,i):
        """ returns the number of atoms of a bcc cube of size i x i x i"""
        return 2*i**3 + 3*i*2 + 3*i
    
    def nAtomsPerShellAnalytic(self):
        n = []
        Sum = 0
        for i in range(self.nOrder+1):
            Sum = sum(n)
            ni = self.nAtomsF(i)
            n.append(ni-Sum)
        return n
    
    def nAtomsPerShellCumulativeAnalytic(self):
        n = []
        Sum = 0
        for i in range(self.nOrder+1):
            Sum = sum(n)
            ni = self.nAtomsF(i)
            n.append(ni)
        return n
    
    def nAtomsfccAnalytic(self):
        n = self.nAtomsfccF(self.nOrder)
        return n
        
    def nAtomsbccAnalytic(self):
        n = self.nAtomsbccF(self.nOrder)
        return n
        
    def edgeLength(self):
        if self.crystalStructure == 'fcc':
            return self.Rnn*self.edgeLengthFfcc*self.nOrder
        elif self.crystalStructure == 'bcc':
            return self.Rnn*self.edgeLengthFbcc*self.nOrder
        
    def latticeConstant(self):
        if self.crystalStructure == 'fcc':
            return self.Rnn*self.edgeLengthFfcc
        elif self.crystalStructure == 'bcc':
            return self.Rnn*self.edgeLengthFbcc
        
    def radiusCircumscribedSphere(self):
        return self.radiusCSF*self.edgeLength()

    def radiusInscribedSphere(self):
        return self.radiusISF*self.edgeLength()

    def area(self):
        el = self.edgeLength()
        return 6 * el**2

    def volume(self):
        el = self.edgeLength()
        return el**3

    # def coords(self):
    #     print(f"Making a {self.nOrder}x{self.nOrder}x{self.nOrder} fcc cube")
    #     surfaces = [(2, 0, 0), (0, 2, 0), (0, 0, 2)]
    #     layers = [self.nOrder, self.nOrder, self.nOrder]
    #     fcc = FaceCenteredCubic(self.element, surfaces, layers, latticeconstant=self.latticeConstant())
    #     natoms = len(fcc.positions)
    #     self.nAtoms=natoms
    #     return fcc

    def coords(self,noOutput):
        if not noOutput: vID.centertxt("Generation of coordinates",bgc='#007a7a',size='14',weight='bold')
        chrono = pNMBu.timer(); chrono.chrono_start()
        if self.crystalStructure == 'fcc':
            cube = bulk(self.element, 'fcc', a=self.latticeConstant(), cubic=True)
        elif self.crystalStructure == 'bcc':
            cube = bulk(self.element, 'bcc', a=self.latticeConstant(), cubic=True)
        view(cube)
        print(f"Now making a {self.nOrder}x{self.nOrder}x{self.nOrder} fcc supercell...")
        M = [[self.nOrder, 0, 0], [0, self.nOrder, 0], [0, 0, self.nOrder]]
        sc=make_supercell(cube, M)
        # now add last layers
        print(f"... and adding the upper layers")
        sc = cut(sc,extend=1.05)
        natoms = len(sc.positions)
        self.nAtoms=natoms
        self.cog = pNMBu.centerOfGravity(sc.get_positions())
        chrono.chrono_stop(hdelay=False); chrono.chrono_show()
        self.NP = sc
        
    def prop(self):
        vID.centertxt("Properties",bgc='#007a7a',size='14',weight='bold')
        print(self)
        pNMBu.plotImageInPropFunction(self.imageFile)
        print("element = ",self.element)
        print("number of vertices = ",self.nVertices)
        print("number of edges = ",self.nEdges)
        print("number of faces = ",self.nFaces)
        print(f"nearest neighbour distance = {self.Rnn:.2f} Å")
        print(f"lattice constant = {self.latticeConstant():.2f} Å")
        print(f"edge length = {self.edgeLength()*0.1:.2f} nm")
        print(f"radius after volume = {pNMBu.RadiusSphereAfterV(self.volume()*1e-3):.2f} nm")
        print(f"radius of the circumscribed sphere = {self.radiusCircumscribedSphere()*0.1:.2f} nm")
        print(f"radius of the inscribed sphere = {self.radiusInscribedSphere()*0.1:.2f} nm")
        print(f"area = {self.area()*1e-2:.1f} nm2")
        print(f"volume = {self.volume()*1e-3:.1f} nm3")
        # print("number of atoms per shell = ",self.nAtomsPerShellAnalytic())
        # print("cumulative number of atoms per shell = ",self.nAtomsPerShellCumulativeAnalytic())
        if self.crystalStructure == 'fcc':
            print("total number of atoms = ",self.nAtomsfccAnalytic())
        elif self.crystalStructure == 'bcc':
            print("total number of atoms = ",self.nAtomsbccAnalytic())
        print("Dual polyhedron: octahedron")

    def propPostMake(self,skipSymmetryAnalyzis,thresholdCoreSurface):
        pNMBu.moi(self.NP)
        if not skipSymmetryAnalyzis: pNMBu.MolSym(self.NP)
        [self.vertices,self.simplices,self.neighbors,self.equations],surfaceAtoms =\
            pNMBu.coreSurface(self.NP.get_positions(),thresholdCoreSurface)
        self.NPcs = self.NP.copy()
        self.NPcs.numbers[np.invert(surfaceAtoms)] = 102 #Nobelium, because it has a nice pinkish color in jmol
