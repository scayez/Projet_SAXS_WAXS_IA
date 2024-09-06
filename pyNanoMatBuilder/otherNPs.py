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
from pyNanoMatBuilder import johnsonNPs as jNP

###########################################################################################################
class fcctpt:
    nFaces = 8
    nEdges = 9
    nVertices = 9
    edgeLengthF = 2

    def __init__(self,
                 element: str='Au',
                 Rnn: float=2.7,
                 nLayerTd: int=1,
                 nLayer: int = 3, #number of layers per pyramid -> total number of layers = nLayer*2 + twinning plane
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
        self.tbpprop = jNP.fcctbp(self.element,self.Rnn,self.nLayerTd,noOutput=True,calcPropOnly=True)
        self.nLayertbp = 2*self.nLayerTd - 1
        self.nLayer = nLayer*2+1 
        self.nAtoms = 0
        self.interLayerDistance = self.tbpprop.interLayerDistance
        self.nAtomsPerEdge = self.nLayerTd+1
        self.cog = np.array([0., 0., 0.])
        if self.nLayer > self.nLayertbp:
            sys.exit(f"Number of layers of the triangular platelet ({self.nLayer}) cannot be > to the total number of layers of the trigonal bipyramid {self.nLayertbp}")
        self.imageFile = pNMBu.imageNameWithPathway("tpt-C.png")
        if not noOutput: vID.centerTitle(f"fcc triangulat platelet with {nLayer*2+1} remaining shells, made from a trigonal bipyramid with {nLayerTd} shells per pyramid")

        if not noOutput: self.prop()
        if not calcPropOnly:
            self.coords(noOutput)
            if aseView: view(self.NP)
            if postAnalyzis:
                self.propPostMake(skipSymmetryAnalyzis,thresholdCoreSurface)
                if aseView: view(self.NPcs)

    def __str__(self):
        return(f"Truncated fcc double tetrahedron with {self.nLayer} layer(s) and Rnn = {self.Rnn}") 

    def edgeLength(self):
        return self.tbpprop.edgeLength()

    def coords(self,noOutput):
        if not noOutput: vID.centertxt("Generation of coordinates",bgc='#007a7a',size='14',weight='bold')
        chrono = pNMBu.timer(); chrono.chrono_start()
        vID.centertxt("Generation of the coordinates of the trigonal bipyramid, based on the fcc tetrahedron",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        tbp = jNP.fcctbp(self.element,self.Rnn,self.nLayerTd+1,postAnalyzis=False,noOutput=True)
        self.NP0 = tbp.NP.copy()
        asetpt = tbp.NP
        nAtoms = asetpt.get_global_number_of_atoms()
        # cog = pNMBu.centerOfGravity(asetbp.get_positions())
        # print("cog = ",cog)
        vID.centertxt("Truncation of the trigonal bipyramid",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        print("Now calculating the coordinates of the twin plane (defined by atoms 0,1,2)")
        coordTwPVertices = asetpt.get_positions()[[0,1,2]]
        twinningPlane = pNMBu.hklPlaneFitting(coordTwPVertices)
        twinningPlane = pNMBu.normalizePlane(twinningPlane)
        print("Now calculating the coordinates of the truncation planes")
        truncationPlane1 = twinningPlane.copy()
        truncationPlane1[3] = -(self.nLayer-1)*self.interLayerDistance/2
        print(f"signed distance between truncation plane 1 and origin = {pNMBu.Pt2planeSignedDistance(truncationPlane1,[0,0,0]):.2f}")
        truncationPlane2 = -twinningPlane.copy()
        truncationPlane2[3] = -(self.nLayer-1)*self.interLayerDistance/2
        print(f"signed distance between truncation plane 2 and origin = {pNMBu.Pt2planeSignedDistance(truncationPlane1,[0,0,0]):.2f}")
        trPlanes = [truncationPlane1, truncationPlane2]
        AtomsAbovePlanes = pNMBu.truncateAboveEachPlane(trPlanes,asetpt.get_positions())
        del asetpt[AtomsAbovePlanes]
        chrono.chrono_stop(hdelay=False); chrono.chrono_show()
        nAtoms = asetpt.get_global_number_of_atoms()
        print(f"Total number of atoms = {nAtoms}")
        self.NP = asetpt

    def prop(self):
        print(self)
        pNMBu.plotImageInPropFunction(self.imageFile)
        print("element = ",self.element)
        print("number of vertices = ",self.nVertices)
        print("number of edges = ",self.nEdges)
        print("number of faces = ",self.nFaces)
        print(f"nearest neighbour distance = {self.Rnn:.2f} Å")
        print(f"edge length = {self.edgeLength()*0.1:.2f} nm")
        print(f"number of atoms per edge at the twin boundary = {self.nAtomsPerEdge}")
        print(f"inter-layer distance = {self.interLayerDistance:.2f} Å")
        print(f"height of the platelet = {self.interLayerDistance*(self.nLayer-1)*0.1:.2f} nm")
        # print(f"area = {6*self.Td.area()/4*1e-2:.1f} nm2")
        # print(f"volume = {2*self.Td.volume()*1e-3:.1f} nm3")
        print(f"face-vertex-edge angle in Td = {self.tbpprop.fveAngle:.1f}°")
        print(f"face-edge-face (dihedral) angle in Td = {self.tbpprop.fefAngle:.1f}°")
        print(f"vertex-center-vertex (tetrahedral bond) angle in Td = {self.tbpprop.vcvAngle:.1f}°")
        print(f"coordinates of the center of gravity = {self.cog}")

    def propPostMake(self,skipSymmetryAnalyzis,thresholdCoreSurface):
        pNMBu.moi(self.NP)
        if not skipSymmetryAnalyzis: pNMBu.MolSym(self.NP)
        [self.vertices,self.simplices,self.neighbors,self.equations],surfaceAtoms =\
            pNMBu.coreSurface(self.NP.get_positions(),thresholdCoreSurface)
        self.NPcs = self.NP.copy()
        self.NPcs.numbers[np.invert(surfaceAtoms)] = 102 #Nobelium, because it has a nice pinkish color in jmol
