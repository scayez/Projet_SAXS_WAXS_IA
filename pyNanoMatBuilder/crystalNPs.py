import sys, os, pathlib
import re
import math
import numpy as np
import pyNanoMatBuilder.utils as pNMBu
from pyNanoMatBuilder import data
from ase.build import bulk
from ase import io
from ase.visualize import view
from ase.build.supercells import make_supercell
from ase.geometry import cellpar_to_cell

from visualID import  fg, hl, bg
import visualID as vID

class Crystal:
    
    def __init__(self,
                 crystal: str='Au',
                 userDefCif: str=None,
                 shape: str='sphere',
                 size: float=[2,2,2], #nm
                 directionsPPD: np.ndarray=np.array([[1,0,0],[0,1,0],[0,0,1]]),
                 buildPPD: str='xyz',
                 directionWire: float=[0,0,1],
                 refPlaneWire: float=[1,0,0],
                 nRotWire: int=6,
                 surfacesWulff: np.ndarray=None,
                 eSurfacesWulff: np.ndarray=None,
                 sizesWulff: np.ndarray=None,
                 symWulff: bool = True,
                 aseSymPrec: float=1e-4,
                 pbc: bool=False,
                 threshold: float=1e-3,
                 dbFolder: str=data.pNMBvar.dbFolder,
                 postAnalyzis: bool = True,
                 aseView: bool = True,
                 thresholdCoreSurface = 1.,
                 skipSymmetryAnalyzis: bool = False,
                 noOutput: bool = False,
                 calcPropOnly: bool = False,
                ):
        self.dbFolder = dbFolder #database folder that contains cif files
        self.crystal = crystal # see list with the pNMBu.ciflist() command
        self.shape = shape # 'sphere', 'ellipsoid', 'cube', 'wire'
        self.size = size
        self.directionsPPD = directionsPPD
        self.buildPPD = buildPPD
        self.directionWire = directionWire
        self.refPlaneWire = refPlaneWire
        self.nRotWire = nRotWire
        self.surfacesWulff = surfacesWulff
        self.eSurfacesWulff = eSurfacesWulff
        self.sizesWulff = sizesWulff
        self.symWulff = symWulff
        self.aseSymPrec = aseSymPrec
        self.pbc = pbc
        self.threshold = threshold
        self.nAtoms = 0
        self.cif = None
        self.cifname = None
        self.userDefCif = userDefCif
        if self.userDefCif is not None: self.loadExternalCif()

        match self.shape:
            case "sphere":
                self.imageFile = pNMBu.imageNameWithPathway("sphere-C.png")
            case "ellipsoid":
                self.imageFile = pNMBu.imageNameWithPathway("ellipsoid-C.png")
            case "wire":
                self.imageFile = pNMBu.imageNameWithPathway("underConstruction.png")
            case "parallepiped":
                self.imageFile = pNMBu.imageNameWithPathway("underConstruction.png")
            case "Wulff":
                self.imageFile = pNMBu.imageNameWithPathway("underConstruction.png")
            case _:
                sys.exit("Shape {self.shape} is unknown")
        if not noOutput: vID.centerTitle(f"{self.crystal} {self.shape}")

        self.bulk(noOutput)
        if aseView: view(self.cif)
        if not noOutput: self.prop()
        if not calcPropOnly:
            self.makeNP(noOutput)
            if aseView: 
                view(self.sc)
                view(self.NP)
            if postAnalyzis:
                self.propPostMake(skipSymmetryAnalyzis,thresholdCoreSurface)
                if aseView: view(self.NPcs)
          
    def __str__(self):
        return(f"Crystal = {self.crystal} {self.shape}")

    def loadExternalCif(self):
        self.cif = io.read(self.userDefCif)
        path2extCif = pathlib.Path(self.userDefCif)
        if not path2extCif.exists():
            sys.exit(f"file {self.userDefCif} not found. Check the file name or its location")
        cifFile =  open(self.userDefCif, 'r')
        name1 = "_chemical_name_systematic"
        name2 = "_chemical_formula_sum"
        name3 = "_chemical_formula_moiety"
        cifFileLines = cifFile.readlines()
        re_name_systematic = re.compile(name1)
        re_name_sum = re.compile(name2)
        re_name_moiety = re.compile(name3)
        crystal1 = None
        crystal2 = None
        crystal3 = None
        for line in cifFileLines:
            if re_name_systematic.search(line):
                parts = line.split()
                crystal1 = ' '.join(parts[1:])
            if re_name_sum.search(line):
                parts = line.split()
                crystal2 = ' '.join(parts[1:])
            if re_name_moiety.search(line):
                parts = line.split()
                crystal3 = ' '.join(parts[1:])
        cifFile.close()
        if crystal1 is not None:
            self.crystal = crystal1
        elif crystal3 is not None:
            self.crystal = crystal3
        elif crystal2 is not None:
            self.crystal = crystal2
        else: self.crystal = "unknown"

    def bulk(self, noOutput):

        if self.userDefCif is None:
            path2cif = os.path.join(pNMBu.pNMB_location(),self.dbFolder)
            match self.crystal.upper():
                case "NACL":
                    self.cifname = "cod1000041_NaCl.cif"
                case "TIO2 RUTILE":
                    self.cifname = "cod9015662-TiO2-rutile.cif"
                case "TIO2 ANATASE":
                    self.cifname = "cod90159291-TiO2-anatase.cif"
                case "RU":
                    self.cifname = "cod9008513_Ru_hcp.cif"
                case "PT":
                    self.cifname = "cod9012957_Pt_fcc.cif"
                case "AU":
                    self.cifname = "cod9008463_Au_fcc.cif"
                case _:
                    sys.exit(f"The database does not contain bulk parameters for the {self.crystal} crystal.\nPlease provide a cif file")
            self.cif = io.read(os.path.join(path2cif,self.cifname))
        else:
            self.cif = io.read(self.userDefCif)
            path2extCif = pathlib.Path(self.userDefCif)
            self.cifname = pathlib.Path(*path2extCif.parts[-1:])

        
        pNMBu.returnUnitcellData(self)
        if not noOutput: print(f"cif parameters for {self.crystal} found in {self.cifname}")
        return 

    def makeSuperCell(self,noOutput):
        chrono = pNMBu.timer(); chrono.chrono_start()
        if not noOutput: vID.centertxt(f"Making a multiple cell",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        extendSizeByFactor = 1.3
        if (self.shape == 'sphere'):
            # first calculate the size of the supercell
            sphereRadius = self.size[0]
            Ma = int(np.round(extendSizeByFactor * sphereRadius*2*10/self.cif.cell.lengths()[0]))
            Mb = int(np.round(extendSizeByFactor * sphereRadius*2*10/self.cif.cell.lengths()[1]))
            Mc = int(np.round(extendSizeByFactor * sphereRadius*2*10/self.cif.cell.lengths()[2]))
        elif (self.shape == 'ellipsoid' or self.shape == 'supercell' or self.shape == 'parallepiped'):
            # first calculate the size of the supercell
            Ma = int(np.round(extendSizeByFactor * self.size[0]*2*10/self.cif.cell.lengths()[0]))
            Mb = int(np.round(extendSizeByFactor * self.size[1]*2*10/self.cif.cell.lengths()[1]))
            Mc = int(np.round(extendSizeByFactor * self.size[2]*2*10/self.cif.cell.lengths()[2]))
        elif (self.shape == 'wire'):
            maxDim = np.max(self.size)*10*1.5
            Ma = int(np.round(extendSizeByFactor * maxDim/self.cif.cell.lengths()[0]))
            Mb = int(np.round(extendSizeByFactor * maxDim/self.cif.cell.lengths()[1]))
            Mc = int(np.round(extendSizeByFactor * maxDim/self.cif.cell.lengths()[2]))
        elif (self.shape == 'Wulff'):
            if np.argmax(self.sizesWulff) == 1:
                maxDim = self.sizesWulff[0]*10*1.5
            else:
                maxDim = np.max(self.sizesWulff)*10*1.5
            print(f"{maxDim = }")
            Ma = int(np.round(extendSizeByFactor * maxDim/self.cif.cell.lengths()[0]))
            Mb = int(np.round(extendSizeByFactor * maxDim/self.cif.cell.lengths()[1]))
            Mc = int(np.round(extendSizeByFactor * maxDim/self.cif.cell.lengths()[2]))
        #finds the nearest even numbers
        Ma = math.ceil(Ma / 2.) * 2
        Mb = math.ceil(Mb / 2.) * 2
        Mc = math.ceil(Mc / 2.) * 2
        if not noOutput: print(f"Making a {Ma}x{Mb}x{Mc} supercell")
        M = [[Ma, 0, 0], [0, Mb, 0], [0, 0, Mc]]
        sc=make_supercell(self.cif,M)
        # print(cif.cell.cellpar())
        # print(cellpar_to_cell(cif.cell.cellpar()))
        # print(sc.cell.cellpar())
        # print(cellpar_to_cell(sc.cell.cellpar()))
        V = cellpar_to_cell(sc.cell.cellpar())
        if not noOutput: print(f"Center of Mass:", [f"{c:.2f}" for c in sc.get_center_of_mass()]," Å")
        if not noOutput: print("Now translating the supercell")
        sc.translate(-V[0]/2)
        sc.translate(-V[1]/2)
        sc.translate(-V[2]/2)
        if not noOutput: print(f"Center of Mass after translation of the supercell: {sc.get_center_of_mass()} Å")
        self.sc = sc.copy()
        nAtoms=len(self.sc.get_positions())
        if not noOutput: print(f"Total number of atoms = {nAtoms}")
        chrono.chrono_stop(hdelay=False); chrono.chrono_show()
        
    def makeSphere(self,noOutput):
        vID.centertxt(f"Removing atoms to make a sphere",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        chrono = pNMBu.timer(); chrono.chrono_start()
        com = self.sc.get_center_of_mass()
        sphereRadius = self.size[0]
        delAtom = []
        for atomCoord in self.sc.positions:
            delAtom.extend(pNMBu.Rbetween2Points(com,atomCoord)/10 > [sphereRadius])
        self.NP = self.sc.copy()
        del self.NP[delAtom]
        chrono.chrono_stop(hdelay=False); chrono.chrono_show()
                
    def makeEllipsoid(self,noOutput):
        if not noOutput: vID.centertxt(f"Removing atoms to make an ellipsoid",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        chrono = pNMBu.timer(); chrono.chrono_start()
        com = self.sc.get_center_of_mass()
        size = np.array(self.size)*10 #nm to angstrom
        def outside(coord,com,size):
            return (coord[0]-com[0])**2/(size[0])**2+(coord[1]-com[1])**2/(size[1])**2+(coord[2]-com[2])**2/(size[2])**2
        delAtom = []
        for atom in self.sc.positions:
            delAtom.extend([outside(atom,com,size) > 1])
        self.NP = self.sc.copy()
        del self.NP[delAtom]
        chrono.chrono_stop(hdelay=False); chrono.chrono_show()

    def makeWire(self,noOutput):
        if not noOutput: vID.centertxt(f"Removing atoms to make a wire",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        chrono = pNMBu.timer(); chrono.chrono_start()
        if self.refPlaneWire is None: self.refPlaneWire = pNMBu.returnPlaneParallel2Line(self.directionWire,[1,0,0],debug=False)
        normal = pNMBu.normal2MillerPlane(self,self.refPlaneWire,printN=True)
        trPlanes = pNMBu.planeRotation(self,normal,self.directionWire,self.nRotWire)
        for i,p in enumerate(trPlanes):
            trPlanes[i] = pNMBu.normV(p)
        radius = 10*self.size[0]/2
        tradius = np.full((self.nRotWire,1),-radius)
        trPlanes = np.append(trPlanes,tradius,axis=1)
        if not self.pbc:
            halfLength = 10*self.size[1]/2
            ptop = np.append(pNMBu.normV(self.directionWire),-halfLength)
            pbottom = np.append(-pNMBu.normV(self.directionWire),-halfLength)
            trPlanes = np.append(trPlanes,ptop)
            trPlanes = np.append(trPlanes,pbottom)
            trPlanes = np.reshape(trPlanes,(self.nRotWire+2,4))
        AtomsAbovePlanes = pNMBu.truncateAbovePlanes(trPlanes,self.sc.positions,eps=self.threshold)
        self.NP = self.sc.copy()
        del self.NP[AtomsAbovePlanes]
        nAtoms = self.NP.get_global_number_of_atoms()
        self.trPlanes = trPlanes
        if not noOutput: vID.centertxt(f"Nanowire moved to the center of the unitcell",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        self.NP.center()
        chrono.chrono_stop(hdelay=False); chrono.chrono_show()

    def makeParallelepiped(self,noOutput):
        if not noOutput: vID.centertxt(f"Removing atoms to make a parallelepiped",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        chrono = pNMBu.timer(); chrono.chrono_start()
        if self.buildPPD == "xyz":
            trPlanes = self.directionsPPD
        else:
            if not noOutput: printN = True
            else: printN = False
            normal = []
            for d in self.directionsPPD:
                normal.append(pNMBu.normal2MillerPlane(self,d,printN=True))
            trPlanes = pNMBu.lattice_cart(self,normal,Bravais2cart=True,printV=True)
        for i,p in enumerate(trPlanes): trPlanes[i] = pNMBu.normV(p)
        # 6 planes defined to cut between 
        # [-a/2 direction, a/2 direction], [-b/2 direction, b/2 direction], [-c/2 direction, c/2 direction]
        size = -np.array(self.size)*10/2 #nm!
        size = np.append(size,size,axis=0)
        trPlanes = np.append(trPlanes,-trPlanes,axis=0)
        trPlanes = np.append(trPlanes,size.reshape(6,1),axis=1)
        AtomsAbovePlanes = pNMBu.truncateAbovePlanes(trPlanes,self.sc.positions,eps=self.threshold,debug=False)
        self.NP = self.sc.copy()
        del self.NP[AtomsAbovePlanes]
        nAtoms = self.NP.get_global_number_of_atoms()
        self.NP.center()
        self.trPlanes = trPlanes
        chrono.chrono_stop(hdelay=False); chrono.chrono_show()

    def makeWulff(self,noOutput):
        if not noOutput: vID.centertxt(f"Calculating truncation distances",bgc='#cbcbcb',size='12',fgc='b',weight='bold')
        chrono = pNMBu.timer(); chrono.chrono_start()
        trPlanes = []
        if self.eSurfacesWulff is None: sizes = []
        if self.eSurfacesWulff is not None: 
            sizes = []
            eSurf = []
        for i,p in enumerate(self.surfacesWulff):
            if self.symWulff:
                symP = self.ucSG.equivalent_lattice_points(p)
                normal = []
                for sp in symP:
                    normal.append(pNMBu.normal2MillerPlane(self,sp,printN=True))
                trPlanes += list(normal)
                if self.eSurfacesWulff is None: sizes.append(len(symP)*[self.sizesWulff[i]])
                if self.eSurfacesWulff is not None: eSurf += (len(symP)*[self.eSurfacesWulff[i]])
            else:
                trPlanes.append(pNMBu.normal2MillerPlane(self,p,printN=True))
                if self.eSurfacesWulff is None: sizes.append(self.sizesWulff[i])
                if self.eSurfacesWulff is not None: eSurf.append(self.eSurfacesWulff[i])
        trPlanes = np.array(trPlanes)
        trPlanes = pNMBu.lattice_cart(self,trPlanes,Bravais2cart=True,printV=True)
        for i,p in enumerate(trPlanes): trPlanes[i] = pNMBu.normV(p)
        # print(trPlanes.tolist())
        if self.eSurfacesWulff is None: 
            sizes = -np.array(sizes)*10/2
            trPlanes = np.append(trPlanes,sizes.reshape(len(trPlanes),1),axis=1)
        else:
            mostStableE = min(eSurf)
            for i, e in enumerate(eSurf):
                sizes.append(-self.sizesWulff[0]*10*e/2/mostStableE)
            sizes = np.array(sizes)
            trPlanes = np.append(trPlanes,sizes.reshape(len(trPlanes),1),axis=1)
        # print(trPlanes)
        AtomsAbovePlanes = pNMBu.truncateAbovePlanes(trPlanes,self.sc.positions,allP=False,eps=self.threshold,debug=False)
        self.NP = self.sc.copy()
        del self.NP[AtomsAbovePlanes]
        nAtoms = self.NP.get_global_number_of_atoms()
        self.NP.center()
        self.trPlanes = trPlanes
        chrono.chrono_stop(hdelay=False); chrono.chrono_show()

    def makeNP(self,noOutput):
        import os
        if not noOutput: vID.centertxt("Builder",bgc='#007a7a',size='14',weight='bold')
        if self.size is None:
            self.length = [2,2,2]
            if not noOutput: print(f"length parameter set up as = {self.size} nm")
        if self.shape == "sphere":
            if not noOutput: print(f"Sphere radius = {self.size[0]} nm")
            self.makeSuperCell(noOutput)
            self.makeSphere(noOutput)
        elif self.shape == "ellipsoid":
            if not noOutput: print(f"Ellipsoid radii = {self.size} nm")
            self.makeSuperCell(noOutput)
            self.makeEllipsoid(noOutput)
        elif self.shape == "parallepiped":
            if not noOutput: print(f"Parallepiped side length = {self.size} nm, directions = {list(self.directionsPPD)}")
            self.makeSuperCell(noOutput)
            self.makeParallelepiped(noOutput)
        elif self.shape == "supercell":
            if not noOutput: print(f"Supercell side length = {self.size} nm")
            if len(self.size) != 3: sys.exit("Please enter lengths along a,b and c axis, i.e. size=[l_a,l_b,l_c]")
            self.makeSuperCell(noOutput)
        elif self.shape == "wire":
            if not noOutput: print(f"Wire in the {self.directionWire} directionWire. Length x width = {self.size[1]} x {self.size[0]} nm")
            if not noOutput: print(f"Reference plane = {self.refPlaneWire}, {self.nRotWire}-th order rotation around {self.directionWire}")
            if not pNMBu.isPlaneParrallel2Line(self.refPlaneWire, self.directionWire):
                print(f"{fg.RED}Warning! The reference truncation plane is not parallel to {self.directionWire}. Are you sure?{fg.OFF}")
                suggestedPlane = pNMBu.returnPlaneParallel2Line(self.directionWire)
                print(f"Among other possibilities, you can try {suggestedPlane}")
            else:
                if not noOutput: print(f"{fg.GREEN}The reference truncation plane is parallel to {self.directionWire}{fg.OFF}")
            self.makeSuperCell(noOutput)
            self.makeWire(noOutput)
        elif self.shape == "Wulff":
            if self.surfacesWulff == None:
                sys.exit("Wulff construction requested, but no planes were given. Define them with the 'surfacesWulff' variable")
            if self.eSurfacesWulff == None and self.sizesWulff == None: 
                sys.exit("Either 'eSurfacesWulff' or 'sizesWulff' variables must be set up")
            if len(self.surfacesWulff) != len(self.eSurfacesWulff) and len(self.surfacesWulff) != len(self.sizesWulff):
                sys.exit("'surfacesWulff' and ('eSurfacesWulff' or 'sizesWulff') lists have different dimensions")
            self.makeSuperCell(noOutput)
            self.makeWulff(noOutput)
        self.nAtoms=len(self.NP.get_positions())
        self.cog = self.NP.get_center_of_mass()
        if not noOutput: print(f"Total number of atoms = {self.nAtoms}")

    def prop(self):
        print(self)
        pNMBu.plotImageInPropFunction(self.imageFile)
        vID.centertxt("Unit cell properties",bgc='#007a7a',size='14',weight='bold')
        pNMBu.print_ase_unitcell(self)

    def propPostMake(self,skipSymmetryAnalyzis,thresholdCoreSurface):
        pNMBu.moi(self.NP)
        if not skipSymmetryAnalyzis: pNMBu.MolSym(self.NP)
        [self.vertices,self.simplices,self.neighbors,self.equations],surfaceAtoms =\
            pNMBu.coreSurface(self.NP.get_positions(),thresholdCoreSurface)
        self.NPcs = self.NP.copy()
        self.NPcs.numbers[np.invert(surfaceAtoms)] = 102 #Nobelium, because it has a nice pinkish color in jmol
