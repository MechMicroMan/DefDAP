# Load packages
import numpy as np
import matplotlib.pyplot as plt

from DataAnalysisUtilities.defdap.quat import Quat
import DataAnalysisUtilities.defdap.fepx as fepx

# Load mesh
revLoadSim = fepx.Mesh(
    "dicMesh",
    meshDir="../../FEPX/Meshes/dicMeshb15/",
    dataDir="../../runs/fepx/dicMesh_b15_reverseLoad/grip_slide_BCs/combined1/"
)

# Plot stress strain
revLoadSim.surfaces[2].plotStressStrain();

# Load simulaton data
revLoadSim.loadSimData(
    [
        "nodePos",
        "angle",
        "hardness",
        "eqvStrain",
        "eqvPlasticStrain",
        "stress",
        "shearRate"
    ],
    numProcs=8,
    numFrames=30,
    numSlipSys=12
)
revLoadSim.loadMeshElStatsData("dicMesh")

# Calculate strain from displacement
# Calculate displacement
s1, s2, s3 = revLoadSim.simData['nodePos'].shape

disp = np.zeros((s1, s2, s3))

for i in range(s3 - 1):
    disp[:, :, i + 1] = revLoadSim.simData['nodePos'][:, :, i + 1] - revLoadSim.simData['nodePos'][:, :, 0]

revLoadSim.createSimData("displacement", disp)

# Calculate Green Lagrange strain
# eij=12(ui,j+uj,i+uk,iuk,j)
# where derivatives are wrt to reference/undeformed coordinates.
revLoadSim.calcGradient("displacement", "displacementGrad")
# Gradient wrt to reference/undeformed coordinates

dispGrad = revLoadSim.simData['displacementGrad']
totalStrain = np.zeros((revLoadSim.numNodes, 6, revLoadSim.numFrames + 1))

comp = 0
for i in range(3):
    for j in range(3):
        if j >= i:
            totalStrain[:, comp, :] = dispGrad[:, i*3 + j, :] + dispGrad[:, j*3 + i, :]
            for k in range(3):
                totalStrain[:, comp, :] += dispGrad[:, k*3 + i, :] * dispGrad[:, k*3 + j, :]

            comp += 1
totalStrain /= 2

revLoadSim.createSimData("greenStrain", totalStrain)

# Calculate misorientation
revLoadSim.calcMisori(-1)

# Get times from displacment file
times = np.loadtxt(revLoadSim.dataDir + "../1_8T.disp", usecols=(0,), skiprows=2, dtype=int)
times = (0,) + tuple(times)
print(times)

# Export vtk for paraview/visit
revLoadSim.writeVTU(
    "manyData_deformedCoords",
    -1,
    [
        "elStats",
        "greenStrain",
        "nodePos",
        "displacement",
        "hardness",
        "eqvPlasticStrain",
        "eqvStrain",
        "stress",
        "angle",
        "misOri",
        "avMisOri"
    ],
    times=times,
    useInitialNodePos=False
)

# Plot surface data
revLoadSim.surfaces[0].plotSimData(
    "greenStrain", component=0, frameNum=10, plotType="image",
    plotGBs=True, invertData=False, label="Axial strain (e11)",
    vmin=-0.06, vmax=0.06, cmap="seismic"
)

# Calculate grainaverage
revLoadSim.calcGrainAverage("eqvPlasticStrain", "eqvPlasticStrainGrainAve")

# Plot IPF
Quat.plotIPF(revLoadSim.simData['avOri'][:, 10], np.array((1, 0, 0)), "cubic",
             c=revLoadSim.simData['eqvPlasticStrainGrainAve'][:, 10],
             marker='o', vmin=0, vmax=0.03)
