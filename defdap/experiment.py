# Copyright 2021 Mechanics of Microstructures Group
#    at The University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from defdap import hrdic, ebsd

from defdap import defaults
from defdap.plotting import MapPlot
from defdap.utils import reportProgress


class Experiment(object):
    """
    Class to encapsulate deformation experiment data.
    """

    def __init__(self, label):
        """Initialise class.

        """
        # Initialise variables
        self.label = label
        self.dicMaps = []
        self.ebsdMaps = []

    @reportProgress("loading dic data")
    def loadDicData(self,
                    fileDir,
                    fileNames,
                    dataType=None,
                    xMin=None, xMax=None, yMin=None, yMax=None, updateHomogPoints=False,
                    scale=None,
                    img_path = None, window_size=None):
        """Load DIC data from files.

        Parameters
        ----------
        fileDir : str
            Path to files.
        fileNames : str
            Name of files including extension.
        dataType : str,  {'DavisText'}
            Type of data file.

        """

        for i, fileName in enumerate(fileNames):

            yield i / len(fileNames)  # Report progress

            self.dicMaps.append(hrdic.Map(fileDir, fileName, dataType))

            self.dicMaps[-1].setCrop(xMin, xMax, yMin, yMax, updateHomogPoints)

            if scale is not None:
                self.dicMaps[-1].setScale(scale)

            if img_path and window_size is not None:
                self.dicMaps[-1].set_pattern(img_path, window_size)

    def loadEbsdData(self,
                     fileDir,
                     fileNames,
                     dataType=None,
                     boundDef=None,
                     minGrainSize=None):
        """Load EBSD data from files.

        Parameters
        ----------
        fileDir : str
            Path to files.
        fileNames : str
            Name of files excluding extension.
        dataType : str, {'OxfordBinary', 'OxfordText'}
            Format of EBSD data file.

        """

        for fileName in fileNames:
            self.ebsdMaps.append(ebsd.Map(fileDir + '/' + fileName, dataType))

        if boundDef is not None:
            self.ebsdMaps[-1].findBoundaries(boundDef)
            if minGrainSize is not None:
                self.ebsdMaps[-1].findGrains(minGrainSize)

    def __str__(self):
        """Print a summary of data in the experiment



        *** Consider refactoring more generally to: 

                        Frame
                        0           1
                        ---------------------------

        Increments  0 |  ebsd.Map
                    1 |              hrdic.Map
                    2 |              hrdic.Map
                    3 |              hrdic.Map
        """

        string = 'Experiment name: {}\n\n'.format(self.label)

        string += 'DIC Maps\n--------\n'

        for i, m in enumerate(self.dicMaps):
            string += 'Map {0}:\tMean Exx = {1:.2f} %\n'.format(i, np.mean(m.crop(m.data.e[0,0]))*100)

        try:
            self.dicMaps[0].checkEbsdLinked()
            string += '\n↕ Linked\n'
        except:
            string += '\n↕ NOT linked\n'

        string += '\nEBSD Map\n---------\n'

        for i, m in enumerate(self.ebsdMaps):
            string += 'Map {0}:\tdimensions: {1} x {2} pixels, step size: {3} um\n'.\
                format(i, m.xDim, m.yDim, m.step_size)

        return string


class Increment(object):
    """
    Class to encapsulate deformation increment. 
    Each increment corresponds to a different material state, i.e. deformation, temperature.
    """


class Frame(object):
    """
    Class to encapsulate reference frames.

    *** Warping logic, homologous points
    """







