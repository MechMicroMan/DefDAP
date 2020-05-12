# Copyright 2020 Mechanics of Microstructures Group
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

from sklearn.cluster import MeanShift
from scipy.stats import linregress
import pandas as pd

from defdap.plotting import Plot, GrainPlot, LineSlice


class GrainInspector:
    """
    Class containing the interactive grain inspector tool for slip trace analysis
    and relative displacement ratio analysis.

    """
    def __init__(self, currMap, vmax=0.1):
        # Initialise some values
        self.grainID = 0
        self.currMap = currMap
        self.currEBSDMap = self.currMap.ebsdMap
        self.currDICGrain = self.currMap[self.grainID]
        self.currEBSDGrain = self.currDICGrain.ebsdGrain
        self.vmax = vmax
        
        # Draw the figure
        self.draw()

    def draw(self):
        """ Draw the main window, buttons, text boxes and axes.

        """
        # Plot window
        self.plot = Plot(ax=None, makeInteractive=True, figsize=(14,8), title='Grain Inspector')
        
        # Buttons
        self.plot.addButton('Save\nLine', self.saveLine, (0.73, 0.48, 0.05, 0.04))
        self.plot.addButton('Previous\nGrain', lambda e, p: self.gotoGrain(self.grainID-1, p), (0.73, 0.94, 0.05, 0.04))
        self.plot.addButton('Next\nGrain', lambda e, p: self.gotoGrain(self.grainID+1, p), (0.79, 0.94, 0.05, 0.04))
        self.plot.addButton('Run\nAll STA', self.batchRunSTA, (0.81, 0.02, 0.1, 0.04))
        self.plot.addButton('Clear\nAll Lines', self.clearAllLines, (0.89, 0.48, 0.05, 0.04))

        # Text boxes
        self.plot.addTextBox(label='Go to \ngrain ID:', loc=(0.9, 0.94, 0.05, 0.04), submitHandler=self.gotoGrain)
        self.plot.addTextBox(label='Remove\nID:', loc=(0.83, 0.48, 0.05, 0.04), submitHandler=self.removeLine)
        self.RDRGroupBox = self.plot.addTextBox(label='Run RDR\non group:',
                                                loc=(0.78, 0.07, 0.05, 0.04), submitHandler=self.runRDRGroup)

        # Axes
        self.maxShearAx = self.plot.addAxes((0.05, 0.4, 0.65, 0.55))
        self.slipTraceAx = self.plot.addAxes((0.2, 0.05, 0.6, 0.3))
        self.unitCellAx = self.plot.addAxes((0.05, 0.055, 0.15, 0.3), proj='3d')
        self.grainInfoAx = self.plot.addAxes((0.73, 0.86, 0.25, 0.06))
        self.lineInfoAx = self.plot.addAxes((0.73, 0.55, 0.25, 0.3))
        self.groupsInfoAx = self.plot.addAxes((0.73, 0.15, 0.25, 0.3))
        self.grainPlot = self.currMap[self.grainID].plotMaxShear(fig=self.plot.fig, ax=self.maxShearAx, 
                                                                 vmax=self.vmax, plotScaleBar=True, plotColourBar=True)
        self.plot.ax.axis('off')
        
        # Draw the stuff that will need to be redrawn often in a seperate function
        self.redraw()

    def gotoGrain(self, event, plot):
        """ Go to a specified grain ID.

        Parameters
        ----------
        event: int
            Grain ID to go to.

        """
        ## Go to grain ID specified in event
        self.grainID=int(event)
        self.grainPlot.arrow=None
        self.currDICGrain = self.currMap[self.grainID]
        self.currEBSDGrain = self.currDICGrain.ebsdGrain
        self.redraw()

    def saveLine(self, event, plot):
        """  Save the start point, end point and angle of drawn line into the grain.

        Parameters
        ----------
        event: numpy.ndarray
            Start x, start y, end x, end y point of line passed from drawn line.

        """
        # Get angle of lines
        lineAngle = 90-np.rad2deg(np.arctan2(self.drawnLine.points[3]-self.drawnLine.points[1], 
                                              self.drawnLine.points[2]-self.drawnLine.points[0]))
        if lineAngle > 180: lineAngle -= 180
        elif lineAngle < 0: lineAngle += 180
        #lineAngle += self.currMap.ebsdTransform.rotation*-180/np.pi
            
        # Save drawn line to the DIC grain
        self.currDICGrain.pointsList.append([self.drawnLine.points, lineAngle, -1])
        
        # Group lines and redraw
        self.groupLines()
        self.redraw()
        
    def groupLines(self):
        """
        Group the lines drawn in the current grain item using a mean shift algorithm,
        save the average angle and detect the active slip planes.

        """
        angles = [x[1] for x in self.currDICGrain.pointsList]
        # For single line, don't group
        if len(angles) == 1:
            self.currDICGrain.pointsList[0][2]=0
            self.currDICGrain.groupsList = [[0, angles[0], 0, 0, 0]]
        else:
            # Run clustering algorithm for >1 line
            ms = MeanShift(bandwidth=10).fit(np.matrix([range(len(angles)), angles]).transpose())
            
            # Add group ID for each line to the points list
            for i, label in enumerate(ms.labels_):
                self.currDICGrain.pointsList[i][2] = label
            
            # Make array of groups with mean angle
            self.currDICGrain.groupsList = []
            for i in range(np.max(ms.labels_+1)):
                self.currDICGrain.groupsList.append([i, ms.cluster_centers_[i][1], 0, 0, 0])
                
        # Detect active slip systems in each group
        for group in self.currDICGrain.groupsList:
            activePlanes = []
            deviation = []
            experimentalAngle = group[1]
            for idx, theoreticalAngle in enumerate(np.rad2deg(self.currEBSDGrain.slipTraceAngles)):
                if theoreticalAngle-5 < experimentalAngle < theoreticalAngle+5:
                    activePlanes.append(idx)
                    deviation.append(experimentalAngle-theoreticalAngle)
            group[2] = activePlanes
            group[3] = deviation
            
    def clearAllLines(self, event, plot):
        """ Clear all lines in a given grain.

        """

        self.currDICGrain.pointsList = []
        self.currDICGrain.groupsList = []
        self.redraw()

    def removeLine(self, event, plot):
        """  Remove single line [runs after submitting a text box].

        Parameters
        ----------
        event: int
            Line ID to remove.

        """
        ## Remove single line
        del self.currDICGrain.pointsList[int(event)]
        self.redraw()

    def redraw(self):
        """
        Draw items which need to be redrawn often (i.e. when changing grain ID).

        """

        # Plot max shear for grain
        self.maxShearAx.clear()
        grainPlot = self.currMap[self.grainID].plotMaxShear(
            fig=self.plot.fig,ax=self.maxShearAx, vmax=self.vmax, plotColourBar=False, plotScaleBar=True)

        # Draw slip traces
        self.slipTraceAx.clear()
        self.slipTraceAx.set_aspect('equal', 'box')
        slipPlot = GrainPlot(fig=self.plot.fig, callingGrain=self.currMap[self.grainID], ax=self.slipTraceAx)
        traces = slipPlot.addSlipTraces(topOnly=True)
        self.slipTraceAx.axis('off')
        
        # Draw slip bands
        bands = [elem[1] for elem in self.currDICGrain.groupsList]
        if self.currDICGrain.groupsList != None:
            slipPlot.addSlipBands(topOnly=True, angles=list(np.deg2rad(bands)))
        
        # Draw unit cell
        self.unitCellAx.clear()
        self.currEBSDGrain.plotUnitCell(fig=self.plot.fig, ax=self.unitCellAx)
        
        # Write grain info text
        self.grainInfoAx.clear()
        self.grainInfoAx.axis('off')
        grainInfoText = 'Grain ID: {0} / {1}\n'.format(self.grainID, len(self.currMap.grainList))
        grainInfoText += 'Min: {0:.1f} %     Mean:{1:.1f} %     Max: {2:.1f} %'.format(
            np.min(self.currDICGrain.maxShearList)*100,
            np.mean(self.currDICGrain.maxShearList)*100,
            np.max(self.currDICGrain.maxShearList)*100)
        self.plot.addText(self.grainInfoAx, 0, 1, grainInfoText,  va='top', ha='left', fontsize=10)
        
        # Detect lines
        self.drawnLine = LineSlice(ax=self.maxShearAx, fig=self.plot.fig, action=self.grainPlot.addArrow)

        # Write lines text and draw lines
        linesTxt = 'List of lines\n\nLineID  x0     y0     x1     y1     Angle  Group\n'

        if self.currDICGrain.pointsList != []:
            for idx, points in enumerate(self.currDICGrain.pointsList):
                linesTxt += '{0}          {1:.1f}   {2:.1f}    {3:.1f}   {4:.1f}   {5:.1f}   {6}\n'.format(idx,
                                    points[0][0], points[0][1], points[0][2], points[0][3], points[1], points[2])
                self.grainPlot.addArrow(startEnd=points[0], clearPrev=False, persistent=True, label=idx)
        
        self.lineInfoAx.clear()
        self.lineInfoAx.axis('off')
        self.plot.addText(self.lineInfoAx, 0, 1, linesTxt, va='top', fontsize=10)
        
        # Write groups info text
        groupsTxt = 'List of groups\n\nGroupID    Angle      System      Dev     RDR\n'
        if self.currDICGrain.groupsList != []:
            for idx, group in enumerate(self.currDICGrain.groupsList):
                groupsTxt += '{0}                {1:.1f}      {2}      {3}      {4:.2f}\n'.format(
                    idx, group[1], group[2], np.round(group[3],3), group[4])

        self.groupsInfoAx.clear()
        self.groupsInfoAx.axis('off')
        self.plot.addText(self.groupsInfoAx, 0, 1, groupsTxt, va='top', fontsize=10)

    def runRDRGroup(self, event, plot):
        """  Run RDR on a specified group, upon submitting a text box.

        Parameters
        ----------
        event: int
            Group ID specified from text box.

        """
        ## Run RDR for group of lines
        if event != '':
            self.calcRDR(grain = self.currDICGrain, group=int(event))
            self.RDRGroupBox.set_val('')
        
    def batchRunSTA(self, event, plot):
        """  Run slip trace analysis on all grains which hve slip trace lines drawn.

        """

        # Print header
        print("Grain\tEul1\tEul2\tEul3\tMaxSF\tGroup\tAngle\tSystem\tDev\RDR")
        
        # Print information for each grain
        for idx, grain in enumerate(self.currMap):
            if grain.pointsList != []:
                for group in grain.groupsList:
                    maxSF = np.max([item for sublist in grain.ebsdGrain.averageSchmidFactors for item in sublist])
                    eulers = self.currEBSDGrain.refOri.eulerAngles()*180/np.pi
                    text = '{0}\t{1:.1f}\t{2:.1f}\t{3:.1f}\t{4:.3f}\t'.format(
                                                    idx, eulers[0], eulers[1], eulers[2], maxSF)
                    text += '{0}\t{1:.1f}\t{2}\t{3}\t{4:.2f}'.format(
                                                    group[0], group[1], group[2], np.round(group[3],3), group[4])
                    print(text)

    def calcRDR(self, grain, group, showPlot=True, length=2.5):
        """ Calculates the relative displacement ratio for a given grain and group.

        Parameters
        ----------
        grain: int
            DIC grain ID to run RDR on.
        group: int
            group ID to run RDR on.
        showPlot: bool
            if True, show plot window.
        length: int
            length of perpendicular lines used for RDR.

        """
        
        ulist=[]; vlist=[]; allxlist = []; allylist = [];      

        # Get all lines belonging to group
        points = []
        for point in grain.pointsList:
            if point[2] == group:
                points.append(point[0])

        for point in points:
            x0=point[0];  y0=point[1]; x1=point[2];   y1=point[3];
            grad = (y1-y0)/(x1-x0)
            invgrad = -1/grad
            profile_length = np.sqrt((y1-y0)**2+(x1-x0)**2)
            num = np.round(profile_length*2)
            
            ### Calculate positions for each point along slip trace line (x,y)
            x, y = np.round(np.linspace(x0, x1, int(num))), np.round(np.linspace(y0, y1, int(num)))
            df = pd.DataFrame({'x':x, 'y':y}).drop_duplicates()
            x,y = df['x'].values.tolist(),df['y'].values.tolist()

            ## Calculate deviation from (0,0) for points along line with angle perpendicular to slip line (xnew,ynew)
            x0new = np.sqrt(length/(invgrad**2+1))*np.sign(grad)
            y0new = -np.sqrt(length/(1/invgrad**2+1))
            x1new = -np.sqrt(length/(invgrad**2+1))*np.sign(grad)
            y1new = np.sqrt(length/(1/invgrad**2+1))
            profile_length=np.sqrt((y1new-y0new)**2+(x1new-x0new)**2)
            num = np.round(profile_length)
            xnew, ynew = np.linspace(x0new, x1new, int(num)), np.linspace(y0new, y1new, int(num))
            xnew, ynew = np.around(xnew).astype(int), np.around(ynew).astype(int)
            df = pd.DataFrame({'x':xnew, 'y':ynew}).drop_duplicates()
            xnew,ynew = df['x'].values.tolist(), df['y'].values.tolist()
                
            for x,y in zip(x,y):
                xperp = []; yperp = [];
                for xdiff, ydiff in zip(xnew, ynew):
                    xperp.append(int(x+xdiff))
                    yperp.append(int(y+ydiff))
                allxlist.append(xperp)
                allylist.append(yperp)

                xmap = self.currDICGrain.extremeCoords[0] + xperp
                ymap = self.currDICGrain.extremeCoords[1] + yperp
                
                ### For all points, append u and v to list
                u = []; v = [];
                for xmap, ymap in zip(xmap,ymap):
                    u.append((self.currMap.crop(self.currMap.x_map))[ymap, xmap])
                    v.append((self.currMap.crop(self.currMap.y_map))[ymap, xmap])

                ### Take away mean
                u = u-np.mean(u); v = v-np.mean(v)

                ### Append to main lists (ulist,vlist)
                ulist.extend(u)
                vlist.extend(v)

        ### Linear regression of ucentered against vcentered
        linRegResults = linregress(x=vlist,y=ulist)
        
        # Save measured RDR
        grain.groupsList[group][4] = linRegResults.slope
        

        if showPlot: self.plotRDR(grain, group, ulist, vlist, allxlist, allylist, linRegResults)

    def plotRDR(self, grain, group, ulist, vlist, allxlist, allylist, linRegResults):
        """
        Plot RDR figure, including location of perpendicular lines and scatter plot of ucentered vs vcentered.
        
        Parameters
        ----------
        grain: int
            DIC grain to plot.
        group: int
            Group ID to plot.
        ulist: list
            List of ucentered values.
        vlist: list
            List of vcentered values.
        allxlist: list
            List of all x values.
        allylist: list
            List of all y values.
        linRegResults: numpy.ndarray, {slope, intercept, rvalue, pvalue, stderr}
            Results from linear regression of ucentered vs vcentered.

        """

        # Draw window and axes
        self.rdrPlot = Plot(ax=None, makeInteractive=True, title='RDR Calculation', figsize=(21, 7))
        self.rdrPlot.ax.axis('off')
        self.rdrPlot.grainAx = self.rdrPlot.addAxes((0.05, 0.07, 0.20, 0.85))
        self.rdrPlot.textAx = self.rdrPlot.addAxes((0.27, 0.07, 0.20, 0.85))
        self.rdrPlot.textAx.axis('off')
        self.rdrPlot.numLineAx = self.rdrPlot.addAxes((0.48, 0.07, 0.2, 0.85))
        self.rdrPlot.numLineAx.axis('off')
        self.rdrPlot.plotAx = self.rdrPlot.addAxes((0.75, 0.07, 0.2, 0.85))

        ## Draw grain plot
        self.rdrPlot.grainPlot = self.currDICGrain.plotMaxShear(fig=self.rdrPlot.fig, ax=self.rdrPlot.grainAx, 
                                                                plotColourBar=False, plotScaleBar = True) 
        self.rdrPlot.grainPlot.addColourBar(label='Effective Shear Strain', fraction=0.046, pad=0.04)

        ## Draw all points
        self.rdrPlot.grainAx.plot(allxlist, allylist, 'rx',lw=0.5)
        for xlist, ylist in zip(allxlist, allylist):
            self.rdrPlot.grainAx.plot(xlist, ylist, '-',lw=1)

        ## Generate scatter plot
        slope = linRegResults.slope
        r_value = linRegResults.rvalue
        intercept = linRegResults.intercept
        std_err = linRegResults.stderr
        
        self.rdrPlot.plotAx.scatter(x=vlist,y=ulist,marker='x', lw=1)
        self.rdrPlot.plotAx.plot(
            [np.min(vlist), np.max(vlist)],[slope*np.min(vlist)+intercept,slope*np.max(vlist)+intercept], '-')
        self.rdrPlot.plotAx.set_xlabel('v-centered')
        self.rdrPlot.plotAx.set_ylabel('u-centered')
        self.rdrPlot.addText(self.rdrPlot.plotAx, 0.95, 0.01, 'Slope = {0:.3f} ± {1:.3f}\nR-squared = {2:.3f}\nn={3}'
                             .format(slope,std_err,r_value**2,len(ulist)), va='bottom', ha='right',
                             transform=self.rdrPlot.plotAx.transAxes, fontsize=10);

        ## Write grain info
        ebsdGrain = grain.ebsdGrain
        ebsdGrain.calcSlipTraces()

        if ebsdGrain.averageSchmidFactors is None:
            raise Exception("Run 'calcAverageGrainSchmidFactors' first")

        eulers = np.rad2deg(ebsdGrain.refOri.eulerAngles())

        text = 'Average angle: {0:.2f}\n'.format(grain.groupsList[group][1])
        text += 'Eulers: {0:.1f}    {1:.1f}    {2:.1f}\n\n'.format(eulers[0], eulers[1], eulers[2])

        self.rdrPlot.addText(self.rdrPlot.textAx, 0.15, 1, text, fontsize=10, va='top')

        ## Write slip system info
        RDRs = []; offset = 0; 
        for idx, (ssGroup, sfGroup, slipTraceAngle) in enumerate(
                zip(grain.ebsdMap.slipSystems, ebsdGrain.averageSchmidFactors, np.rad2deg(ebsdGrain.slipTraceAngles))):
            text = "{0:s}    {1:.1f}\n".format(ssGroup[0].slipPlaneLabel, slipTraceAngle)
            tempRDRs = [];
            for ss, sf in zip(ssGroup, sfGroup):
                slipDirSample = ebsdGrain.refOri.conjugate.transformVector(ss.slipDir)
                text = text + "          {0:s}    SF: {1:.3f}    RDR: {2:.3f}\n".format\
                    (ss.slipDirLabel, sf,-slipDirSample[0]/slipDirSample[1])
                RDR = -slipDirSample[0]/slipDirSample[1]
                tempRDRs.append(RDR)
            RDRs.append(tempRDRs)    

            if idx in grain.groupsList[group][2]:
                self.rdrPlot.addText(self.rdrPlot.textAx, 0.15, 0.9-offset, text, weight='bold', fontsize=10, va='top')
            else:
                self.rdrPlot.addText(self.rdrPlot.textAx, 0.15, 0.9-offset, text, fontsize=10, va='top')

            offset += 0.0275 * text.count('\n')

        # Plot RDR values on number line
        uniqueRDRs = set()
        for x in [item for sublist in RDRs for item in sublist]: uniqueRDRs.add(x)
        self.rdrPlot.numLineAx.axvline(x=0, ymin=-20, ymax=20, c='k')
        self.rdrPlot.numLineAx.plot(np.zeros(len(uniqueRDRs)), list(uniqueRDRs), 'bo', label='Theroretical RDR values')
        self.rdrPlot.numLineAx.plot([0], slope, 'ro', label='Measured RDR value')
        self.rdrPlot.addText(self.rdrPlot.numLineAx, -0.009, slope-0.01, '{0:.3f}'.format(float(slope)))
        self.rdrPlot.numLineAx.legend(bbox_to_anchor=(1.15, 1.05))
        
        # Label RDRs by slip system on number line 
        for RDR in list(uniqueRDRs):
            self.rdrPlot.addText(self.rdrPlot.numLineAx, -0.009, RDR-0.01, '{0:.3f}'.format(float(RDR)))
            txt = ''
            for idx, ssGroup in enumerate(RDRs):
                for idx2, rdr in enumerate(ssGroup):
                    if rdr == RDR:
                        txt += str('{0} {1}  '.format(self.currEBSDMap.slipSystems[idx][idx2].slipPlaneLabel, 
                                                      self.currEBSDMap.slipSystems[idx][idx2].slipDirLabel))
            self.rdrPlot.addText(self.rdrPlot.numLineAx,0.002, RDR-0.01, txt)

        self.rdrPlot.numLineAx.set_ylim(slope-1, slope+1)
        self.rdrPlot.numLineAx.set_xlim(-0.01, 0.05)
