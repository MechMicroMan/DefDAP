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

import ast

from typing import List

from scipy.stats import linregress
from scipy.stats._stats_mstats_common import LinregressResult
import pandas as pd

from defdap.plotting import Plot, GrainPlot
from defdap import hrdic


class GrainInspector:
    """
    Class containing the interactive grain inspector tool for slip trace analysis
    and relative displacement ratio analysis.

    """
    def __init__(self, 
        currMap: 'hrdic.Map', 
        vmax: float = 0.1,
        corrAngle: float = 0):
        # Initialise some values
        self.grainID = 0
        self.currMap = currMap
        self.currEBSDMap = self.currMap.ebsdMap
        self.currDICGrain = self.currMap[self.grainID]
        self.currEBSDGrain = self.currDICGrain.ebsdGrain
        self.vmax = vmax
        self.corrAngle = corrAngle
        self.filename = str(self.currMap.retrieveName()) + '_RDR.txt'
        
        # Draw the figure
        self.draw()        

    def draw(self):
        """ Draw the main window, buttons, text boxes and axes.

        """
        # Plot window
        self.plot = Plot(ax=None, makeInteractive=True, figsize=(14,8), title='Grain Inspector')
        
        ######## Buttons
        self.plot.addButton(
            'Save\nLine', self.saveLine, (0.73, 0.48, 0.05, 0.04))
        self.plot.addButton(
            'Previous\nGrain', lambda e, p: self.gotoGrain(self.grainID-1, p), (0.73, 0.94, 0.05, 0.04))
        self.plot.addButton(
            'Next\nGrain', lambda e, p: self.gotoGrain(self.grainID+1, p), (0.79, 0.94, 0.05, 0.04))
        self.plot.addButton(
            'Run\nAll STA', self.batchRunSTA, (0.85, 0.07, 0.11, 0.04))
        self.plot.addButton(
            'Clear\nAll Lines', self.clearAllLines, (0.89, 0.48, 0.05, 0.04))
        self.plot.addButton(
            'Load\nFile', self.loadFile, (0.85, 0.02, 0.05, 0.04))
        self.plot.addButton(
            'Save\nFile', self.saveFile, (0.91, 0.02, 0.05, 0.04))
        

        # Text boxes
        self.plot.addTextBox(label='', loc=(0.7, 0.02, 0.13, 0.04),
            changeHandler=self.updateFilename, initial = self.filename)
        self.plot.addTextBox(label='Go to \ngrain ID:', loc=(0.9, 0.94, 0.05, 0.04), 
            submitHandler=self.gotoGrain)
        self.plot.addTextBox(label='Remove\nID:', loc=(0.83, 0.48, 0.05, 0.04), 
            submitHandler=self.removeLine)
        self.RDRGroupBox = self.plot.addTextBox(label='Run RDR only\non group:', loc=(0.78, 0.07, 0.05, 0.04), 
            submitHandler=self.runRDRGroup)

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

    def gotoGrain(self, 
        event: int, 
        plot):
        """ Go to a specified grain ID.

        Parameters
        ----------
        event
            Grain ID to go to.

        """
        ## Go to grain ID specified in event
        self.grainID=int(event)
        self.grainPlot.arrow=None
        self.currDICGrain = self.currMap[self.grainID]
        self.currEBSDGrain = self.currDICGrain.ebsdGrain
        self.redraw()

    def saveLine(self, 
        event: np.ndarray,
        plot):
        """  Save the start point, end point and angle of drawn line into the grain.

        Parameters
        ----------
        event
            Start x, start y, end x, end y point of line passed from drawn line.

        """

        # Get angle of lines
        lineAngle = 90-np.rad2deg(np.arctan2(self.grainPlot.p2[1]-self.grainPlot.p1[1],
                                              self.grainPlot.p2[0]-self.grainPlot.p1[0]))
        if lineAngle > 180: lineAngle -= 180
        elif lineAngle < 0: lineAngle += 180
        
        lineAngle -= self.corrAngle

        # Two decimal places
        points = [float("{:.2f}".format(point)) for point in self.grainPlot.p1+self.grainPlot.p2]
        lineAngle = float("{:.2f}".format(lineAngle))

        # Save drawn line to the DIC grain
        self.currDICGrain.pointsList.append([points, lineAngle, -1])
        
        # Group lines and redraw
        self.groupLines()
        self.redrawLine()
        
    def groupLines(self,
                   grain: 'defdap.hrdic.Grain'=None):
        """
        Group the lines drawn in the current grain item using a mean shift algorithm,
        save the average angle and then detect the active slip planes.

        groupsList is a list of line groups: [id, angle, [slip plane id], [angular deviation]

        Parameters
        ----------
        grain
            Grain for which to group the slip lines.

        """

        if grain == None:
            grain = self.currDICGrain

        if grain.pointsList == []:
            grain.groupsList = []
        else:
            for i, line in enumerate(grain.pointsList):
                angle = line[1]
                if i == 0:       
                    line[2]=0       # Make group 0 for first detected angle
                    grain.groupsList = [[0, angle, 0, 0, 0]]
                    nextGroup=1
                else:       # If there is more that one angle
                    if np.any(np.abs(np.array([x[1] for x in grain.groupsList])-angle)<10):
                        # If within +- 5 degrees of exisitng group, set that as the group
                        group =  np.argmin(np.abs(np.array([x[1] for x in grain.groupsList])-angle))
                        grain.pointsList[i][2]=group
                        newAv = float('{0:.2f}'.format(np.average([x[1] for x in grain.pointsList if x[2]==group])))
                        grain.groupsList[group][1] = newAv
                    else:
                        # Make new group and set
                        grain.groupsList.append([nextGroup, angle, 0, 0, 0])
                        line[2]=nextGroup
                        nextGroup += 1
          
            # Detect active slip systems in each group
            for group in grain.groupsList:
                activePlanes = []
                deviation = []
                experimentalAngle = group[1]
                for idx, theoreticalAngle in enumerate(np.rad2deg(grain.ebsdGrain.slipTraceAngles)):
                    if theoreticalAngle-5 < experimentalAngle < theoreticalAngle+5:
                        activePlanes.append(idx)
                        deviation.append(float('{0:.2f}'.format(experimentalAngle-theoreticalAngle)))
                group[2] = activePlanes
                group[3] = deviation
            
    def clearAllLines(self, 
        event, 
        plot):
        """ Clear all lines in a given grain.

        """

        self.currDICGrain.pointsList = []
        self.currDICGrain.groupsList = []
        self.redraw()

    def removeLine(self, 
        event: int, 
        plot):
        """  Remove single line [runs after submitting a text box].

        Parameters
        ----------
        event
            Line ID to remove.

        """
        ## Remove single line
        del self.currDICGrain.pointsList[int(event)]
        self.groupLines()
        self.redraw()

    def redraw(self):
        """Draw items which need to be redrawn when changing grain ID.

        """

        # Plot max shear for grain
        self.maxShearAx.clear()
        self.grainPlot = self.currMap[self.grainID].plotMaxShear(
            fig=self.plot.fig, ax=self.maxShearAx, vmax=self.vmax, plotColourBar=False, plotScaleBar=True)
        
        # Draw unit cell
        self.unitCellAx.clear()
        self.currEBSDGrain.plotUnitCell(fig=self.plot.fig, ax=self.unitCellAx)
        
        # Write grain info text
        self.grainInfoAx.clear()
        self.grainInfoAx.axis('off')
        grainInfoText = 'Grain ID: {0} / {1}\n'.format(self.grainID, len(self.currMap.grainList)-1)
        grainInfoText += 'Min: {0:.1f} %     Mean:{1:.1f} %     Max: {2:.1f} %'.format(
            np.min(self.currDICGrain.maxShearList)*100,
            np.mean(self.currDICGrain.maxShearList)*100,
            np.max(self.currDICGrain.maxShearList)*100)
        self.plot.addText(self.grainInfoAx, 0, 1, grainInfoText,  va='top', ha='left', fontsize=10)
        
        # Detect lines
        self.plot.addEventHandler('button_press_event',lambda e, p: self.grainPlot.lineSlice(e, p))
        self.plot.addEventHandler('button_release_event', lambda e, p: self.grainPlot.lineSlice(e, p))

        self.redrawLine()

    def redrawLine(self):
        """
        Draw items which need to be redrawn when adding a line.

        """
        # Write lines text and draw lines
        linesTxt = 'List of lines\n\nLineID  x0     y0     x1     y1     Angle  Group\n'

        if self.currDICGrain.pointsList != []:
            for idx, points in enumerate(self.currDICGrain.pointsList):
                linesTxt += '{0}          {1:.1f}   {2:.1f}    {3:.1f}   {4:.1f}   {5:.1f}   {6}\n'.format(idx,
                         points[0][0],points[0][1],points[0][2],points[0][3],points[1],points[2])
                self.grainPlot.addArrow(startEnd=points[0], clearPrev=False, persistent=True, label=idx)

        self.lineInfoAx.clear()
        self.lineInfoAx.axis('off')
        self.plot.addText(self.lineInfoAx, 0, 1, linesTxt, va='top', fontsize=10)

        # Write groups info text
        groupsTxt = 'List of groups\n\nGroupID    Angle      System      Dev     RDR\n'
        if self.currDICGrain.groupsList != []:
            for idx, group in enumerate(self.currDICGrain.groupsList):
                groupsTxt += '{0}                {1:.1f}      {2}      {3}      {4:.2f}\n'.format(
                    idx, group[1], group[2], np.round(group[3], 3), group[4])

        self.groupsInfoAx.clear()
        self.groupsInfoAx.axis('off')
        self.plot.addText(self.groupsInfoAx, 0, 1, groupsTxt, va='top', fontsize=10)

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

    def runRDRGroup(self,
        event: int, 
        plot):
        """  Run RDR on a specified group, upon submitting a text box.

        Parameters
        ----------
        event
            Group ID specified from text box.

        """
        ## Run RDR for group of lines
        if event != '':
            self.calcRDR(grain = self.currDICGrain, group=int(event))
            self.RDRGroupBox.set_val('')
        
    def batchRunSTA(self, 
        event, 
        plot):
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

    def calcRDR(self, 
        grain: int, 
        group: int, 
        showPlot: bool = True, 
        length: float = 2.5):
        """ Calculates the relative displacement ratio for a given grain and group.

        Parameters
        ----------
        grain
            DIC grain ID to run RDR on.
        group
            group ID to run RDR on.
        showPlot
            if True, show plot window.
        length
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

    def plotRDR(self, 
        grain: int, 
        group: int, 
        ulist: List[float], 
        vlist: List[float], 
        allxlist: List[float], 
        allylist: List[float], 
        linRegResults: 'LinregressResult'):
        """
        Plot RDR figure, including location of perpendicular lines and scatter plot of ucentered vs vcentered.
        
        Parameters
        ----------
        grain
            DIC grain to plot.
        group
            Group ID to plot.
        ulist
            List of ucentered values.
        vlist
            List of vcentered values.
        allxlist
            List of all x values.
        allylist
            List of all y values.
        linRegResults
            Results from linear regression of ucentered vs vcentered 
            {slope, intercept, rvalue, pvalue, stderr}.

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

    def updateFilename(self, 
        event: str,
        plot):
        """  Update class variable filename, based on text input from textbox handler.

        event: 
            Text in textbox.

        """
        
        self.filename = event

    def saveFile(self,
        event, 
        plot):
        """  Save a file which contains definitions of slip lines drawn in grains
            [(x0, y0, x1, y1), angle, groupID]
            and groups of lines, defined by an average angle and identified sip plane
            [groupID, angle, [slip plane id(s)], [angular deviation(s)]]

        """

        with open(self.currMap.path + str(self.filename), 'w') as file:
            file.write('# This is a file generated by defdap which contains definitions of slip lines drawn in grains by grainInspector\n')
            file.write('# [(x0, y0, x1, y1), angle, groupID]\n')
            file.write('# and groups of lines, defined by an average angle and identified sip plane\n')
            file.write('# [groupID, angle, [slip plane id], [angular deviation]\n\n')

            for i, grain in enumerate(self.currMap):
                if grain.pointsList != []:
                    file.write('Grain {0}\n'.format(i))
                    file.write('{0} Lines\n'.format(len(grain.pointsList)))
                    for point in grain.pointsList:
                        file.write(str(point)+'\n')
                    file.write('{0} Groups\n'.format(len(grain.groupsList)))
                    for group in grain.groupsList:
                        file.write(str(group)+'\n')
                    file.write('\n')

    def loadFile(self,
        event, 
        plot):
        """  Load a file which contains definitions of slip lines drawn in grains
            [(x0, y0, x1, y1), angle, groupID]
            and groups of lines, defined by an average angle and identified sip plane
            [groupID, angle, [slip plane id(s)], [angular deviation(s)]]

        """

        with open(self.currMap.path + str(self.filename), 'r') as file:
            lines = file.readlines()

        # Parse file and make list of 
        # [start index, grain ID, number of lines, number of groups]
        indexlist=[]
        for i, line in enumerate(lines):
            if line[0] != '#' and len(line) >1:
                if ('Grain') in line:
                    grainID = int(line.split(' ')[-1])
                    startIndex = i
                if ('Lines') in line:
                    numLines = int(line.split(' ')[0])
                if ('Groups') in line:
                    numGroups = int(line.split(' ')[0])
                    indexlist.append([startIndex, grainID, numLines, numGroups])

        # Write data from file into grain
        for startIndex, grainID, numLines, numGroups in indexlist:
            startIndexLines = startIndex+2
            grainPoints = lines[startIndexLines:startIndexLines+numLines]
            for point in grainPoints:
                self.currMap[grainID].pointsList.append(ast.literal_eval(point.split('\\')[0]))
            
            startIndexGroups = startIndex+3+numLines
            grainGroups = lines[startIndexGroups:startIndexGroups+numGroups]
            for group in grainGroups:
                self.currMap[grainID].groupsList.append(ast.literal_eval(group.split('\\')[0]))

        self.redraw()
