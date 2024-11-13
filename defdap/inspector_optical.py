# Copyright 2023 Mechanics of Microstructures Group
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
from scipy.stats import linregress
from skimage.draw import line as skimage_line
import ast

from defdap.plotting import Plot, GrainPlot
from defdap import hrdic

from typing import List


class GrainInspector:
    """
    Class containing the interactive grain inspector tool for slip trace analysis
    and relative displacement ratio analysis.

    """

    def __init__(self,
                 selected_dic_map: 'optical.Map'):
        """

        Parameters
        ----------
        selected_dic_map
            DIC map to run grain inspector on.
        """
        
        # Initialise some values
        self.grain_id = 0
        self.selected_dic_map = selected_dic_map
        self.selected_ebsd_map = self.selected_dic_map.ebsd_map
        self.selected_dic_grain = self.selected_dic_map[self.grain_id]
        self.selected_ebsd_grain = self.selected_dic_grain.ebsd_grain

        # Plot window
        self.plot = Plot(ax=None, make_interactive=True, figsize=(13, 8), title='Grain Inspector')
        div_frac = 0.7

        # Remove key bindings for figure to suppress errors
        self.plot.fig.canvas.mpl_disconnect(self.plot.fig.canvas.manager.key_press_handler_id)

        # Buttons
        self.plot.add_button(
            'Save\nLine', self.save_line, (div_frac, 0.48, 0.05, 0.04))
        self.plot.add_button(
            'Previous\nGrain', lambda e, p: self.goto_grain(self.grain_id - 1, p), (div_frac, 0.94, 0.05, 0.04))
        self.plot.add_button(
            'Next\nGrain', lambda e, p: self.goto_grain(self.grain_id + 1, p), (div_frac + 0.06, 0.94, 0.05, 0.04))
        self.plot.add_button(
            'Run All STA', self.batch_run_sta, (0.85, 0.07, 0.11, 0.04))
        self.plot.add_button(
            'Clear\nAll Lines', self.clear_all_lines, (div_frac + 0.2, 0.48, 0.05, 0.04))


        # Text boxes
        self.plot.add_text_box(label='Go to \ngrain ID:', loc=(div_frac + 0.17, 0.94, 0.05, 0.04),
                               submit_handler=self.goto_grain)
        self.plot.add_text_box(label='Remove\nID:', loc=(div_frac + 0.1, 0.48, 0.05, 0.04),
                               submit_handler=self.remove_line)

        # Axes
        self.max_shear_axis = self.plot.add_axes((0.05, 0.4, 0.65, 0.55))
        self.slip_trace_axis = self.plot.add_axes((0.25, 0.05, 0.5, 0.3))
        self.unit_cell_axis = self.plot.add_axes((0.05, 0.055, 0.2, 0.3), proj='3d')
        self.grain_info_axis = self.plot.add_axes((div_frac, 0.86, 0.25, 0.06))
        self.line_info_axis = self.plot.add_axes((div_frac, 0.55, 0.25, 0.3))
        self.groups_info_axis = self.plot.add_axes((div_frac, 0.15, 0.25, 0.3))
        
        self.grain_plot = self.selected_dic_grain.plot_grain_data(grain_data=self.selected_dic_grain.data.image, 
                                                                  fig=self.plot.fig,
                                                                  ax=self.max_shear_axis)

        self.plot.ax.axis('off')

        # Draw the stuff that will need to be redrawn often in a separate function
        self.redraw()

    def goto_grain(self,
                   event: int,
                   plot):
        """ Go to a specified grain ID.

        Parameters
        ----------
        event
            Grain ID to go to.

        """
        # Go to grain ID specified in event
        self.grain_id = int(event)
        self.grain_plot.arrow = None
        self.selected_dic_grain = self.selected_dic_map[self.grain_id]
        self.selected_ebsd_grain = self.selected_dic_grain.ebsd_grain
        self.redraw()

    def save_line(self,
                  event: np.ndarray,
                  plot):
        """  Save the start point, end point and angle of drawn line into the grain.

        Parameters
        ----------
        event
            Start x, start y, end x, end y point of line passed from drawn line.

        """

        # Get angle of lines
        line_angle = 90 - np.rad2deg(np.arctan2(self.grain_plot.p2[1] - self.grain_plot.p1[1],
                                                self.grain_plot.p2[0] - self.grain_plot.p1[0]))
        if line_angle > 180:
            line_angle -= 180
        elif line_angle < 0:
            line_angle += 180

        # Two decimal places
        points = [float("{:.2f}".format(point)) for point in self.grain_plot.p1 + self.grain_plot.p2]
        line_angle = float("{:.2f}".format(line_angle))

        # Save drawn line to the DIC grain
        self.selected_dic_grain.points_list.append([points, line_angle, -1])

        # Group lines and redraw
        self.group_lines()
        self.redraw_line()

    def group_lines(self,
                    grain: 'defdap.hrdic.Grain' = None):
        """
        Group the lines drawn in the current grain item using a mean shift algorithm,
        save the average angle and then detect the active slip planes.

        groups_list is a list of line groups: [id, angle, [slip plane id], [angular deviation]

        Parameters
        ----------
        grain
            Grain for which to group the slip lines.

        """

        if grain is None:
            grain = self.selected_dic_grain

        if grain.points_list == []:
            grain.groups_list = []
        else:
            for i, line in enumerate(grain.points_list):
                angle = line[1]
                if i == 0:
                    line[2] = 0  # Make group 0 for first detected angle
                    grain.groups_list = [[0, angle, 0, 0, 0]]
                    next_group = 1
                else:  # If there is more that one angle
                    if np.any(np.abs(np.array([x[1] for x in grain.groups_list]) - angle) < 10):
                        # If within +- 5 degrees of existing group, set that as the group
                        group = np.argmin(np.abs(np.array([x[1] for x in grain.groups_list]) - angle))
                        grain.points_list[i][2] = group
                        new_average = float('{0:.2f}'.format(
                            np.average([x[1] for x in grain.points_list if x[2] == group])))
                        grain.groups_list[group][1] = new_average
                    else:
                        # Make new group and set
                        grain.groups_list.append([next_group, angle, 0, 0, 0])
                        line[2] = next_group
                        next_group += 1

            # Detect active slip systems in each group
            for group in grain.groups_list:
                active_planes = []
                deviation = []
                experimental_angle = group[1]
                for idx, theoretical_angle in enumerate(np.rad2deg(grain.ebsd_grain.slip_trace_angles)):
                    if theoretical_angle - 5 < experimental_angle < theoretical_angle + 5:
                        active_planes.append(idx)
                        deviation.append(float('{0:.2f}'.format(experimental_angle - theoretical_angle)))
                group[2] = active_planes
                group[3] = deviation

    def clear_all_lines(self,
                        event,
                        plot):
        """ Clear all lines in a given grain.

        """

        self.selected_dic_grain.points_list = []
        self.selected_dic_grain.groups_list = []
        self.redraw()

    def remove_line(self,
                    event: int,
                    plot):
        """  Remove single line [runs after submitting a text box].

        Parameters
        ----------
        event
            Line ID to remove.

        """
        # Remove single line
        del self.selected_dic_grain.points_list[int(event)]
        self.group_lines()
        self.redraw()

    def redraw(self):
        """Draw items which need to be redrawn when changing grain ID.

        """

        # Plot max shear for grain
        self.max_shear_axis.clear()
        self.grain_plot = self.selected_dic_grain.plot_grain_data(grain_data=self.selected_dic_grain.data.image, 
                                                                  fig=self.plot.fig,
                                                                  ax=self.max_shear_axis)


        # Draw unit cell
        self.unit_cell_axis.clear()
        self.selected_ebsd_grain.plot_unit_cell(fig=self.plot.fig, ax=self.unit_cell_axis)

        # Write grain info text
        self.grain_info_axis.clear()
        self.grain_info_axis.axis('off')
        grain_info_text = 'Grain ID: {0} / {1}\n'.format(self.grain_id, len(self.selected_dic_map.grains) - 1)

        self.plot.add_text(self.grain_info_axis, 0, 1, grain_info_text, va='top', ha='left',
                           fontsize=10, fontfamily='monospace')

        # Detect lines
        self.plot.add_event_handler('button_press_event', lambda e, p: self.grain_plot.line_slice(e, p))
        self.plot.add_event_handler('button_release_event', lambda e, p: self.grain_plot.line_slice(e, p))

        self.redraw_line()

    def redraw_line(self):
        """
        Draw items which need to be redrawn when adding a line.

        """
        # Write lines text and draw lines
        title_text = 'List of lines'
        lines_text = 'ID  x0    y0    x1    y1    Angle   Group\n' \
                   '-----------------------------------------\n'
        if self.selected_dic_grain.points_list:
            for idx, points in enumerate(self.selected_dic_grain.points_list):
                lines_text += '{0:<3} {1:<5.0f} {2:<5.0f} {3:<5.0f} {4:<5.0f} {5:<7.1f} {6:<5}\n'.format(
                    idx, *points[0], points[1], points[2])
                self.grain_plot.add_arrow(start_end=points[0], clear_previous=False, persistent=True, label=idx)

        self.line_info_axis.clear()
        self.line_info_axis.axis('off')
        self.plot.add_text(self.line_info_axis, 0, 1, title_text, va='top',
                           fontsize=10, fontfamily='monospace', weight='bold')
        self.plot.add_text(self.line_info_axis, 0, 0.9, lines_text, va='top',
                           fontsize=10, fontfamily='monospace')

        # Write groups info text
        title_text = 'List of groups'

        groupsTxt = 'ID  Av. Angle  System  Dev\n' \
                    '--------------------------\n'
        if self.selected_dic_grain.groups_list:
            for idx, group in enumerate(self.selected_dic_grain.groups_list):
                groupsTxt += '{0:<3} {1:<10.1f} {2:<7} {3:<12}\n'.format(
                    idx,
                    group[1],
                    ','.join([str(np.round(i, 1)) for i in group[2]]),
                    ','.join([str(np.round(i, 1)) for i in group[3]]))

        self.groups_info_axis.clear()
        self.groups_info_axis.axis('off')
        self.plot.add_text(self.groups_info_axis, 0, 1, title_text, va='top', fontsize=10, fontfamily='monospace',
                           weight='bold')
        self.plot.add_text(self.groups_info_axis, 0, 0.9, groupsTxt, va='top', fontsize=10, fontfamily='monospace')

        # Draw slip traces
        self.slip_trace_axis.clear()
        self.slip_trace_axis.set_aspect('equal', 'box')

        
        slipPlot = GrainPlot(fig=self.plot.fig,
                             calling_grain=self.selected_dic_map[self.grain_id].ebsd_grain, ax=self.slip_trace_axis)
        traces = slipPlot.add_slip_traces(top_only=True)
        
        self.slip_trace_axis.axis('off')

        # Draw slip bands
        bands = [elem[1] for elem in self.selected_dic_grain.groups_list]
        if self.selected_dic_grain.groups_list != None:
            slipPlot.add_slip_bands(top_only=True, angles=list(np.deg2rad(bands)))


    def batch_run_sta(self,
                      event,
                      plot):
        """  Run slip trace analysis on all grains which hve slip trace lines drawn.

        """

        # Print header
        print("Grain\tEul1\tEul2\tEul3\tMaxSF\tGroup\tAngle\tSystem\tDev")

        # Print information for each grain
        for idx, grain in enumerate(self.selected_dic_map):
            if grain.points_list != []:
                for group in grain.groups_list:
                    maxSF = np.max([item for sublist in grain.ebsd_grain.average_schmid_factors for item in sublist])
                    eulers = self.selected_ebsd_grain.ref_ori.euler_angles() * 180 / np.pi
                    text = '{0}\t{1:.1f}\t{2:.1f}\t{3:.1f}\t{4:.3f}\t'.format(
                        idx, eulers[0], eulers[1], eulers[2], maxSF)
                    text += '{0}\t{1:.1f}\t{2}\t{3}'.format(
                        group[0], group[1], group[2], np.round(group[3], 3))
                    print(text)


