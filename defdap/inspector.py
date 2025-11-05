# Copyright 2025 Mechanics of Microstructures Group
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
import pandas as pd

from defdap.plotting import Plot, GrainPlot
from defdap import hrdic

from typing import List


class GrainInspector:
    """
    Class containing the interactive grain inspector tool for slip trace analysis
    and relative displacement ratio analysis.

    """

    def __init__(self,
                 selected_dic_map: 'hrdic.Map',
                 vmax: float,
                 correction_angle: float = 0,
                 rdr_line_length: int = 3):
        """

        Parameters
        ----------
        selected_dic_map
            DIC map to run grain inspector on.
        vmax
            Maximum effective shear strain in colour scale.
        correction_angle
            Angle (in degrees) to subtract from drawn line angle.
        rdr_line_length
            Length on lines perpendicular to slip trace (can be any odd number above default 3).
        """
        # Initialise some values
        self.grain_id = 0
        self.selected_dic_map = selected_dic_map
        self.selected_ebsd_map = self.selected_dic_map.ebsd_map
        self.selected_dic_grain = self.selected_dic_map[self.grain_id]
        self.selected_ebsd_grain = self.selected_dic_grain.ebsd_grain
        self.vmax = vmax
        self.correction_angle = correction_angle
        self.rdr_line_length = rdr_line_length
        self.filename = str(self.selected_dic_map.retrieve_name()) + '_RDR.txt'

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
        self.plot.add_button(
            'Load\nFile', self.load_file, (0.85, 0.02, 0.05, 0.04))
        self.plot.add_button(
            'Save\nFile', self.save_file, (0.91, 0.02, 0.05, 0.04))

        # Text boxes
        self.plot.add_text_box(label='', loc=(0.7, 0.02, 0.13, 0.04),
                               change_handler=self.update_filename, initial=self.filename)
        self.plot.add_text_box(label='Go to \ngrain ID:', loc=(div_frac + 0.17, 0.94, 0.05, 0.04),
                               submit_handler=self.goto_grain)
        self.plot.add_text_box(label='Remove\nID:', loc=(div_frac + 0.1, 0.48, 0.05, 0.04),
                               submit_handler=self.remove_line)
        self.rdr_group_text_box = self.plot.add_text_box(label='Run RDR only\non group:', loc=(0.78, 0.07, 0.05, 0.04),
                                                         submit_handler=self.run_rdr_group)

        # Axes
        self.max_shear_axis = self.plot.add_axes((0.05, 0.4, 0.65, 0.55))
        self.slip_trace_axis = self.plot.add_axes((0.25, 0.05, 0.5, 0.3))
        self.unit_cell_axis = self.plot.add_axes((0.05, 0.055, 0.2, 0.3), proj='3d')
        self.grain_info_axis = self.plot.add_axes((div_frac, 0.86, 0.25, 0.06))
        self.line_info_axis = self.plot.add_axes((div_frac, 0.55, 0.25, 0.3))
        self.groups_info_axis = self.plot.add_axes((div_frac, 0.15, 0.25, 0.3))
        self.grain_plot = self.selected_dic_map[self.grain_id].plot_map(
            'max_shear', 
            fig=self.plot.fig,
            ax=self.max_shear_axis,
            vmax=self.vmax,
            plot_scale_bar=True,
            plot_colour_bar=True
        )
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

        line_angle -= self.correction_angle

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
        self.grain_plot = self.selected_dic_map[self.grain_id].plot_map(
            'max_shear', fig=self.plot.fig, ax=self.max_shear_axis, 
            vmax=self.vmax, plot_colour_bar=False, plot_scale_bar=True
        )

        # Draw unit cell
        self.unit_cell_axis.clear()
        self.selected_ebsd_grain.plot_unit_cell(fig=self.plot.fig, ax=self.unit_cell_axis)

        # Write grain info text
        self.grain_info_axis.clear()
        self.grain_info_axis.axis('off')
        grain_info_text = 'Grain ID: {0} / {1}\n'.format(self.grain_id, len(self.selected_dic_map.grains) - 1)
        grain_info_text += 'Min: {0:.2f} %     Mean:{1:.2f} %     Max: {2:.2f} %'.format(
            np.min(self.selected_dic_grain.data.max_shear) * 100,
            np.mean(self.selected_dic_grain.data.max_shear) * 100,
            np.max(self.selected_dic_grain.data.max_shear) * 100)
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

        groupsTxt = 'ID  Av. Angle  System  Dev          RDR\n' \
                    '----------------------------------------\n'
        if self.selected_dic_grain.groups_list:
            for idx, group in enumerate(self.selected_dic_grain.groups_list):
                groupsTxt += '{0:<3} {1:<10.1f} {2:<7} {3:<12} {4:.2f}\n'.format(
                    idx,
                    group[1],
                    ','.join([str(np.round(i, 1)) for i in group[2]]),
                    ','.join([str(np.round(i, 1)) for i in group[3]]),
                    group[4])

        self.groups_info_axis.clear()
        self.groups_info_axis.axis('off')
        self.plot.add_text(self.groups_info_axis, 0, 1, title_text, va='top', fontsize=10, fontfamily='monospace',
                           weight='bold')
        self.plot.add_text(self.groups_info_axis, 0, 0.9, groupsTxt, va='top', fontsize=10, fontfamily='monospace')

        # Draw slip traces
        self.slip_trace_axis.clear()
        self.slip_trace_axis.set_aspect('equal', 'box')
        slipPlot = GrainPlot(fig=self.plot.fig,
                             calling_grain=self.selected_dic_map[self.grain_id], ax=self.slip_trace_axis)
        traces = slipPlot.add_slip_traces(top_only=True)
        self.slip_trace_axis.axis('off')

        # Draw slip bands
        bands = [elem[1] for elem in self.selected_dic_grain.groups_list]
        if self.selected_dic_grain.groups_list != None:
            slipPlot.add_slip_bands(top_only=True, angles=list(np.deg2rad(bands)))

    def run_rdr_group(self,
                      event: int,
                      plot):
        """  Run RDR on a specified group, upon submitting a text box.

        Parameters
        ----------
        event
            Group ID specified from text box.

        """
        # Run RDR for group of lines
        if event != '':
            self.calc_rdr(grain=self.selected_dic_grain, group=int(event))
            self.rdr_group_text_box.set_val('')

    def batch_run_sta(self,
                      event,
                      plot):
        """  Run slip trace analysis on all grains which hve slip trace lines drawn.

        """

        # Print header
        print("Grain\tEul1\tEul2\tEul3\tMaxSF\tGroup\tAngle\tSystem\tDev\tRDR")

        # Print information for each grain
        for idx, grain in enumerate(self.selected_dic_map):
            if grain.points_list != []:
                for group in grain.groups_list:
                    maxSF = np.max([item for sublist in grain.ebsd_grain.average_schmid_factors for item in sublist])
                    eulers = self.selected_ebsd_grain.ref_ori.euler_angles() * 180 / np.pi
                    text = '{0}\t{1:.1f}\t{2:.1f}\t{3:.1f}\t{4:.3f}\t'.format(
                        idx, eulers[0], eulers[1], eulers[2], maxSF)
                    text += '{0}\t{1:.1f}\t{2}\t{3}\t{4:.2f}'.format(
                        group[0], group[1], group[2], np.round(group[3], 3), group[4])
                    print(text)

    def calc_rdr(self,
                 grain,
                 group: int,
                 show_plot: bool = True):
        """ Calculates the relative displacement ratio for a given grain and group.

        Parameters
        ----------
        grain
            DIC grain to run RDR on.
        group
            group ID to run RDR on.
        show_plot
            if True, show plot window.

        """

        u_list, v_list, x_list, y_list = [], [], [], []

        # Get all lines belonging to group
        point_array = np.array(grain.points_list, dtype=object)
        points = list(point_array[:, 0][point_array[:, 2] == group])
        angle = grain.groups_list[group][1]

        # Lookup deviation from (0,0) for 3 points along line perpendicular to slip line (x_new,y_new)
        x_new = np.array([[-1, 0, 1], [1, 0, -1], [0, 0, 0], [1, 0, -1], [-1, 0, 1]])[int(np.round(angle / 45, 0))]
        y_new = np.array([[0, 0, 0], [-1, 0, 1], [-1, 0, 1], [1, 0, -1], [0, 0, 0]])[int(np.round(angle / 45, 0))]
        tuples = list(zip(x_new, y_new))

        # Allow increasing line length from default 3 to any odd number
        num = np.arange(0, int((self.rdr_line_length - 1) / 2)) + 1
        coordinateOffsets = np.unique(np.array([np.array(tuples)*i for i in num]).reshape(-1, 2), axis=0)

        # For each slip trace line
        for point in points:
            x0, y0, x1, y1 = point

            # Calculate positions for each pixel along slip trace line
            x, y = skimage_line(int(x0), int(y0), int(x1), int(y1))

            # Get x and y coordinates for points to be samples for RDR
            xmap = np.array(x).T[:, None] + coordinateOffsets[:,0] + self.selected_dic_grain.extreme_coords[0]
            ymap = np.array(y).T[:, None] + coordinateOffsets[:,1] + self.selected_dic_grain.extreme_coords[1]

            x_list.extend(xmap - self.selected_dic_grain.extreme_coords[0])
            y_list.extend(ymap - self.selected_dic_grain.extreme_coords[1])

            # Get u and v values at each coordinate
            u = self.selected_dic_map.crop(self.selected_dic_map.data.displacement[0])[ymap, xmap]
            v = self.selected_dic_map.crop(self.selected_dic_map.data.displacement[1])[ymap, xmap]

            # Subtract mean u and v value for each row
            u_list.extend(u - np.mean(u, axis=1)[:, None])
            v_list.extend(v - np.mean(v, axis=1)[:, None])

        # Linear regression of ucentered against vcentered
        lin_reg_result = linregress(x=np.array(v_list).flatten(), y=np.array(u_list).flatten())

        # Save measured RDR
        grain.groups_list[group][4] = lin_reg_result.slope

        if show_plot:
            self.plot_rdr(grain, group, u_list, v_list, x_list, y_list, lin_reg_result)

    def plot_rdr(self,
                 grain,
                 group: int,
                 u_list: List[float],
                 v_list: List[float],
                 x_list: List[List[int]],
                 y_list: List[List[int]],
                 lin_reg_result: List):
        """
        Plot rdr figure, including location of perpendicular lines and scatter plot of ucentered vs vcentered.
        
        Parameters
        ----------
        grain
            DIC grain to plot.
        group
            Group ID to plot.
        u_list
            List of ucentered values.
        v_list
            List of vcentered values.
        x_list
            List of all x values.
        y_list
            List of all y values.
        lin_reg_result
            Results from linear regression of ucentered vs vcentered 
            {slope, intercept, rvalue, pvalue, stderr}.

        """

        # Draw window and axes
        self.rdr_plot = Plot(ax=None, make_interactive=True, title='RDR Calculation', figsize=(15, 8))
        self.rdr_plot.ax.axis('off')
        self.rdr_plot.grain_axis = self.rdr_plot.add_axes((0.05, 0.5, 0.3, 0.45))
        self.rdr_plot.text_axis = self.rdr_plot.add_axes((0.37, 0.05, 0.3, 0.85))
        self.rdr_plot.text_axis.axis('off')
        self.rdr_plot.number_line_axis = self.rdr_plot.add_axes((0.64, 0.05, 0.3, 0.83))
        self.rdr_plot.number_line_axis.axis('off')
        self.rdr_plot.plot_axis = self.rdr_plot.add_axes((0.05, 0.1, 0.3, 0.35))

        # Draw grain plot
        self.rdr_plot.grainPlot = self.selected_dic_grain.plot_grain_data(
            grain_data=self.selected_dic_grain.data.max_shear,
            fig=self.rdr_plot.fig,
            ax=self.rdr_plot.grain_axis,
            plot_colour_bar=False,
            plot_scale_bar=True)

        self.rdr_plot.grainPlot.add_colour_bar(label='Effective Shear Strain', fraction=0.046, pad=0.04)

        # Draw all points
        self.rdr_plot.grain_axis.plot(x_list, y_list, 'rx', lw=0.5)
        for xlist, ylist in zip(x_list, y_list):
            self.rdr_plot.grain_axis.plot(xlist, ylist, '-', lw=1)

        # Generate scatter plot
        slope = lin_reg_result.slope
        r_value = lin_reg_result.rvalue
        intercept = lin_reg_result.intercept
        std_err = lin_reg_result.stderr

        self.rdr_plot.plot_axis.scatter(x=v_list, y=u_list, marker='x', lw=1)
        self.rdr_plot.plot_axis.plot(
            [np.min(v_list), np.max(v_list)],
            [slope * np.min(v_list) + intercept, slope * np.max(v_list) + intercept], '-')
        self.rdr_plot.plot_axis.set_xlabel('v-centered')
        self.rdr_plot.plot_axis.set_ylabel('u-centered')
        self.rdr_plot.add_text(self.rdr_plot.plot_axis, 0.95, 0.01,
                               'Slope = {0:.3f} ± {1:.3f}\nR-squared = {2:.3f}\nn={3}'
                               .format(slope, std_err, r_value ** 2, len(u_list)),
                               va='bottom', ha='right',
                               transform=self.rdr_plot.plot_axis.transAxes, fontsize=10, fontfamily='monospace');

        self.selected_ebsd_grain.calc_slip_traces()
        self.selected_ebsd_grain.calc_rdr()

        if self.selected_ebsd_grain.average_schmid_factors is None:
            raise Exception("Run 'calc_average_grain_schmid_factors' first")

        # Write grain info
        eulers = np.rad2deg(self.selected_ebsd_grain.ref_ori.euler_angles())
        text = 'Average angle: {0:.2f}\n'.format(grain.groups_list[group][1])
        text += 'Eulers: {0:.1f}    {1:.1f}    {2:.1f}\n\n'.format(eulers[0], eulers[1], eulers[2])

        self.rdr_plot.add_text(self.rdr_plot.text_axis, 0.15, 1, text, fontsize=10, va='top', fontfamily='monospace')

        # Write slip system info
        offset = 0

        # Loop over groups of slip systems with same slip plane
        for i, slip_system_group in enumerate(self.selected_ebsd_grain.phase.slip_systems):
            slip_trace_angle = np.rad2deg(self.selected_ebsd_grain.slip_trace_angles[i])
            text = "Plane: {0:s}    Angle: {1:.1f}\n".format(slip_system_group[0].slip_plane_label,
                                                             slip_trace_angle)

            # Then loop over individual slip systems
            for j, slip_system in enumerate(slip_system_group):
                schmid_factor = self.selected_ebsd_grain.average_schmid_factors[i][j]

                text = text + "          {0:s}    SF: {1:.3f}    RDR: {2:.3f}\n".format(
                    slip_system.slip_dir_label, schmid_factor, self.selected_ebsd_grain.rdr[i][j])

            if i in grain.groups_list[group][2]:
                self.rdr_plot.add_text(self.rdr_plot.text_axis, 0.15, 0.9 - offset, text, va='top',
                                       weight='bold', fontsize=10)
            else:
                self.rdr_plot.add_text(self.rdr_plot.text_axis, 0.15, 0.9 - offset, text, va='top',
                                       fontsize=10)

            offset += 0.0275 * text.count('\n')

        # Finf all unique rdr values
        unique_rdrs = set([item for sublist in self.selected_ebsd_grain.rdr for item in sublist])

        # Plot number line
        self.rdr_plot.number_line_axis.axvline(x=0, ymin=-20, ymax=20, c='k')

        # Theoretical values as blue points
        self.rdr_plot.number_line_axis.plot(np.zeros(len(unique_rdrs)), list(unique_rdrs),
                                            'bo', label='Theoretical RDR values')

        # Measured values as red points
        self.rdr_plot.number_line_axis.plot([0], slope, 'ro', label='Measured RDR value')
        self.rdr_plot.add_text(self.rdr_plot.number_line_axis, -0.002, slope, '{0:.3f}'.format(float(slope)),
                               fontfamily='monospace', horizontalalignment='right', verticalalignment='center')

        self.rdr_plot.number_line_axis.legend(bbox_to_anchor=(1.15, 1.05))

        # Label rdrs by slip system on number line
        for unique_rdr in list(unique_rdrs):
            if (unique_rdr > slope - 1.5) & (unique_rdr < slope + 1.5):
                # Add number to the left of point
                self.rdr_plot.add_text(self.rdr_plot.number_line_axis, -0.002, unique_rdr,
                                       '{0:.3f}'.format(float(unique_rdr)),
                                       fontfamily='monospace', horizontalalignment='right', verticalalignment='center')

                # Go through all planes and directions and add to string if they have the rdr from above loop
                txt = ''
                for i, slip_system_group in enumerate(self.selected_ebsd_grain.phase.slip_systems):
                    # Then loop over individual slip systems
                    for j, slip_system in enumerate(slip_system_group):
                        rdr = self.selected_ebsd_grain.rdr[i][j]
                        if rdr == unique_rdr:
                            txt += str('{0} {1}  '.format(slip_system.slip_plane_label, slip_system.slip_dir_label))

                self.rdr_plot.add_text(self.rdr_plot.number_line_axis, 0.002, unique_rdr - 0.01,
                                       txt)

        self.rdr_plot.number_line_axis.set_ylim(slope - 1.5, slope + 1.5)
        self.rdr_plot.number_line_axis.set_xlim(-0.01, 0.05)

    def update_filename(self,
                        event: str,
                        plot):
        """  Update class variable filename, based on text input from textbox handler.

        event: 
            Text in textbox.

        """

        self.filename = event

    def save_file(self,
                  event,
                  plot):
        """  Save a file which contains definitions of slip lines drawn in grains
            [(x0, y0, x1, y1), angle, groupID]
            and groups of lines, defined by an average angle and identified sip plane
            [groupID, angle, [slip plane id(s)], [angular deviation(s)]]

        """

        with open(self.selected_dic_map.file_name.parent / str(self.filename), 'w') as file:
            file.write('# This is a file generated by defdap which contains ')
            file.write('definitions of slip lines drawn in grains by grainInspector\n')
            file.write('# [(x0, y0, x1, y1), angle, groupID]\n')
            file.write('# and groups of lines, defined by an average angle and identified sip plane\n')
            file.write('# [groupID, angle, [slip plane id], [angular deviation]\n\n')

            for i, grain in enumerate(self.selected_dic_map):
                if grain.points_list != []:
                    file.write('Grain {0}\n'.format(i))
                    file.write('{0} Lines\n'.format(len(grain.points_list)))
                    for point in grain.points_list:
                        file.write(str(point) + '\n')
                    file.write('{0} Groups\n'.format(len(grain.groups_list)))
                    for group in grain.groups_list:
                        file.write(str(group) + '\n')
                    file.write('\n')

    def load_file(self,
                  event,
                  plot):
        """  Load a file which contains definitions of slip lines drawn in grains
            [(x0, y0, x1, y1), angle, groupID]
            and groups of lines, defined by an average angle and identified sip plane
            [groupID, angle, [slip plane id(s)], [angular deviation(s)]]

        """

        with open(self.selected_dic_map.file_name.parent / str(self.filename), 'r') as file:
            lines = file.readlines()

        # Parse file and make list of 
        # [start index, grain ID, number of lines, number of groups]
        index_list = []
        for i, line in enumerate(lines):
            if line[0] != '#' and len(line) > 1:
                if 'Grain' in line:
                    grain_id = int(line.split(' ')[-1])
                    start_index = i
                if 'Lines' in line:
                    num_lines = int(line.split(' ')[0])
                if 'Groups' in line:
                    num_groups = int(line.split(' ')[0])
                    index_list.append([start_index, grain_id, num_lines, num_groups])

        # Write data from file into grain
        for start_index, grain_id, num_lines, num_groups in index_list:
            start_index_lines = start_index + 2
            grain_points = lines[start_index_lines:start_index_lines + num_lines]
            for point in grain_points:
                self.selected_dic_map[grain_id].points_list.append(ast.literal_eval(point.split('\\')[0]))

            start_index_groups = start_index + 3 + num_lines
            grain_groups = lines[start_index_groups:start_index_groups + num_groups]
            for group in grain_groups:
                self.selected_dic_map[grain_id].groups_list.append(ast.literal_eval(group.split('\\')[0]))

        self.redraw()
