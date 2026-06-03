Grain inspector (`defdap.inspector.GrainInspector`)
===================================================

The inspector class in DefDAP provides GUI tools for interrogating HRDIC maps on a grain-by-grain basis.

Launching the Grain Inspector
-----------------------------

To launch the grain inspector, use the ``grain_inspector`` method of the HRDIC Map class:

.. code-block:: python

    dic_map.grain_inspector()

The ``vmax`` argument can be used to set the maximum value for the colorbar in the inspector. 
The ``rdr_line_length`` argument can be used to set the length of the line segments used to measure the RDR direction in the inspector. The default value is 3 pixels.

Interface Overview
------------------

.. image:: /_static/inspector.png

Navigating between grains
-------------------------

To navigate between grains, use the 'Previous Grain' and 'Next Grain' buttons, or enter a grain ID into the 'Go to Grain ID' field and press 'Enter'.
The grain ID and minimum, mean and maximum values of effective shear strain are shown below the navigation buttons.

Drawing lines for slip trace analysis and RDR
---------------------------------------------

- To add a line, click and draw on the grain in the top left and click 'Save Line'. 
- To delete a line, type the line number in 'Remove ID' and press 'Enter'.
- To clear all lines, click 'Clear All Lines'

Grouping Lines
--------------

All drawn lines are grouped into clusters based on their orientation. 
A list of all groups is shown under 'List of Groups'. 
The average orientation of each group is shown in the 'Av. Angle' columns. 
0 degrees corresponds to a slip trace pointing upwards, and angles increase anticlockwise (more information in :doc:`../userguide/conventions`).
Any slip system within +-5 degrees is shown under 'System' and the deviation away from the slip system(s) is shown nunder 'Dev'. 
When calculated, the experimental RDR is also shown under 'RDR'.

Running RDR
-----------

Type a group number into the 'Run RDR on group' field and press 'Enter' to run RDR on that group.
The following interface will load:

.. image:: /_static/rdr.png

Data Structure
--------------

The grain inspector stores data for each grain under ``points_list`` and ``groups_list`` attributes.
- The ``points_list`` contains a list of [[x1, y1, x2, y2], angle, group] each line drawn on the grain.
- The ``groups_list`` contains a list of [id, angle, active plane(s), deviation(s), rdr] corresponding to each group.

Save/Load points and groups
---------------------------

All these selections will be lost upon restarting the Python kernel.

- To save to a file, hit the 'Save File' button.
- To load from a file, hit the 'Load File' button.

The file location is relative to the DIC map directory.