Conventions
===================================================

Spatial
----------------------

The origin of plots is in the top left, with x increasing to the right and y increasing downwards.

Orthonormal reference
----------------------

Oxford Instruments and EDAX use different conventions when attaching an orthonormal frame to a crystal structure. 
This can be set in ``defdap/__init__.py`` by changing the ``crystal_ortho_conv`` argument. The ``hkl`` 
convention is x // [10-10] and y // a2 [-12-10], whereas the ``tsl`` convention is x // a1 [2-1-10], y // [01-10]

Pole figure projection
----------------------

There are two common conventions for the pole figure projection, which can be set in ``defdap/__init__.py`` 
by changing the ``pole_projection`` argument. The default is the ``stereographic`` (equal-angle) convention, 
but the ``lambert`` (equal-area) convention is also available.

IPF Triangle
--------------

The orientation of the hexagonal IPF triangle can be set in ``defdap/__init__.py`` by changing the ``ipf_triangle_convention`` argument. 
The ``up`` and ``down`` conventions looks lke this:

.. image:: /_static/IPF_up_down.png

Slip systems
----------------------

Slip system definition files are in the ``defdap/slip_systems`` folder. 
The slip system definition file used for each crystal structure can be chosen in ``defdap/__init__.py``, under the ``slip_system_file`` argument.
By default, the FCC slip systems are defined in ``cubic_fcc.txt``, contatining [111] planes and (011) directions.
By default, the BCC slip systems are defined in ``cubic_bcc.txt``, contatining [110] planes and (111) directions, 
[112] planes and (111) directions and [312] planes and (111) directions.
By default, the HCP slip systems are defined in ``hexagonal_withca.txt``, basal <a>, prismatic <a>, pyramidal <a> and pyramidal <c+a> slip systems.

Slip trace angles
----------------------

These are calculated with the convention that 0 degrees corresponds to a slip trace pointing upwards, and angles increase anticlockwise.
