EBSD Map (`defdap.ebsd.Map`)
===================================================

The EBSD class in DefDAP provides tools for loading, processing, and analyzing EBSD data. 

Supported Data Formats
----------------------

DefDAP supports loading data from various commercial EBSD vendors:

.. list-table::
    :header-rows: 1
    :widths: 18 12 50

    * - ``data_type``
      - Extension
      - Description
    * - ``oxfordbinary``
      - ``.cpr/.crc``
      - Oxford Instruments binary files.
    * - ``oxfordtext``
      - ``.ctf``
      - Oxford Instruments text files
    * - ``edaxang``
      - ``.ang``
      - EDAX text file

.. note::

    Oxford Instruments and EDAX use different conventions when attaching an orthonormal frame to a crystal structure. 
    More information in :doc:`../userguide/conventions`.

.. note::

    If you have issues loading one of these files, please make an 'Issue' on GitHub, including a copy of the file.
    If you are trying to load a new file type then please provide a sample file and we will try to add support for it.

Loading EBSD Data
------------------

EBSD data can be loaded (for example, from a Oxford Instruments binary file) as follows:

.. code-block:: python

    import defdap.ebsd as ebsd
    
    ebsd_map = ebsd.Map("path/to/ebsd_data.txt", data_type="oxfordbinary")

Data Structure
--------------

The EBSD Map stores several key data structures as a  :class:`defdap.utils.Datastore` object under the ``data`` attribute.
You can print a list of all the attributed stored: ``print(ebsd_map.data)``.
These structures must be present in the imported data:

.. list-table::
    :header-rows: 1
    :widths: 8 24

    * - Attribute
      - Description
    * - ``phase``
      - Phase ID map (1-based; 0 for non-indexed points).
    * - ``euler_angle``
      - Euler angles stored as (3, y, x) in radians.

These are optional, but often present.

.. list-table::
    :header-rows: 1
    :widths: 8 24
    
    * - Attribute
      - Description    
    * - ``band_contrast``
      - Band contrast map from the EBSD scan.
    * - ``band_slope``
      - Band slope map from the EBSD scan.
    * - ``mean_angular_deviation``
      - Mean angular deviation (MAD) map.

These are generated from the above data:

.. list-table::
    :header-rows: 1
    :widths: 8 24

    * - Attribute
      - Description
    * - ``orientation``
      - Quaternion map (generated from ``euler_angle``).
    * - ``grain_boundaries``
      - Grain boundary set.
    * - ``phase_boundaries``
      - Phase boundary set.
    * - ``grains``
      - Grain ID map (1-based in the map).
    * - ``KAM``
      - Kernel average misorientation (radians).
    * - ``GND``
      - Geometrically necessary dislocation density map.
    * - ``Nye_tensor``
      - 3x3 Nye tensor at each point.
    * - ``proxigram``
      - Proxigram values for each pixel.
    * - ``point``
      - Point locations used for the proxigram calculation.
    * - ``GROD``
      - Grain reference orientation deviation map.
    * - ``GROD_axis``
      - Grain reference orientation deviation axis map.
    * - ``grain_data_to_map``
      - Derived grain list data mapped back to the pixel grid.

Plotting and Visualization
---------------------------

To plot a maximum shear strain map, with scale bar:

.. code-block:: python

    ebsd_map.plot_map('band_contrast', vmin=0, vmax=0.1, plot_scale_bar=True)

Further Reading
---------------

For detailed API documentation, see :doc:`../defdap/defdap.ebsd`.