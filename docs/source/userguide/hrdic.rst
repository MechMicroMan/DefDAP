HRDIC Map (`defdap.hrdic.Map`)
===================================================

The HRDIC class in DefDAP provides tools for loading, processing, and analyzing DIC data. 

Supported Data Formats
----------------------

DefDAP supports loading data from various commercial and open-source software packages:

.. list-table::
    :header-rows: 1
    :widths: 18 12 50

    * - ``data_type``
      - Extension
      - Description
    * - ``davis``
      - ``.txt``
      - LaVision DaVis text files
    * - ``openpivbinary``
      - ``.npz``
      - OpenPIV binary  files
    * - ``openpivtext``
      - ``.txt``
      - OpenPIV text files
    * - ``pyvale-csv``
      - ``.csv``
      - PyVale text files
    * - ``pyvale-binary``
      - ``.2ddic``
      - PyVale binary files

.. note::

    Only files from version 8 of DaVis are currently supported. 
    In DaVis, ensure the decimal point chartacter in exported files is set to dot, 
    by going to Project → Global Options → Export → Decimal Point Character and selecting 'Dot'

Loading HRDIC Data
------------------

HRDIC data can be loaded (for example, from a LaVision DaVis text file) as follows:

.. code-block:: python

    import defdap.hrdic as hrdic
    
    dic_map = hrdic.Map("path/to/dic_data.txt", data_type="davis")

Data Structure
--------------

The HRDIC Map stores several key data structures as a  :class:`defdap.utils.Datastore` object under the ``data`` attribute.
You can print a list of all the attributed stored: ``print(dic_map.data)`` and a more detailed summmary is below:

.. list-table::
    :header-rows: 1
    :widths: 8 24

    * - Attribute
      - Description
    * - ``coordinate``
      - Pixel coordinate grid for the DIC map.
    * - ``displacement``
      - Displacement field arrays (first element is x, second element is y).
    * - ``e``
      - Deformation gradient components (e.g. ``Exx`` is element [0][0] ``Eyy`` is element [1][1]).
    * - ``f``
      - Green strain components (e.g. ``Fxx`` is element [0][0] ``Fyy`` is element [1][1]).
    * - ``max_shear``
      - Maximum shear strain field.
    * - ``pattern``
      - Image/pattern data associated with DIC, set this with :class:`defdap.hridic.set_pattern`.
    * - ``mask``
      - Validity mask for data points, set this with :class:`defdap.hridic.generate_mask`.

When linked to a :class:`defdap.ebsd.EBSDMap` object, the HRDIC Map also stores the following data structures:

.. list-table::
    :header-rows: 1
    :widths: 8 24

    * - Attribute
      - Description
    * - ``proxigram``
      - Distance away from grain boundary for each point in the DIC map.
    * - ``grains`` 
      - Grain map derived from EBSD map.
    * - ``phase_boundaries`` 
      - Phase boundries derived from EBSD map.
    * - ``grain_boundaries`` 
      - Grain boundaries derived from EBSD map.

Setting and plotting pattern
-----------------------------

The undeformed pattern image, from which the DIC data was derived can contain microstructural information which will be useful to link the HRDIC to EBSD data later.
First, scale the image down, ideally a factor of the interregation window size for the DIC data.
For example, if the DIC interregation window size was (16 x 16) pixels, then the pattern image should be scaled down by a factor of 16.
The path can then set with :class:`defdap.hridic.set_pattern`, where the second argument is the scaling factor of the pattern image relative to the DIC interregation window size.

.. code-block:: python

    dic_map.set_pattern("pattern_image.bmp", 1)

.. note::

    Ensure this is the same image as the one used to generate the DIC data, 
    otherwise the pattern will not be correctly aligned with the DIC data and the subsequent correlation with EBSD data will be incorrect.

.. note::

    DefDAP calculates the expected size of the pattern image based on the size of the DIC map and the scaling factor, 
    so if the pattern image is not the expected size, an error will be raised.

To plot the pattern image, use the :class:`defdap.hridic.plot_map` method with the argument 'pattern'

.. code-block:: python

    dic_map.plot_map("pattern")

Setting scale, crop and mask
----------------------------

HRDIC data is stored in a pixel-based coordinate system, with no knowledge of the physical resolution of the data.
The scale of the map can be set so that a scale bar in microns is 
plotted when the map is plotted with the argument ``plot_scale_bar=True``.
If the original image has a horizontal field width of 30 microns and a horizontal resolution of 2048 pixels, the scale can be set as follows:

.. code-block:: python

    dic_map.set_scale(scale=30/2048)

.. note::

    The sub-window size of the DIC data is automatically taken into account when setting the scale.
    For the above example, the scale of the DIC data (16 x 16 pixels) would give (30 / 2048) * 16 = 0.2344 microns per DIC pixel, 
    so a scale bar of 10 microns would be plotted as 10 / 0.2344 = 42.7 pixels long.

There are normally some anomalous points near the edges of a DIC map, 
so it is often desirable to crop the map to a region of interest, which can be done using this command:

.. code-block:: python
    
    # Crop to region of interest
    dic_map.crop(left=100, right=100, top=100, bottom=100)

Finally, a mask can be generated to identify valid and invalid points in the DIC map, using the :class:`defdap.hridic.generate_mask` method.
The boolean array passed as ``mask`` should have the same shape as the DIC map, 
where ``True`` values indicate invalid points and ``False`` values indicate valid points.
These are some examples of how to generate a mask based on the DIC data:

.. code-block:: python
    
    #To remove data points in dic_map where max_shear is above 0.8, use:
    mask = dic_map.data.max_shear > 0.8

    #To remove data points in dic_map where e11 is above 1 or less than -1, use:
    mask = (dic_map.data.e[0, 0] > 1) | (dic_map.data.e[0, 0] < -1)

    #To disable masking:
    mask = None

Plotting and Visualization
---------------------------

To plot a maximum shear strain map, with scale bar:

.. code-block:: python

    dic_map.plot_map('max_shear', vmin=0, vmax=0.1, plot_scale_bar=True)

Further Reading
---------------

For detailed API documentation, see :doc:`../defdap/defdap.hrdic`.