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
    * - ``pyvale``
      - ``.csv``
      - PyVale text files

.. note::

    Only files from version 8 of DaVis are currently supported. 
    In David, ensure the decimal point chartacter in exported files is set to dot, 
    by going to Project → Global Options → Export → Decimal Point Character and selecting 'Dot'

Loading HRDIC Data
------------------

HRDIC data can be loaded (for example, from a LaVision DaVis test file) as follows:

.. code-block:: python

    import defdap.hrdic as hrdic
    
    # Load HRDIC map from DaVis format
    dic_map = hrdic.Map("path/to/dic_data.txt", data_type="davis")

Data Structure
--------------

The HRDIC Map stores several key data structures as a  :class:`defdap.utils.Datastore` object under the ``data`` attribute.
You can print a list of all the attributed stored ``print(dic_map.data)`` and a more detailed summmary is below:

.. list-table::
    :header-rows: 1
    :widths: 8 24

    * - Attribute
      - Description
    * - ``coordinate``
      - pixel coordinate grid for the DIC map.
    * - ``displacement``
      - displacement field arrays (first element is x, second element is y).
    * - ``e``
      - deformation gradient components (e.g. ``Exx`` is element [0][0] ``Eyy`` is element [1][1]).
    * - ``f``
      - green strain components (e.g. ``Fxx`` is element [0][0] ``Fyy`` is element [1][1]).
    * - ``max_shear``
      - maximum shear strain field.
    * - ``pattern``
      - image/pattern data associated with DIC, set this with :class:`defdap.hridic.set_pattern`.
    * - ``mask``
      - validity mask for data points.
    * - ``proxigram``
      - ???


- **Grains**: ``grains`` - grain/region objects derived from segmentation
- **Phase boundaries**: ``phase_boundaries`` - phase boundary features
- **Grain boundaries**: ``grain_boundaries`` - grain boundary features

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

    dic_map.plot_map("pattern")

Setting scale
------------------

HRDIC data is stored in a pixel-based coordinate system. 
To set the scale of the map so that a scale bar in microns is plotted when the map is plotted:

.. code-block:: python

    # Set the scale (micron per pixel)
    dic_map.set_scale(scale=2.5)

Setting crop
------------------

There are normally some anomalous points near the edges of a DIC map, 
so it is often desirable to crop the map to a region of interest, using this command:

.. code-block:: python
    
    # Crop to region of interest
    dic_map.crop(left=100, right=100, top=100, bottom=100)

Plotting and Visualization
---------------------------

To plot a maximum shear strain map, with scale bar:

.. code-block:: python

    # Plot strain field
    dic_map.plot_map('max_shear', vmin=0, vmax=0.1, plot_scale_bar=True)

Further Reading
---------------

For detailed API documentation, see :doc:`../defdap/defdap.hrdic`.