# Change Log

## Current

### Added

### Changed

### Fixed


## 1.0.1 (03-12-2025)

### Fixed
- Plotting slip traces for ebsd grain with colours from phase


## 1.0.0 (05-11-2025)
This was a overhaul of large sections of the code and many changes/additions have been missed from the chnagelog.

### Added
- Each grain is assigned a phase and slip systems are automatically loaded 
  for a given phase based on crystal structure. 
  - This means that unit cells and slip traces plot correctly for grains
  in a multi-phase EBSD map
- Add slip system file for FCC but in same order as DAMASK
- Use example_notebook to generate a 'How To Use' page in the documentation
- Add reader for EDAX .ang EBSD files, pyvale .csv files and openPIV-XL .npx files
- Added a `plot_map` function for grains
- Added more testing

### Changed
- All functions and arguments are now in snake_case instead of CamelCase
- Cropping and masking are now performed upon access to data
- Changed function names from CamelCase to snake_case
- Overhaul of data storage in the Map classes
- RDR calculation `calcRDR` in grain inspector is faster and more robust
- Improve formatting of grain inspector and RDR plot window
- Refactor boundary lines calculations
- Use GitHub actions to run `pytest` on commit push or pull request

### Fixed
- Fix bug in grain inspector (`None` passed to `corrAngle` inadvertently)
- Fix EBSD grain linker
- Remove `IPython` and `jupyter` as requirements
- Bug in IPF traiangle plotting now fixed with options for `up` triangle (like MTEX) and `down` triangle (like OI)


## 0.93.5 (20-11-2023)

### Added
- Add more options for colouring lines

### Fixed
- Fix bug with accessing slip systems in grain inspector
- Replace np.float with python float
- Remove in_place argument to skimage.morphology.remove_small_objects
- set_window_title has been moved from figure.canvas to figure.canvas.manager


## 0.93.5 (20-11-2023)

### Added
- Add more options for colouring lines

### Fixed
- Fix bug with accessing slip systems in grain inspector
- Replace `np.float` with python `float`
- Remove `in_place` argument to `skimage.morphology.remove_small_objects`
- `set_window_title` has been moved from `figure.canvas` to `figure.canvas.manager`


## 0.93.5 (17-01-2023)

### Added
- Add equality check for `Quat`

### Fixed
- Fix bug in boundary line warping


## 0.93.4 (07-03-2022)

### Changed
- Update BCC slip system file and add a separate FCC file with same ordering as in DAMASK
- Refactor boundary lines in `ebsd.map` class and add methods for warping lines to a DIC map
- Refactor `linkEbsdMap` method and pass all arguments to transform estimate method
- Remove IPython and jupyter as requirements
- Move slip systems to `Phase` class and load automatically based on crystal stucture
- Make Oxford bonary loader tolerate of unknown data fields

### Fixed
- Fix ebsd grain linker so it works again


## 0.93.3 (23-08-2021)

### Added
- Store EDX data as a dictionary called `EDX` accessible from the EBSD `Map` object
- Add option to change IPF map background colour

### Fixed
- Fix bug with reading cpr EBSD file without EDX data
- Fix issue with plotting Schmid factor maps
- Fix bug with maps `component` not updating after masking


## 0.93.2 (16-04-2021)

### Added
- Reading of Channel5 project files that contain EDX data

### Fixed
- Plotting unit cells
- Plotting lines with grain inspector


## 0.93.1 (12-04-2021)

### Added
- Started adding type hinting
- Add save/load support to grainInspector
- Add `drawLineProfile` function. Click and drag a line, then plot an intensity profile across the line

### Changed
- Speed up EBSD map data rotation 
- Speed up `warp` grain finding
- Store band slope and MAD arrays from EBSD
- Update `nCorrToDavis.m` script
  - Better description of how to use the function
  - Sub-window size is subset radius * 2, as defined in nCorr
  - Subset spacing is subset spacing, as defined in nCorr
- Generate phase boundary points and lines at same time as grain boundaries
- Improve histogram plotting
  - Options for scatter (as before), `step` and `bar`
  - Options for `logx`, `logy`, `loglog` and `linear`
- Updates to example notebook

### Fixed 
- Fixed docstring links
- Fix bug in `warp` grain finding algorithm


## 0.93.0 (20-02-2021)

### Added
- Add EBSD file writing to CTF format.
- Add cretation of EBSD maps from runtime data.
- Add method to generate all slip systems in a family from a single system.
- Add grouping of slip systems by family.
- Add mechanism to define default parameter values stored in single loaction.
- Add basic filtering for HRDIC maps.
  - Filtering based on threshold of effective shear strain and subsequent binary dilation.
  - All current DefDAP functions work with NaN, but the RDR might not work as expected.
- New grain finding algorithm for HRDIC map which warps the EBSD grain map.
- Add `addLegend` command to add a marker size legend to a pole plot.
- Add misorientation calculation between neighbouring EBSD grains.
- Add a `BoundarySegment` class to represent a section of grain boundary between 2 grain in an EBSD map. Objects of this class are assigned to edges of the neighbour network and contain all the boundary points between the grains.
- Add Kuwahara filter for EBSD map noise reduction.
- Add `shape` property to maps.
- Read EBSD phases from file.
- Add classes to represent phases and crystal structures.

### Changed
- Update progress reporting to print elapsed time.
- Speed up grain finding algorithm.
- Update plot IPF and Euler map to consider multiple phasess.
- Drop support of python 3.5.
- Update boundary and grain finding to consider phase boundaries. 
- Assign a phase to each grain in an EBSD map.
- Change equality and hash of slip systems, slip plane and direction must now be equal (-ve allowed but not different norms).
- Update slip system plane and direction lables  to have overbars - very fancy.
- Updates to example notebook.
- Change docs over to readthedocs.
- Move version number to own file.
- Update neighbour network to use grains as nodes.
- Store `grainID` in grain objects.
- Split plotGrainDataMap into separate array construction and plotting function.
- Update neighbour network construction to use new EBSD boundary definition.
- Update flood fill algorithm for grain finding in a EBSD map.
- Vertical and horizontal boundary pixels are now considered separately.
- Load phases from CTF file.

### Fixed 
- Fix bug with comment and blank lines when reading CPR files.
- Fixes to quat class:
  - Construction from axis/angle now in passive sense not just a conversion.
  - Fix big with transform vector where -ve vector was returned in some cases due to intermediate quat being transformed to northern hemisphere.
