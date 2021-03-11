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
import pathlib

from typing import Type

from defdap.quat import Quat


class EBSDDataWriter(object):
    def __init__(self) -> None:
        self.metadata = {
            'shape': (0, 0),
            'step_size': 0.,
            'acquisition_rotation': Quat(1.0, 0.0, 0.0, 0.0),
            'phases': []
        }
        self.data = {
            'phase': None,
            'quat': None,
            'band_contrast': None
        }
        self.data_format = None

    @staticmethod
    def get_writer(datatype: str) -> "Type[EBSDDataLoader]":
        if datatype is None:
            datatype = "OxfordText"

        if datatype == "OxfordText":
            return OxfordTextWriter()
        else:
            raise ValueError(f"No loader for EBSD data of type {datatype}.")


class OxfordTextWriter(EBSDDataWriter):
    def write(self, file_name: str, file_dir: str = "") -> None:
        """ Write an Oxford Instruments .ctf file, which is a HKL single
        orientation file.

        Parameters
        ----------
        file_name
            File name.
        file_dir
            Path to file.

        """

        # check output file
        file_name = "{}.ctf".format(file_name)
        file_path = pathlib.Path(file_dir) / pathlib.Path(file_name)
        if file_path.exists():
            raise FileExistsError(f"File already exits {file_path}")

        shape = self.metadata['shape']
        step_size = self.metadata['step_size']

        # convert quats to Euler angles
        out_euler_array = np.zeros(shape + (3,))
        for idx in np.ndindex(shape):
            out_euler_array[idx] = self.data['quat'][idx].eulerAngles()
        out_euler_array *= 180 / np.pi
        acq_rot = self.metadata['acquisition_rotation'].eulerAngles()
        acq_rot *= 180 / np.pi

        # create coordinate grids
        x_grid, y_grid = np.meshgrid(
            np.arange(0, shape[1]) * step_size,
            np.arange(0, shape[0]) * step_size,
            indexing='xy'
        )

        with open(str(file_path), 'w') as ctf_file:
            # write header
            ctf_file.write("Channel Text File\n")
            ctf_file.write("Prj\t\n")
            ctf_file.write("Author\t\n")
            ctf_file.write("JobMode\tGrid\n")
            ctf_file.write(f"XCells\t{shape[1]}\n")
            ctf_file.write(f"YCells\t{shape[0]}\n")
            ctf_file.write(f"XStep\t{step_size :.4f}\n")
            ctf_file.write(f"YStep\t{step_size :.4f}\n")
            ctf_file.write(f"AcqE1\t{acq_rot[0]:.4f}\n")
            ctf_file.write(f"AcqE2\t{acq_rot[1]:.4f}\n")
            ctf_file.write(f"AcqE3\t{acq_rot[2]:.4f}\n")
            ctf_file.write(
                "Euler angles refer to Sample Coordinate system (CS0)!\n")
            ctf_file.write(f"Phases\t{len(self.metadata['phases'])}\n")
            for phase in self.metadata['phases']:
                dims = "{:.3f};{:.3f};{:.3f}".format(*phase.latticeParams[:3])
                angles = (f * 180 / np.pi for f in phase.latticeParams[3:])
                angles = "{:.3f};{:.3f};{:.3f}".format(*angles)

                ctf_file.write(f"{dims}\t{angles}\t{phase.name}"
                               f"\t{phase.laueGroup}\t0\t\t\t\n")

            ctf_file.write("Phase\tX\tY\tBands\tError\tEuler1\tEuler2"
                           "\tEuler3\tMAD\tBC\tBS\n")

            for x, y, phase, eulers, bc in zip(
                x_grid.flat, y_grid.flat,
                self.data['phase'].flat,
                out_euler_array.reshape((shape[0] * shape[1], 3)),
                self.data['band_contrast'].flat,
            ):
                error = 3 if phase == 0 else 0
                ctf_file.write(
                    f"{phase}\t{x:.3f}\t{y:.3f}\t10\t{error}\t{eulers[0]:.3f}"
                    f"\t{eulers[1]:.3f}\t{eulers[2]:.3f}\t0.0000\t{bc}\t0\n"
                )
