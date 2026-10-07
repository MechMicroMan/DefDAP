from pathlib import Path
import pickle

import numpy as np
import shapely

from defdap import feature


def create_abaqus_input(map_obj, roi, save_dir="./", phase_offset=0):
    if not isinstance(roi, feature.Rectangle):
        raise TypeError("Only rectangle features supported.")
    if roi.frame is not map_obj.frame:
        raise ValueError("`map_obj` and `roi` must be defined in same frame.")
    save_dir = Path(save_dir)

    bounds = np.array([0, roi.size[0] - 1, 0, roi.size[1] - 1]) * map_obj.scale

    umat_data = roi.get_grain_data(
        map_obj, ori_frame="spatial", ori_hex_ortho_conv="tsl"
    )[:, 1:]
    umat_data[:, :2] *= map_obj.scale
    umat_data[:, 2:5] *= 180 / np.pi
    umat_data[:, 5] += phase_offset
    umat_data = np.insert(umat_data, 5, np.arange(1, len(umat_data)+1), axis=1)
    umat_data = np.insert(umat_data, 7, 0, axis=1)

    gb_lines = roi.clip_lines(map_obj.data.simple_boundaries.lines)
    gb_lines *= map_obj.scale

    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "bounds.npy", bounds)
    np.save(save_dir / "UMAT.npy", umat_data)
    with open(save_dir / "GBlist", "wb") as f:
        pickle.dump(gb_lines.reshape((-1, 4)).tolist(), f, protocol=2)


def add_to_unique_list(unique_list, val):
     # TODO use different data structure (ordered set?) for lookup speed
    try:
        idx = unique_list.index(val)
    except ValueError:
        idx = len(unique_list)
        unique_list.append(val)
    return idx


def polygons_to_line_loops(polygons, round_decimals=4):
    points = []
    lines = []
    line_loops = []
    for polygon in polygons:
        # create list of polygon points indexed to global points list
        point_idxs = []
        for point in np.asarray(polygon.exterior.coords.xy).round(round_decimals).T:
            point = tuple(point.tolist())
            point_idx = add_to_unique_list(points, point)
            point_idxs.append(point_idx)

        # lines are between adjacent points, stored as sorted 2-tuples of 
        # point indexes in a global line list
        line_idxs = []
        for line in zip(point_idxs[:-1], point_idxs[1:]):
            line = tuple(sorted(line))
            line_idx = add_to_unique_list(lines, line)
            line_idxs.append(line_idx)

        # then the polygon is list of lines
        line_loops.append(tuple(line_idxs))

    return points, lines, line_loops


def get_boundary_lines(points, lines):
    points_array = np.array(points)
    point_min = points_array.min(axis=0)
    point_max = points_array.max(axis=0)

    boundary_lines = {}
    for label, comp, val in zip(
        ["x0", "y0", "x1", "y1"], 
        [0, 1, 0, 1], 
        np.concat((point_min, point_max))
    ):
        # find points on boundary and sort along the boundary
        b_point_idxs = np.nonzero(points_array[:, comp] == val)[0]
        b_point_idxs = b_point_idxs[
            points_array[b_point_idxs, int(not comp)].argsort()
        ].tolist()
        # lines are the ordered 2-tuples of the points indexed into global list
        boundary_lines[label] = [
            lines.index(tuple(sorted(line)))
            for line in zip(b_point_idxs[:-1], b_point_idxs[1:])
        ]

    return boundary_lines


# 11 Quasi-structured Quad
def create_gmsh_geo_2d(
    file_name, 
    points, 
    lines, 
    line_loops, 
    boundary_lines, 
    mesh_size=None, 
    mesh_algorithm=None, 
    save_dir="./",
):
    with open(Path(save_dir) / f"{file_name}.geo", "w") as f:
        f.write('SetFactory("OpenCASCADE");\n')
        f.write("\n")

        f.write("// Points\n")
        for i, point in enumerate(points, start=1):
            f.write(f"Point({i}) = {{{point[0]}, {point[1]}, 0}};\n")
        f.write("\n")

        f.write("// Lines\n")
        for i, line in enumerate(lines, start=1):
            f.write(f"Line({i}) = {{{line[0] + 1}, {line[1] + 1}}};\n")
        f.write("\n")

        f.write("// Physical lines\n")
        for i, (label, vals) in enumerate(boundary_lines.items(), start=1):
            vals_str = ", ".join(map(str, np.array(vals) + 1 ))
            f.write(f'Physical Curve("{label}", {i}) = {{{vals_str}}};\n')
        f.write("\n")

        f.write("// Line loops\n")
        for i, line_loop in enumerate(line_loops, start=1):
            line_loop_str = ", ".join(str(l+1) for l in line_loop)
            f.write(f"Curve Loop({i}) = {{{line_loop_str}}};\n")
        f.write("\n")

        f.write("// Surfaces\n")
        for i in range(1, len(line_loops)+1):
            f.write(f"Plane Surface({i}) = {{{i}}};\n")
        f.write("\n")

        f.write("// Physical surfaces\n")
        for i in range(1, len(line_loops)+1):
            f.write(f'Physical Surface("grain_{i}", {i}) = {{{i}}};\n')
        f.write("\n")

        f.write("// Mesh\n")
        if mesh_size is not None:
            f.write(f"MeshSize {{:}} = {mesh_size};\n")
        if mesh_algorithm is not None:
            f.write(f"Mesh.Algorithm = {mesh_algorithm};\n")
        f.write("Mesh 2;\n")
        f.write(f'Save "{file_name}.msh";\n')


def create_gmsh_geo_3d(
    file_name, 
    points, 
    lines, 
    line_loops, 
    boundary_lines, 
    thickness=1,
    mesh_size=None, 
    mesh_algorithm=None, 
    save_dir="./",
):
    n_points = len(points)
    n_lines = len(lines)
    n_grains = len(line_loops)
    with open(Path(save_dir) / f"{file_name}.geo", "w") as f:
        f.write('SetFactory("OpenCASCADE");\n')
        f.write("\n")

        f.write("// Points\n")
        f.write("// z- (1 to n_points)\n")
        for i, point in enumerate(points, start=1):
            f.write(f"Point({i}) = {{{point[0]}, {point[1]}, 0}};\n")
        f.write("// z+ (n_points+1 to 2*n_points)\n")
        for i, point in enumerate(points, start=n_points+1):
            f.write(f"Point({i}) = {{{point[0]}, {point[1]}, {thickness}}};\n")
        f.write("\n")

        f.write("// Lines\n")
        f.write("// z- (1 to n_lines)\n")
        for i, line in enumerate(lines, start=1):
            f.write(f"Line({i}) = {{{line[0] + 1}, {line[1] + 1}}};\n")
        f.write("// z+ (n_lines+1 to 2*n_lines)\n")
        for i, line in enumerate(lines, start=n_lines+1):
            f.write(
                f"Line({i}) = {{{line[0] + n_points + 1}, "
                f"{line[1] + n_points + 1}}};\n"
            )
        f.write("// z dir (2*n_lines+1 to 2*n_lines+n_points)\n")
        for i in range(1, n_points+1):
            f.write(f"Line({2*n_lines + i}) = {{{i}, {i + n_points}}};\n")
        f.write("\n")

        f.write("// Line loops\n")
        f.write("// z- (1 to n_grains)\n")
        for i, line_loop in enumerate(line_loops, start=1):
            line_loop_str = ", ".join(str(l+1) for l in line_loop)
            f.write(f"Curve Loop({i}) = {{{line_loop_str}}};\n")
        f.write("// z+ (n_grains+1 to 2*n_grains)\n")
        for i, line_loop in enumerate(line_loops, start=n_grains+1):
            line_loop_str = ", ".join(str(l + n_lines + 1) for l in line_loop)
            f.write(f"Curve Loop({i}) = {{{line_loop_str}}};\n")
        f.write("// z dir (2*n_grains+1 to 2*n_grains+n_lines)\n")
        for i, line in enumerate(lines, start=1):
            f.write(
                f"Curve Loop({2*n_grains + i}) = "
                f"{{{i}, {2*n_lines + line[0] + 1}, "
                f"{n_lines+i}, {2*n_lines + line[1] + 1}}};\n"
            )
        f.write("\n")

        f.write("// Surfaces\n")
        for i in range(1, 2*n_grains+n_lines+1):
            f.write(f"Plane Surface({i}) = {{{i}}};\n")
        f.write("\n")

        f.write("// Physical surfaces\n")
        physical_surfaces = {k : np.array(v) + 2*n_grains + 1 
                             for k, v in boundary_lines.items()}
        physical_surfaces.update({
            "z0": range(1, n_grains+1),
            "z1": range(n_grains+1, 2*n_grains+1),
        })
        for i, (label, vals) in enumerate(physical_surfaces.items(), start=1):
            vals_str = ", ".join(map(str, vals))
            f.write(f'Physical Surface("{label}", {i}) = {{{vals_str}}};\n')
        f.write("\n")

        f.write("// Surface loops\n")
        for i, line_loop in enumerate(line_loops, start=1):
            surface_loop = [i, n_grains + i] + [l+2*n_grains+1 for l in line_loop]
            surface_loop_str = ", ".join(map(str, surface_loop))
            f.write(f"Surface Loop({i}) = {{{surface_loop_str}}};\n")
        f.write("\n")

        f.write("// Volumes\n")
        for i in range(1, n_grains+1):
            f.write(f"Volume({i}) = {{{i}}};\n")
        f.write("\n")

        f.write("// Physical Volumes\n")
        for i in range(1, n_grains+1):
            f.write(f'Physical Volume("grain_{i}", {i}) = {{{i}}};\n')
        f.write("\n")

        f.write("// Mesh\n")
        if mesh_size is not None:
            f.write(f"MeshSize {{:}} = {mesh_size};\n")
        if mesh_algorithm is not None:
            f.write(f"Mesh.Algorithm = {mesh_algorithm};\n")
        f.write("Mesh 3;\n")
        f.write(f'Save "{file_name}.msh";\n')


def create_gmsh_geometry(
    file_name, map_obj, roi, kind="3d", **kwargs
):
    gb_lines = roi.clip_lines(
        map_obj.data.simple_boundaries.lines, add_box_lines=True
    ) * map_obj.scale

    polygons = list(shapely.polygonize(shapely.linestrings(gb_lines)).geoms)
    # order polygons to map grains
    grain_centres = roi.get_grain_data(map_obj)[:, 1:3] * map_obj.scale
    polygons_sorted = []
    for centre in grain_centres:
        for i, polygon in enumerate(polygons):
            if polygon.contains(shapely.Point(centre)):
                polygons_sorted.append(polygon)
                polygons.pop(i)
                break
        else:
            raise ValueError(f"No matching polygon for point {centre}")
    polygons = polygons_sorted

    points, lines, line_loops = polygons_to_line_loops(polygons)
    boundary_lines = get_boundary_lines(points, lines)

    create_gmsh_geo_method = {
        "2d": create_gmsh_geo_2d,
        "3d": create_gmsh_geo_3d,
    }[kind.lower()]
    create_gmsh_geo_method(
        file_name, points, lines, line_loops, boundary_lines, **kwargs
    )


def create_moose_umat_materials(
    file_name, map_obj, roi, plugin_path, save_dir="./", phase_offset=0
):
    umat_data = roi.get_grain_data(
        map_obj, ori_frame="spatial", ori_hex_ortho_conv="tsl"
    )[:, 3:]
    umat_data[:, :-1] *= 180 / np.pi
    umat_data[:, -1] += phase_offset

    with open(Path(save_dir) / f"{file_name}_mats.i", "w") as f:
        f.write("[Materials]\n")
        for i, props in enumerate(umat_data, start=1):
            f.write(f"  [grain_{i}]\n")
            f.write("    type = AbaqusUMATStress\n")
            f.write(f"    block = grain_{i}\n")
            f.write(f"    constant_properties = '{props[0]} {props[1]} "
                    f"{props[2]} {i} {props[3]} 0.'\n")
            f.write(f"    plugin = '{plugin_path}'\n")
            f.write("    num_state_vars = 1\n")
            f.write("    external_fields = 'stress_xx strain_xx stress_yy "
                    "strain_yy stress_zz strain_zz'\n")
            f.write("    use_one_based_indexing = true\n")
            f.write("  []\n")
        f.write("[]\n")
