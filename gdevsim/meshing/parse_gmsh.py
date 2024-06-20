"""Adapted from J. Sanchez https://github.com/devsim/devsim_misc/blob/main/gmsh/test_convert.py

Improvements:
    - functionalize "run" (renamed to parse_gmsh)
    - parametrize interface delimiter, set default to "___"
    - type hints
    - docstrings
"""

from gdevsim.meshing.mesh_convert import (
    read_gmsh_info,
    write_elements_to_gmsh,
    write_format_to_gmsh,
    write_nodes_to_gmsh,
    write_physical_names_to_gmsh,
)


def find_interfaces(dimension, elements):
    """
    For a list of elements:
    * break them up by physical number
    * find intersections in each region
    """
    if dimension not in (2, 3):
        raise RuntimeError(f"Unexpected Dimension {dimension}")
    set_dict = {}
    for t in elements:
        pnum = (t[-2], t[-1])
        set_dict.setdefault(pnum, set())
        the_set = set_dict[pnum]
        if dimension == 3:  # tetrahedra volumes
            n = sorted(t[0:4])
            tuples_to_add = [tuple(n[i : i + 3]) for i in range(4)]
        elif dimension == 2:  # triangle volumes
            n = sorted(t[0:3])
            tuples_to_add = [tuple(n[i : i + 2]) for i in range(3)]
        for u in tuples_to_add:
            the_set.discard(u) if u in the_set else the_set.add(u)
    pnums = sorted(set_dict.keys())
    boundaries = {
        (pnum_i, pnums[j]): sorted(set_dict[pnum_i].intersection(set_dict[pnums[j]]))
        for i, pnum_i in enumerate(pnums)
        for j in range(i + 1, len(pnums))
        if pnum_i[0] != pnums[j][0]
        and set_dict[pnum_i].intersection(set_dict[pnums[j]])
    }
    return boundaries


def delete_coordinates(dimension, coordinates, surfaces, volumes):
    """
    Given a list of coordinates, surface and volume elements:
    * convert elements into vertices
    * renumber coordinates with orphaned vertices removed
    """
    cmap = [None] * (len(coordinates) + 1)
    vertices = set()

    slice_surface = slice(0, 2) if dimension == 2 else slice(0, 3)
    slice_volume = slice(0, 3) if dimension == 2 else slice(0, 4)

    if dimension not in (2, 3):
        raise RuntimeError(f"Unhandled Dimension {dimension}")

    for t in surfaces:
        vertices.update(t[slice_surface])
    for t in volumes:
        vertices.update(t[slice_volume])

    vertices = sorted(vertices)

    for i, j in enumerate(vertices, 1):
        cmap[j] = i

    new_coordinates = [None] * len(vertices)
    for i, j in enumerate(vertices, 1):
        c = coordinates[j - 1].split()
        c[0] = str(i)
        new_coordinates[i - 1] = " ".join(c)

    new_surfaces = [None] * len(surfaces)
    for i, t in enumerate(surfaces):
        nv = [cmap[x] for x in t[slice_surface]]
        nv.extend(t[-2:])
        new_surfaces[i] = tuple(nv)

    new_volumes = [None] * len(volumes)
    for i, t in enumerate(volumes):
        nv = [cmap[x] for x in t[slice_volume]]
        nv.extend(t[-2:])
        new_volumes[i] = tuple(nv)

    return new_coordinates, new_surfaces, new_volumes


def get_next_elem_id(elem_ids):
    """
    return a new unique elem id
    """
    return max(elem_ids) + 1


def get_next_phys_id(pname_map):
    """
    return a new unique phys id
    """
    return max(pname_map.keys()) + 1


def delete_region_elements(dimension, pname_map, name, elements):
    """
    filter out elements from deleted regions
    """
    for k, v in pname_map.items():
        if v[1] == name:
            d = v[0]
            pnum = k
    if d != dimension:
        raise RuntimeError(f"Expecting {name} to have dimension {dimension}")
    return [x for x in elements if x[-2] != pnum]


def get_name(name0, name1, name_priority, interface_names, interface_delimiter="___"):
    """
    return name of added interface
    picks names based on yaml file or priority index
    """
    for i in interface_names:
        if name0 in i["regions"] and name1 in i["regions"]:
            return i["interface"]
    if name0 in name_priority and name1 in name_priority:
        return (
            f"{name0}{interface_delimiter}{name1}"
            if name_priority.index(name0) < name_priority.index(name1)
            else f"{name1}{interface_delimiter}{name0}"
        )
    return "%s_%s" % tuple(sorted([name0, name1]))


def process_elements(elements):
    """
    converts input tetrahedra from strings to ints
    gets unique set of elementary ids
    """
    int_elements = []
    elem_ids = set()
    for t in elements:
        ints = [int(x) for x in t]
        if ints[-2] != 0:
            int_elements.append(ints[1:])
            elem_ids.add(ints[-1])
    return int_elements, elem_ids


def get_pname_map(gmsh_pnames):
    """
    processes physical names from mesh format
    """
    pname_map = {}
    for p in gmsh_pnames:
        data = p.split()
        pname_map[int(data[1])] = (int(data[0]), data[2][1:-1])
    return pname_map


def get_interface_map(
    dimension,
    interfaces,
    pname_map,
    elem_ids,
    name_priority,
    interface_names,
    existing_interfaces,
    interface_delimiter="___",
):
    """
    names for new interfaces
    """
    interface_map = {}

    sl = slice(0, 3) if dimension == 3 else slice(0, 2)

    if dimension not in (2, 3):
        raise RuntimeError(f"Unhandled Dimension {dimension}")

    new_priority = []

    for ei in existing_interfaces:
        phys_id = ei[-2]
        elem_id = ei[-1]
        if phys_id not in pname_map:
            continue
        pname = pname_map[phys_id][1]
        if pname not in interface_map:
            interface_map[pname] = {"phys_id": phys_id, "elem_id": {}}
            new_priority.append(pname)
        if elem_id not in interface_map[pname]["elem_id"]:
            interface_map[pname]["elem_id"][elem_id] = []
        interface_map[pname]["elem_id"][elem_id].append(sorted(ei[sl]))

    new_priority.extend(name_priority)

    for i in sorted(interfaces.keys()):
        interface = interfaces[i]
        new_elem_id = get_next_elem_id(elem_ids)
        elem_ids.add(new_elem_id)

        name0 = pname_map[i[0][0]][1]
        name1 = pname_map[i[1][0]][1]

        new_name = get_name(
            name0,
            name1,
            name_priority,
            interface_names,
            interface_delimiter=interface_delimiter,
        )
        if new_name not in interface_map:
            phys_id = get_next_phys_id(pname_map)
            pname_map[phys_id] = (dimension - 1, new_name)
            interface_map[new_name] = {"phys_id": phys_id, "elem_id": {}}

        interface_map[new_name]["elem_id"][new_elem_id] = interface
        print(
            f'INFO: Adding to interface "{new_name}" with physical id "{phys_id}" from intersecting surface of {i}'
        )
    return interface_map, new_priority


def get_surface_elements(interface_map):
    """
    gets all of the surface elements based on vertices
    they are close to the form to being written out
    """
    elements = []
    for i in sorted(interface_map.keys()):
        phys_id = interface_map[i]["phys_id"]
        print(f"{i} {phys_id}")
        for elem_id in sorted(interface_map[i]["elem_id"].keys()):
            ielements = interface_map[i]["elem_id"][elem_id]
            print(f"  {elem_id} {len(ielements)}")
            for t in ielements:
                u = list(t)
                u.extend([phys_id, elem_id])
                elements.append(tuple(u))
    return elements


def fix_surface_conflicts(dimension, surfaces, pname_map, name_priority):
    nmap = {}
    data = {}
    for phys_id, info in pname_map.items():
        if info[0] == (dimension - 1):
            data[phys_id] = []
            nmap[info[1]] = phys_id
    for s in surfaces:
        data[s[-2]].append(s)

    sl = slice(0, 3) if dimension == 3 else slice(0, 2)

    if dimension not in (2, 3):
        raise RuntimeError(f"Unhandled Dimension {dimension}")

    all_vertexes = set()
    priority_vertexes = {}
    errors = ""
    for n in name_priority:
        if n not in nmap:
            continue
        nid = nmap[n]
        nset = set()
        for s in data[nid]:
            nset.update(s[sl])
        intersection = all_vertexes.intersection(nset)
        if intersection:
            for lid, lvertexes in priority_vertexes.items():
                tmp = lvertexes.intersection(nset)
                if tmp:
                    hpname = pname_map[lid][1]
                    errors += f'WARNING: boundaries "{n}" and "{hpname}" are touching at {len(tmp)} nodes\n'
        priority_vertexes[nid] = nset
        all_vertexes |= nset
    if errors:
        errors += "WARNING: this may cause issues when the boundaries are solving the same equations on the same regions\n"
        print(errors)

    new_surfaces = []
    removed_surfaces = set()
    for phys_id, elements in data.items():
        if phys_id in priority_vertexes:
            new_surfaces.extend(elements)
            continue

        other_boundaries = set()
        local_new_elements = []
        local_vertexes = set()

        for surface in elements:
            nset = set(surface[sl])
            if nset.intersection(all_vertexes):
                for lid, lvertexes in priority_vertexes.items():
                    tmp = lvertexes.intersection(nset)
                    if tmp:
                        other_boundaries.add(f'"{pname_map[lid][1]}"')
            else:
                local_new_elements.append(surface)
                local_vertexes |= nset
        new_surfaces.extend(local_new_elements)
        all_vertexes |= local_vertexes
        priority_vertexes[phys_id] = local_vertexes
        kept = len(local_new_elements)
        removed = len(elements) - kept
        if removed > 0:
            print(
                f'INFO: removed {removed}/{removed+kept} elements from generated surface "{pname_map[phys_id][1]}" for overlap with {", ".join(other_boundaries)}'
            )
        if kept == 0:
            print(
                f'INFO: generated surface "{pname_map[phys_id][1]}" removed for 0 elements'
            )
            removed_surfaces.add(phys_id)
    return new_surfaces, removed_surfaces


def delete_regions(
    dimension, regions_to_delete, pname_map, coordinates, surfaces, volumes
):
    """
    delete volume elements from specified regions
    then remove unneeded coordinates
    """
    new_volumes = volumes[:]
    for r in regions_to_delete:
        new_volumes = delete_region_elements(dimension, pname_map, r, new_volumes)

    return delete_coordinates(dimension, coordinates, surfaces, new_volumes)


def scale_coordinates(coordinates, scale):
    """
    constant scale on all coordinate positions
    """
    new_coordinates = [None] * len(coordinates)
    for i, c in enumerate(coordinates):
        e = c.split()
        v = [scale * float(x) for x in e[1:]]
        new_coordinates[i] = e[0] + " " + " ".join([f"{x:1.15g}" for x in v])
    return new_coordinates


def get_fix_interfaces(
    input_mesh, output_mesh, name_priority=None, interface_delimiter="___"
):
    gmshinfo = read_gmsh_info(input_mesh)

    yaml_map = {i: [] for i in ("name_priority", "interfaces", "contact_regions")}

    outfile = output_mesh

    if gmshinfo["tetrahedra"]:
        dimension = 3
        volumes, elem_ids = process_elements(gmshinfo["tetrahedra"])
        existing_surfaces, existing_surface_ids = process_elements(
            gmshinfo["triangles"]
        )
        elem_ids |= existing_surface_ids
    elif gmshinfo["triangles"]:
        dimension = 2
        volumes, elem_ids = process_elements(gmshinfo["triangles"])
        existing_surfaces, existing_surface_ids = process_elements(gmshinfo["edges"])
        elem_ids |= existing_surface_ids
    else:
        raise RuntimeError("Could not find 2D or 3D elements in mesh")

    interfaces = find_interfaces(dimension, volumes)

    pname_map = get_pname_map(gmshinfo["pnames"])

    interface_names = yaml_map["interfaces"]

    interface_map, interface_priority = get_interface_map(
        dimension,
        interfaces,
        pname_map,
        elem_ids,
        name_priority,
        interface_names,
        existing_surfaces,
        interface_delimiter=interface_delimiter,
    )

    surfaces = get_surface_elements(interface_map)

    surfaces, removed_surface_ids = fix_surface_conflicts(
        dimension, surfaces, pname_map, interface_priority
    )

    pnames = [
        f'{pname_map[i][0]} {i} "{pname_map[i][1]}"'
        for i in sorted(pname_map.keys())
        if i not in removed_surface_ids
    ]

    regions_to_delete = [
        x["contact"] for x in yaml_map["contact_regions"] if x["remove"]
    ]

    coordinates = gmshinfo["coordinates"]
    coordinates, surfaces, volumes = delete_regions(
        dimension, regions_to_delete, pname_map, coordinates, surfaces, volumes
    )

    with open(outfile, "w") as ofh:
        write_format_to_gmsh(ofh)
        write_physical_names_to_gmsh(ofh, pnames)
        write_nodes_to_gmsh(ofh, coordinates)
        if dimension == 2:
            write_elements_to_gmsh(ofh, surfaces, volumes, [])
        elif dimension == 3:
            write_elements_to_gmsh(ofh, [], surfaces, volumes)
        else:
            raise RuntimeError(f"Unhandled Dimension {dimension}")
