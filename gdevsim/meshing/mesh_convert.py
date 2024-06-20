"""Adapted from J. Sanchez https://github.com/devsim/devsim_misc/blob/main/gmsh/mesh_convert.py

Improvements:
    - handle files with Pathlib
    - type hints
    - docstrings
"""

def read_nodes_from_tetgen(ifh):
    """
    read node
    """
    ncoordinates = None
    coordinates = []
    for line in ifh:
        line = line.strip()
        if line and line[0] == "#":
            pass
        elif not ncoordinates:
            ncoordinates = int(line.split()[0])
        else:
            coordinates.append(line)
    if ncoordinates != len(coordinates):
        raise RuntimeError("issue reading coordinates")
    return coordinates


def read_physical_names(ifh):
    """
    read physical names from gmsh
    """
    nphysicalnames = None
    inphysicalnames = False
    physicalnames = []

    for line in ifh:
        line = line.strip()
        if not inphysicalnames:
            if line.find("$PhysicalNames") == 0:
                inphysicalnames = True
                continue
        elif line.find("$EndPhysicalNames") == 0:
            break
        else:
            if not nphysicalnames:
                nphysicalnames = int(line)
            else:
                physicalnames.append(line)
    return physicalnames


def read_nodes_from_gmsh(ifh):
    """
    read physical names from gmsh
    """
    ncoordinates = None
    coordinates = []
    innodes = False

    for line in ifh:
        line = line.strip()

        if not innodes:
            if line.find("$Nodes") == 0:
                innodes = True
                continue
        elif line.find("$EndNodes") == 0:
            break
        elif not ncoordinates:
            ncoordinates = int(line)
        else:
            coordinates.append(line)

    if ncoordinates != len(coordinates):
        raise RuntimeError("issue reading coordinates")

    return coordinates


def read_elements_from_gmsh(ifh):
    """
    read elements from gmsh
    """
    nelements = None
    edges = []
    triangles = []
    tetrahedra = []
    inelements = False

    for line in ifh:
        line = line.strip()

        if not inelements:
            if line.find("$Elements") == 0:
                inelements = True
                continue
        elif line.find("$EndElements") == 0:
            break
        elif not nelements:
            nelements = int(line)
        else:
            data = line.split()
            skip = int(data[2])
            outdata = data[2 + skip :]
            # put physical region at the end
            outdata.append(data[3])
            # plus the elementary id
            outdata.append(data[4])
            etype = int(data[1])
            if etype == 1:
                edges.append(outdata)
            elif etype == 2:
                triangles.append(outdata)
            elif etype == 4:
                tetrahedra.append(outdata)
            else:
                raise RuntimeError("Issue reading elements gmsh type %d" % etype)
    return (edges, triangles, tetrahedra)


# we only want the triangles on a marked boundary


def read_triangles_from_tetgen(ifh):
    """
    read triangles from tetgen
    """
    ntriangles = None
    triangles = []
    for line in ifh:
        line = line.strip()
        if line and line[0] == "#":
            pass
        elif not ntriangles:
            ntriangles = int(line.split()[0])
        else:
            data = line.split()
            # we are in a group
            if data[-1] != "0":
                triangles.append(data[1:5])
    return triangles


def read_tetrahedra_from_tetgen(ifh):
    """
    read tetrahedra from tetgen
    """
    ntetrahedra = None
    tetrahedra = []
    for line in ifh:
        line = line.strip()
        if line and line[0] == "#":
            pass
        elif not ntetrahedra:
            ntetrahedra = int(line.split()[0])
        else:
            data = line.split()
            # we are in a group
            if data[-1] != "0":
                tetrahedra.append(data[1:6])
    return tetrahedra


def write_format_to_gmsh(ofh):
    """
    write gmsh file header
    """
    ofh.write(
        """\
$MeshFormat
2.2 0 8
$EndMeshFormat
"""
    )


def write_nodes_to_gmsh(ofh, coordinates):
    """
    write coordinates to gmsh
    """
    ofh.write("$Nodes\n%d\n" % len(coordinates))
    for coordinate in coordinates:
        ofh.write(coordinate + "\n")
    ofh.write("$EndNodes\n")


def write_elements_to_gmsh(ofh, edges, triangles, tetrahedra):
    """
    write element list to gmsh
    """
    nelements = len(edges) + len(triangles) + len(tetrahedra)
    ofh.write("$Elements\n%d\n" % nelements)
    index = 1
    for edge in edges:
        ofh.write("%d 1 2 %s %s " % (index, str(edge[-2]), str(edge[-1])))
        ofh.write(" ".join(str(x) for x in edge[0:2]))
        ofh.write("\n")
    for triangle in triangles:
        ofh.write("%d 2 2 %s %s " % (index, str(triangle[-2]), str(triangle[-1])))
        ofh.write(" ".join(str(x) for x in triangle[0:3]))
        ofh.write("\n")
        index += 1
    for tetrahedron in tetrahedra:
        ofh.write("%d 4 2 %s %s " % (index, str(tetrahedron[-2]), str(tetrahedron[-1])))
        ofh.write(" ".join(str(x) for x in tetrahedron[0:4]))
        ofh.write("\n")
        index += 1
    ofh.write("$EndElements\n")


def write_physical_names_to_gmsh(ofh, physical_names):
    """
    write physical names go gmsh file
    """
    ofh.write("$PhysicalNames\n%d\n" % len(physical_names))
    for pname in physical_names:
        ofh.write(pname + "\n")
    ofh.write("$EndPhysicalNames\n")


def tetgen_to_gmsh(basename, gmshname, groupsname):
    """
    convert from tetgen .node, .face, .ele files to gmsh format
    """
    with open(basename + ".node") as ifh:
        coordinates = read_nodes_from_tetgen(ifh)
    with open(basename + ".face") as ifh:
        triangles = read_triangles_from_tetgen(ifh)
    with open(basename + ".ele") as ifh:
        tetrahedra = read_tetrahedra_from_tetgen(ifh)
    if groupsname:
        with open(groupsname) as ifh:
            physical_names = read_physical_names(ifh)
    with open(gmshname, "w") as ofh:
        write_format_to_gmsh(ofh)
        write_physical_names_to_gmsh(ofh, physical_names)
        write_nodes_to_gmsh(ofh, coordinates)
        edges = []
        write_elements_to_gmsh(ofh, edges, triangles, tetrahedra)


def gmsh_to_tetgen(gmshname, basename, groupsname):
    """
    convert to gmsh to tetgen format
    """
    with open(gmshname) as ifh:
        pnames = read_physical_names(ifh)
        if groupsname:
            with open(groupsname, "w") as ofh:
                write_physical_names_to_gmsh(ofh, pnames)
        ifh.seek(0)
        coordinates = read_nodes_from_gmsh(ifh)
        with open(basename + ".node", "w") as ofh:
            ofh.write("%d 3 0 0\n" % len(coordinates))
            for node in coordinates:
                ofh.write(node + "\n")
        ifh.seek(0)
        _, triangles, tetrahedra = read_elements_from_gmsh(ifh)
        with open(basename + ".face", "w") as ofh:
            ofh.write("%d 1\n" % len(triangles))
            for triangle in triangles:
                ofh.write(" ".join(triangle))
                ofh.write("\n")
        with open(basename + ".ele", "w") as ofh:
            ofh.write("%d 1\n" % len(tetrahedra))
            for tetrahedron in tetrahedra:
                ofh.write(" ".join(tetrahedron))
                ofh.write("\n")


def read_gmsh_info(gmshname):
    """
    read entire gmsh file
    """
    with open(gmshname) as ifh:
        pnames = read_physical_names(ifh)
        ifh.seek(0)
        coordinates = read_nodes_from_gmsh(ifh)
        ifh.seek(0)
        edges, triangles, tetrahedra = read_elements_from_gmsh(ifh)
    return {
        "pnames": pnames,
        "coordinates": coordinates,
        "edges": edges,
        "triangles": triangles,
        "tetrahedra": tetrahedra,
    }
