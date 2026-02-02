"""
Export 1 file DXF với 2 layer (AIR, IRON) + file JSON chứa centroid.

Workflow:
1. Chạy file này để tạo combined_regions.dxf + centroids.json
2. Chạy plot_fem.py để import vào FEMM và tự động đặt block labels
"""

import os
import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import warnings

warnings.filterwarnings('ignore')

DXF_SCALE = 1000.0


class Material(Enum):
    IRON = 1
    AIR = 0


# ============================================================================
# TRA Parsing
# ============================================================================
def _find_line_idx(lines: List[str], needle: str) -> int:
    for i, s in enumerate(lines):
        if needle in s:
            return i
    raise ValueError(f"Could not find: {needle!r}")


def parse_tra_nodes(path: str) -> Dict[int, Tuple[float, float]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    coord_idx = _find_line_idx(lines, "Coordinates of the nodes")
    nodes = {}
    for s in lines[coord_idx + 1:]:
        parts = s.split()
        if len(parts) < 3:
            break
        try:
            nid = int(parts[0])
            x = float(parts[1].replace("D", "E"))
            y = float(parts[2].replace("D", "E"))
            nodes[nid] = (x, y)
        except ValueError:
            break
    return nodes


@dataclass
class Tri6Element:
    eid: int
    region: int
    n1: int
    n2: int
    n3: int
    n4: int
    n5: int
    n6: int
    material: Material = Material.IRON

    @property
    def corner_nodes(self):
        return [self.n1, self.n2, self.n3]


def parse_tra_elements(path: str) -> List[Tri6Element]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    eidx = _find_line_idx(lines, "Description of elements")
    cidx = _find_line_idx(lines, "Coordinates of the nodes")
    elements = []
    i = eidx + 1
    while i < cidx:
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        parts = s.split()
        if len(parts) >= 3:
            try:
                eid, nper, region = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                i += 1
                continue
            if nper == 6 and i + 1 < cidx:
                conn = lines[i + 1].split()
                if len(conn) >= 6:
                    n1, n2, n3, n4, n5, n6 = [int(conn[j]) for j in range(6)]
                    elements.append(Tri6Element(eid=eid, region=region,
                                                n1=n1, n2=n2, n3=n3, n4=n4, n5=n5, n6=n6))
                i += 2
                continue
        i += 1
    return elements


# ============================================================================
# NGnet
# ============================================================================
@dataclass
class NGnetConfig:
    r_min: float
    r_max: float
    theta_min: float
    theta_max: float
    n_radial: int = 9
    n_angular: int = 6


class NGnet:
    def __init__(self, config: NGnetConfig):
        self.config = config
        self.centers = []
        self.weights = None
        cfg = config
        r_vals = np.linspace(cfg.r_min, cfg.r_max, cfg.n_radial)
        t_vals = np.linspace(cfg.theta_min, cfg.theta_max, cfg.n_angular)
        dr = (cfg.r_max - cfg.r_min) / max(cfg.n_radial - 1, 1)
        self.sigma = dr * 0.8
        for r in r_vals:
            for t in t_vals:
                self.centers.append((r * np.cos(t), r * np.sin(t)))
        self.weights = np.zeros(len(self.centers))
        print(f"NGnet: {len(self.centers)} centers")

    def set_random_weights(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.weights = np.random.uniform(-1, 1, len(self.centers))

    def phi(self, x, y):
        g = [np.exp(-((x-cx)**2 + (y-cy)**2) / (2*self.sigma**2)) for cx, cy in self.centers]
        s = sum(g)
        if s < 1e-12:
            return 0
        return sum(w * gi / s for w, gi in zip(self.weights, g))

    def get_material(self, x, y):
        return Material.IRON if self.phi(x, y) >= 0 else Material.AIR

    def is_in_design_region(self, x, y):
        cfg = self.config
        r = np.sqrt(x**2 + y**2)
        t = np.arctan2(y, x)
        if t < 0:
            t += 2 * np.pi
        return cfg.r_min <= r <= cfg.r_max and cfg.theta_min <= t <= cfg.theta_max


def assign_materials(elements, nodes, ngnet):
    for e in elements:
        cx = sum(nodes[n][0] for n in e.corner_nodes) / 3
        cy = sum(nodes[n][1] for n in e.corner_nodes) / 3
        if ngnet.is_in_design_region(cx, cy):
            e.material = ngnet.get_material(cx, cy)
        else:
            e.material = Material.IRON


# ============================================================================
# Geometry helpers
# ============================================================================
def element_to_polygon(elem, nodes, scale):
    coords = [(nodes[n][0] * scale, nodes[n][1] * scale) for n in elem.corner_nodes]
    coords.append(coords[0])
    try:
        p = Polygon(coords)
        return p.buffer(0) if not p.is_valid else p
    except:
        return None


def mirror_mesh(nodes: Dict[int, Tuple[float, float]], elements: List[Tri6Element]):
    """Mirror mesh across y=x line: (x,y) -> (y,x).

    Returns:
        Tuple[mirrored_nodes, mirrored_elements]: New nodes and elements with mirrored coordinates.
        Node IDs in mirrored mesh are offset by max_node_id to avoid conflicts.
    """
    if not nodes:
        return {}, []

    # Offset for new node IDs
    max_node_id = max(nodes.keys())
    offset = max_node_id + 1

    # Mirror nodes: (x, y) -> (y, x)
    mirrored_nodes = {}
    for nid, (x, y) in nodes.items():
        new_nid = nid + offset
        mirrored_nodes[new_nid] = (y, x)  # Swap x and y

    # Mirror elements: update node references
    mirrored_elements = []
    for e in elements:
        new_elem = Tri6Element(
            eid=e.eid + len(elements),
            region=e.region,
            n1=e.n1 + offset,
            n2=e.n2 + offset,
            n3=e.n3 + offset,
            n4=e.n4 + offset,
            n5=e.n5 + offset,
            n6=e.n6 + offset,
            material=e.material,  # Copy material from original
        )
        mirrored_elements.append(new_elem)

    return mirrored_nodes, mirrored_elements


def mirror_polygon_45(poly):
    """Mirror across y=x line: (x,y) -> (y,x)"""
    if poly.is_empty or not poly.is_valid:
        return poly
    ext = [(y, x) for x, y in poly.exterior.coords]
    ints = [[(y, x) for x, y in interior.coords] for interior in poly.interiors]
    try:
        m = Polygon(ext, ints)
        return m.buffer(0) if not m.is_valid else m
    except:
        return Polygon()


def polygon_to_list(geom):
    """Convert Polygon/MultiPolygon to list of Polygons"""
    if isinstance(geom, Polygon):
        return [geom] if geom.is_valid and not geom.is_empty else []
    elif isinstance(geom, MultiPolygon):
        return [p for p in geom.geoms if p.is_valid and not p.is_empty]
    return []


def cluster_adjacent_elements(elements: List[Tri6Element], material: Material) -> List[List[Tri6Element]]:
    """Cluster elements of same material by adjacency (shared edges).

    Two triangles are adjacent if they share at least 2 corner nodes (an edge).
    Returns list of clusters, each cluster is a list of connected elements.
    """
    # Filter elements by material
    mat_elements = [e for e in elements if e.material == material]
    if not mat_elements:
        return []

    # Build adjacency: two elements are adjacent if they share >= 2 corner nodes
    n = len(mat_elements)
    adj = [[] for _ in range(n)]

    for i in range(n):
        nodes_i = set(mat_elements[i].corner_nodes)
        for j in range(i + 1, n):
            nodes_j = set(mat_elements[j].corner_nodes)
            if len(nodes_i & nodes_j) >= 2:  # Share an edge
                adj[i].append(j)
                adj[j].append(i)

    # BFS to find connected components
    visited = [False] * n
    clusters = []

    for start in range(n):
        if visited[start]:
            continue
        # BFS
        cluster = []
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            cluster.append(mat_elements[node])
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        clusters.append(cluster)

    return clusters


# ============================================================================
# DXF Export
# ============================================================================
def write_combined_dxf(out_path: str, layers_data: Dict[str, List[Polygon]]):
    """Write multiple layers to a single DXF file.

    Args:
        out_path: Output DXF path
        layers_data: Dict mapping layer_name -> list of polygons
    """

    def polyline_dxf(coords, layer_name):
        if len(coords) < 3:
            return []
        lines = ["0", "POLYLINE", "8", layer_name, "66", "1", "70", "1",
                 "10", "0.0", "20", "0.0", "30", "0.0"]
        for x, y in coords:
            lines.extend(["0", "VERTEX", "8", layer_name,
                         "10", f"{x:.12g}", "20", f"{y:.12g}", "30", "0.0"])
        lines.extend(["0", "SEQEND"])
        return lines

    # Header
    content = [
        "0", "SECTION", "2", "HEADER",
        "9", "$ACADVER", "1", "AC1009",
        "0", "ENDSEC",
        # Tables
        "0", "SECTION", "2", "TABLES",
        "0", "TABLE", "2", "LTYPE", "70", "1",
        "0", "LTYPE", "2", "CONTINUOUS", "70", "0", "3", "Solid", "72", "65", "73", "0", "40", "0.0",
        "0", "ENDTAB",
    ]

    # Layer table - add all layers
    layer_names = list(layers_data.keys())
    content.extend(["0", "TABLE", "2", "LAYER", "70", str(len(layer_names))])
    colors = {"AIR": "7", "IRON": "5"}  # 7=white, 5=blue
    for lname in layer_names:
        color = colors.get(lname, "7")
        content.extend(["0", "LAYER", "2", lname, "70", "0", "62", color, "6", "CONTINUOUS"])
    content.extend(["0", "ENDTAB", "0", "ENDSEC"])

    # Blocks
    content.extend(["0", "SECTION", "2", "BLOCKS", "0", "ENDSEC"])

    # Entities
    content.extend(["0", "SECTION", "2", "ENTITIES"])

    total_polys = 0
    for layer_name, polygons in layers_data.items():
        for poly in polygons:
            if poly.exterior:
                content.extend(polyline_dxf(list(poly.exterior.coords), layer_name))
                total_polys += 1
            for interior in poly.interiors:
                content.extend(polyline_dxf(list(interior.coords), layer_name))

    content.extend(["0", "ENDSEC", "0", "EOF", ""])

    with open(out_path, "w") as f:
        f.write("\n".join(content))

    print(f"  Wrote: {out_path} ({total_polys} polygons, layers: {layer_names})")


def get_element_centroids(elements: List[Tri6Element], nodes: Dict, scale: float,
                          material: Material) -> List[Tuple[float, float]]:
    """Get centroids of original triangle elements for a given material.

    This is more reliable than using merged polygon's representative_point
    because triangle centroids are always inside the triangle.
    """
    centroids = []
    for e in elements:
        if e.material == material:
            cx = sum(nodes[n][0] for n in e.corner_nodes) / 3 * scale
            cy = sum(nodes[n][1] for n in e.corner_nodes) / 3 * scale
            centroids.append((cx, cy))
    return centroids


def find_representative_points(merged_polygons: List[Polygon],
                                element_centroids: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """For each merged polygon, find ONE element centroid that lies inside it.

    This ensures the block label point is definitely from the correct material.
    """
    from shapely.geometry import Point

    points = []
    used_centroids = set()

    for poly in merged_polygons:
        if not poly.is_valid or poly.is_empty:
            continue

        found = False
        for i, (cx, cy) in enumerate(element_centroids):
            if i in used_centroids:
                continue
            if poly.contains(Point(cx, cy)):
                points.append((cx, cy))
                used_centroids.add(i)
                found = True
                break

        # Fallback to representative_point if no element centroid found
        if not found:
            pt = poly.representative_point()
            points.append((pt.x, pt.y))

    return points


def extract_boundary_corner_nodes(elements: List[Tri6Element], nodes: Dict[int, Tuple[float, float]],
                                   tol: float = 1e-9):
    """Extract CORNER nodes that lie on X-axis (y≈0).

    Only uses corner nodes (n1, n2, n3) from Tri6 elements, ignoring mid-edge nodes.
    Returns sorted list of (x, y) tuples, sorted by x coordinate.
    """
    # Collect all corner node IDs
    corner_node_ids = set()
    for e in elements:
        corner_node_ids.update(e.corner_nodes)  # n1, n2, n3 only

    # Find corner nodes on X-axis
    axis_nodes = []
    for nid in corner_node_ids:
        x, y = nodes[nid]
        if abs(y) < tol and x > tol:  # On X-axis, x > 0
            axis_nodes.append((x, y))

    # Sort by x (which is the radial distance on X-axis)
    axis_nodes.sort(key=lambda p: p[0])
    return axis_nodes


def create_boundary_segments(elements: List[Tri6Element], nodes: Dict[int, Tuple[float, float]], scale: float):
    """Create boundary segments from mesh CORNER nodes on X-axis.

    Mesh gốc từ 0-45°, các corner nodes trên trục X (y=0) sẽ là biên.
    Mirror qua y=x: các nodes (x, 0) -> (0, x) trên trục Y.

    Mỗi segment trên X-axis sẽ khớp với segment tương ứng trên Y-axis.
    """
    # Extract CORNER nodes on X-axis from original mesh (ignore mid-edge nodes)
    axis_nodes = extract_boundary_corner_nodes(elements, nodes)

    if len(axis_nodes) < 2:
        print("      WARNING: Not enough nodes on X-axis for boundary segments")
        return []

    n_segments = len(axis_nodes) - 1
    print(f"      Found {len(axis_nodes)} nodes on X-axis -> {n_segments} segments")

    boundary_pairs = []
    for i in range(n_segments):
        x1, _ = axis_nodes[i]
        x2, _ = axis_nodes[i + 1]

        # Scale to mm
        x1_mm = x1 * scale
        x2_mm = x2 * scale

        boundary_pairs.append({
            "name": f"boundary_side_{i+1}",
            "x_axis": {
                "x1": x1_mm, "y1": 0.0,
                "x2": x2_mm, "y2": 0.0
            },
            "y_axis": {
                "x1": 0.0, "y1": x1_mm,  # Mirror: (x, 0) -> (0, x)
                "x2": 0.0, "y2": x2_mm
            }
        })
        print(f"        Segment {i+1}: r = {x1_mm:.2f} → {x2_mm:.2f} mm")

    return boundary_pairs


def extract_boundary_nodes_on_axis(elements: List[Tri6Element], nodes: Dict[int, Tuple[float, float]],
                                    axis: str, tol: float = 1e-9) -> List[Tuple[float, float]]:
    """Extract CORNER nodes on specified axis.

    Args:
        axis: 'x' for X-axis (y≈0), 'y' for Y-axis (x≈0)

    Returns sorted list of (x, y) tuples.
    """
    corner_node_ids = set()
    for e in elements:
        corner_node_ids.update(e.corner_nodes)

    axis_nodes = []
    for nid in corner_node_ids:
        x, y = nodes[nid]
        if axis == 'x':
            if abs(y) < tol and x > tol:  # On X-axis, x > 0
                axis_nodes.append((x, y))
        elif axis == 'y':
            if abs(x) < tol and y > tol:  # On Y-axis, y > 0
                axis_nodes.append((x, y))

    # Sort by distance from origin
    if axis == 'x':
        axis_nodes.sort(key=lambda p: p[0])
    else:
        axis_nodes.sort(key=lambda p: p[1])

    return axis_nodes


def create_boundary_segments_symmetric(elements: List[Tri6Element], nodes: Dict[int, Tuple[float, float]], scale: float):
    """Create boundary segments from combined mesh (gốc + mirrored).

    Vì mesh đã được mirror, cả X-axis và Y-axis đều có corner nodes từ mesh.
    Điều này đảm bảo segments trên 2 trục có độ dài khớp hoàn hảo.
    """
    # Extract nodes on both axes
    x_axis_nodes = extract_boundary_nodes_on_axis(elements, nodes, 'x')
    y_axis_nodes = extract_boundary_nodes_on_axis(elements, nodes, 'y')

    if len(x_axis_nodes) < 2 or len(y_axis_nodes) < 2:
        print("      WARNING: Not enough nodes on axes for boundary segments")
        return []

    # Verify symmetry: số nodes trên 2 trục phải bằng nhau
    if len(x_axis_nodes) != len(y_axis_nodes):
        print(f"      WARNING: Asymmetric mesh! X-axis: {len(x_axis_nodes)}, Y-axis: {len(y_axis_nodes)}")
        # Dùng số nhỏ hơn
        n_nodes = min(len(x_axis_nodes), len(y_axis_nodes))
    else:
        n_nodes = len(x_axis_nodes)
        print(f"      Found {n_nodes} nodes on each axis -> {n_nodes - 1} segment pairs")

    boundary_pairs = []
    for i in range(n_nodes - 1):
        # X-axis segment
        x1_x, _ = x_axis_nodes[i]
        x2_x, _ = x_axis_nodes[i + 1]

        # Y-axis segment
        _, y1_y = y_axis_nodes[i]
        _, y2_y = y_axis_nodes[i + 1]

        # Scale to mm
        x1_mm = x1_x * scale
        x2_mm = x2_x * scale
        y1_mm = y1_y * scale
        y2_mm = y2_y * scale

        # Verify lengths match (should be identical due to mesh mirroring)
        len_x = abs(x2_mm - x1_mm)
        len_y = abs(y2_mm - y1_mm)
        if abs(len_x - len_y) > 0.01:  # 0.01mm tolerance
            print(f"        WARNING: Segment {i+1} length mismatch: X={len_x:.3f}, Y={len_y:.3f}")

        boundary_pairs.append({
            "name": f"boundary_side_{i+1}",
            "x_axis": {
                "x1": x1_mm, "y1": 0.0,
                "x2": x2_mm, "y2": 0.0
            },
            "y_axis": {
                "x1": 0.0, "y1": y1_mm,
                "x2": 0.0, "y2": y2_mm
            }
        })

    return boundary_pairs


def save_centroids_json(out_path: str, air_centroids: List, iron_centroids: List,
                        boundary_pairs: List = None):
    """Save centroids and boundary info to JSON for FEMM."""
    data = {
        "air": [{"x": x, "y": y} for x, y in air_centroids],
        "iron": [{"x": x, "y": y} for x, y in iron_centroids],
        "boundaries": boundary_pairs or [],
        "scale": DXF_SCALE,
        "note": "Coordinates in mm (scaled)."
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    n_bounds = len(boundary_pairs) if boundary_pairs else 0
    print(f"  Wrote: {out_path} (AIR: {len(air_centroids)}, IRON: {len(iron_centroids)}, Boundaries: {n_bounds})")


# ============================================================================
# Main / API
# ============================================================================
def generate_geometry(weights: np.ndarray, tra_path: str, output_dir: str = ".",
                      verbose: bool = True, output_prefix: str = None) -> Tuple[str, str]:
    """
    Generate DXF và JSON từ weights của NGNet.

    Args:
        weights: numpy array chứa weights cho NGNet (kích thước = n_radial * n_angular)
        tra_path: đường dẫn file TRA (mesh)
        output_dir: thư mục output
        verbose: in thông tin chi tiết
        output_prefix: prefix cho tên file output (để chạy song song)
                       Nếu None, dùng "combined_regions" mặc định

    Returns:
        Tuple[dxf_path, json_path]: đường dẫn file DXF và JSON đã tạo
    """
    if verbose:
        print("=" * 60)
        print("Generate geometry from weights")
        print("=" * 60)

    # Load mesh
    if verbose:
        print("\n1. Loading mesh...")
    nodes = parse_tra_nodes(tra_path)
    elements = parse_tra_elements(tra_path)
    if verbose:
        print(f"   {len(nodes)} nodes, {len(elements)} elements")

    radii = [np.sqrt(x**2 + y**2) for x, y in nodes.values()]
    r_min, r_max = min(radii), max(radii)

    # NGnet config                                                                          #gaucess center
    n_radial = 6
    n_angular = 5
    config = NGnetConfig(
        r_min=r_min + (r_max - r_min) * 0,                                                #mechanical constraint
        r_max=r_max - (r_max - r_min) * 0,
        theta_min=0,
        theta_max=np.pi / 4,
        n_radial=n_radial,
        n_angular=n_angular,
    )
    ngnet = NGnet(config)

    # Set weights từ input
    if len(weights) != len(ngnet.weights):
        raise ValueError(f"weights size mismatch: expected {len(ngnet.weights)}, got {len(weights)}")
    ngnet.weights = np.array(weights)

    # Assign materials và tạo geometry
    return _build_geometry(elements, nodes, ngnet, output_dir, verbose, output_prefix)


def get_ngnet_size(tra_path: str, n_radial: int = 6, n_angular: int = 5) -> int:
    """Trả về số lượng weights cần cho NGNet."""
    return n_radial * n_angular


def _build_geometry(elements, nodes, ngnet, output_dir, verbose, output_prefix=None) -> Tuple[str, str]:
    """Internal function để build geometry từ ngnet đã có weights.

    Quy trình đúng: Mirror MESH trước, rồi mới tạo polygon.
    Điều này đảm bảo đường biên X-axis và Y-axis hoàn toàn đối xứng.

    Args:
        output_prefix: prefix cho tên file (để chạy song song). Mặc định "combined_regions"
    """
    if output_prefix is None:
        output_prefix = "combined_regions"

    # Assign materials cho mesh gốc (0-45°)
    if verbose:
        print("\n2. Assigning materials to original mesh (0-45°)...")
    assign_materials(elements, nodes, ngnet)
    n_iron = sum(1 for e in elements if e.material == Material.IRON)
    n_air = sum(1 for e in elements if e.material == Material.AIR)
    if verbose:
        print(f"   Original mesh - Iron: {n_iron}, Air: {n_air}")

    # Mirror mesh (0-45° → 45-90°) - material được copy từ original
    if verbose:
        print("\n3. Mirroring mesh to create symmetric 0-90° mesh...")
    mirrored_nodes, mirrored_elements = mirror_mesh(nodes, elements)
    if verbose:
        print(f"   Mirrored mesh: {len(mirrored_nodes)} nodes, {len(mirrored_elements)} elements")

    # Combine original + mirrored mesh
    combined_nodes = {**nodes, **mirrored_nodes}
    combined_elements = elements + mirrored_elements
    if verbose:
        print(f"   Combined mesh: {len(combined_nodes)} nodes, {len(combined_elements)} elements")

    # Convert to polygons (từ combined mesh 0-90°)
    if verbose:
        print("\n4. Building polygons from combined mesh...")
    iron_polys = []
    air_polys = []
    all_polys = []

    for e in combined_elements:
        p = element_to_polygon(e, combined_nodes, DXF_SCALE)
        if p and p.is_valid and not p.is_empty:
            all_polys.append(p)
            if e.material == Material.IRON:
                iron_polys.append(p)
            else:
                air_polys.append(p)

    # Merge polygons
    if verbose:
        print("\n5. Merging polygons...")

    # Total region (all elements)
    if verbose:
        print("   Merging total...")
    total_final = unary_union(all_polys)
    total_list = polygon_to_list(total_final)
    if verbose:
        print(f"   -> Total: {len(total_list)} region(s)")

    # AIR regions - cluster by adjacency to keep separate regions
    if verbose:
        print("   Clustering AIR regions by adjacency...")
    air_clusters = cluster_adjacent_elements(combined_elements, Material.AIR)
    if verbose:
        print(f"   -> Found {len(air_clusters)} AIR cluster(s)")

    air_list = []
    for cluster in air_clusters:
        # Merge triangles within this cluster
        cluster_polys = [element_to_polygon(e, combined_nodes, DXF_SCALE) for e in cluster]
        cluster_polys = [p for p in cluster_polys if p and p.is_valid and not p.is_empty]
        if cluster_polys:
            cluster_union = unary_union(cluster_polys)
            air_list.extend(polygon_to_list(cluster_union))

    if verbose:
        print(f"   -> AIR: {len(air_list)} region(s)")

    # IRON regions - cluster by adjacency (giống AIR, KHÔNG dùng difference)
    # Điều này đảm bảo IRON boundaries khớp chính xác với mesh nodes
    if verbose:
        print("   Clustering IRON regions by adjacency...")
    iron_clusters = cluster_adjacent_elements(combined_elements, Material.IRON)
    if verbose:
        print(f"   -> Found {len(iron_clusters)} IRON cluster(s)")

    iron_list = []
    for cluster in iron_clusters:
        cluster_polys = [element_to_polygon(e, combined_nodes, DXF_SCALE) for e in cluster]
        cluster_polys = [p for p in cluster_polys if p and p.is_valid and not p.is_empty]
        if cluster_polys:
            cluster_union = unary_union(cluster_polys)
            iron_list.extend(polygon_to_list(cluster_union))

    if verbose:
        print(f"   -> IRON: {len(iron_list)} region(s)")

    # KHÔNG simplify - giữ nguyên vertices từ mesh để đảm bảo đối xứng hoàn hảo
    # (Nếu cần simplify, phải làm theo cách đối xứng qua đường 45°)

    # Export combined DXF
    if verbose:
        print("\n6. Exporting DXF and centroids...")
    dxf_path = os.path.join(output_dir, f"{output_prefix}.dxf")
    json_path = os.path.join(output_dir, f"{output_prefix}.json")

    write_combined_dxf(dxf_path, {"AIR": air_list, "IRON": iron_list})

    # Compute centroids using combined element centroids
    if verbose:
        print("   Computing block label positions from combined mesh elements...")
    air_elem_centroids = get_element_centroids(combined_elements, combined_nodes, DXF_SCALE, Material.AIR)
    iron_elem_centroids = get_element_centroids(combined_elements, combined_nodes, DXF_SCALE, Material.IRON)

    # Find one representative point per merged polygon
    air_centroids = find_representative_points(air_list, air_elem_centroids)
    iron_centroids = find_representative_points(iron_list, iron_elem_centroids)

    # Create boundary segments from COMBINED mesh (đảm bảo symmetric)
    if verbose:
        print("   Creating boundary segments from combined mesh corner nodes...")
    boundary_pairs = create_boundary_segments_symmetric(combined_elements, combined_nodes, DXF_SCALE)
    if verbose:
        print(f"   -> Created {len(boundary_pairs)} boundary segment pair(s)")

    save_centroids_json(json_path, air_centroids, iron_centroids, boundary_pairs)

    # Visualization (only when verbose)
    if verbose:
        print("\n7. Creating preview...")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=150)

            # AIR
            ax1 = axes[0]
            ax1.set_aspect("equal")
            ax1.set_title(f"AIR regions ({len(air_list)})")
            for p in air_list:
                if p.exterior:
                    ax1.fill(*zip(*p.exterior.coords), facecolor='lightyellow',
                            edgecolor='orange', linewidth=1.5)
                    for interior in p.interiors:
                        ax1.fill(*zip(*interior.coords), facecolor='white',
                                edgecolor='orange', linewidth=1)
            ax1.autoscale()
            ax1.set_xlabel("x (mm)")
            ax1.set_ylabel("y (mm)")

            # IRON
            ax2 = axes[1]
            ax2.set_aspect("equal")
            ax2.set_title(f"IRON regions ({len(iron_list)})")
            for p in iron_list:
                if p.exterior:
                    ax2.fill(*zip(*p.exterior.coords), facecolor='steelblue',
                            edgecolor='darkblue', linewidth=1.5)
                    for interior in p.interiors:
                        ax2.fill(*zip(*interior.coords), facecolor='white',
                                edgecolor='darkblue', linewidth=1)
            ax2.autoscale()
            ax2.set_xlabel("x (mm)")
            ax2.set_ylabel("y (mm)")

            plt.tight_layout()
            png_path = os.path.join(output_dir, f"{output_prefix}_preview.png")
            fig.savefig(png_path)
            plt.close()
            print(f"  Wrote: {png_path}")
        except Exception as e:
            print(f"  Preview failed: {e}")

        print("\n" + "=" * 60)
        print("DONE!")
        print(f"\nFiles đã tạo:")
        print(f"  1. {dxf_path}  <- DXF với 2 layer: AIR, IRON")
        print(f"  2. {json_path} <- Tọa độ centroid để đặt block labels")
        print("=" * 60)

    return dxf_path, json_path


def main(tra_path: str, output_dir: str = "."):
    """Main function để chạy standalone với random weights."""
    print("=" * 60)
    print("Export AIR và IRON riêng biệt (standalone mode)")
    print("=" * 60)

    # Load mesh để lấy config
    nodes = parse_tra_nodes(tra_path)
    elements = parse_tra_elements(tra_path)

    radii = [np.sqrt(x**2 + y**2) for x, y in nodes.values()]
    r_min, r_max = min(radii), max(radii)

    # NGnet config
    n_radial = 6
    n_angular = 5
    config = NGnetConfig(
        r_min=r_min + (r_max - r_min) * 0.1,
        r_max=r_max - (r_max - r_min) * 0.1,
        theta_min=0,
        theta_max=np.pi / 4,
        n_radial=n_radial,
        n_angular=n_angular,
    )
    ngnet = NGnet(config)
    ngnet.set_random_weights(seed=3)

    # Build geometry
    _build_geometry(elements, nodes, ngnet, output_dir, verbose=True)
    print("\nBước tiếp theo: Chạy plot_fem.py để import vào FEMM")


if __name__ == "__main__":
    import sys
    tra = sys.argv[1] if len(sys.argv) > 1 else "rotor_2.TRA"
    if os.path.exists(tra):
        main(tra)
    else:
        print(f"File not found: {tra}")