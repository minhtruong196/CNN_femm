# plot_fem.py - Import DXF vào FEMM và tự động đặt block labels
import json
from pathlib import Path

import femm


def create_model_with_rotor(
    base_fem="basic.FEM",
    rotor_dxf="combined_regions.dxf",
    centroids_json="centroids.json",
    output_fem=None,
):
    """
    Mở file FEM gốc, import rotor từ DXF, đặt block labels tự động.
    File gốc không bị thay đổi.

    Args:
        base_fem: File FEM gốc
        rotor_dxf: File DXF chứa geometry (2 layer: AIR, IRON)
        centroids_json: File JSON chứa tọa độ centroid
        output_fem: File FEM output (mặc định: {base}_with_rotor.FEM)
    """
    base_path = Path(base_fem).absolute()
    dxf_path = Path(rotor_dxf).absolute()
    json_path = Path(centroids_json).absolute()

    if output_fem is None:
        output_fem = base_path.parent / (base_path.stem + "_with_rotor.FEM")
    else:
        output_fem = Path(output_fem).absolute()

    # Load centroids
    print("Loading centroids...")
    with open(json_path, "r") as f:
        centroids_data = json.load(f)

    air_points = centroids_data.get("air", [])
    iron_points = centroids_data.get("iron", [])
    boundary_pairs = centroids_data.get("boundaries", [])
    print(f"  AIR: {len(air_points)} points, IRON: {len(iron_points)} points")
    print(f"  Boundaries: {len(boundary_pairs)} pairs")

    # Mở FEMM
    print("Opening FEMM...")
    femm.openfemm()

    # Mở file gốc
    femm.opendocument(base_path.as_posix())

    # Vật liệu Air và M350_50A đã có sẵn trong file FEM gốc
    # Không cần gọi mi_getmaterial

    # Import DXF
    print(f"Importing DXF: {dxf_path}")
    femm.mi_readdxf(dxf_path.as_posix())

    # Đặt block labels cho AIR
    # Tọa độ trong JSON đã là mm (cùng đơn vị với DXF và FEMM)
    print("Placing AIR block labels...")
    for pt in air_points:
        x, y = pt["x"], pt["y"]
        femm.mi_addblocklabel(x, y)
        femm.mi_selectlabel(x, y)
        femm.mi_setblockprop("Air", 0, 6, "<None>", 0, 0, 0)
        femm.mi_clearselected()

    # Đặt block labels cho IRON (M350_50A, mesh size 4)
    print("Placing IRON block labels (M350_50A)...")
    for pt in iron_points:
        x, y = pt["x"], pt["y"]
        femm.mi_addblocklabel(x, y)
        femm.mi_selectlabel(x, y)
        # mi_setblockprop(blockname, automesh, meshsize, incircuit, magdir, group, turns)
        femm.mi_setblockprop("M350_50A", 0, 4, "<None>", 0, 0, 0)
        femm.mi_clearselected()

    # Tạo Anti-periodic boundary conditions
    # Mỗi cặp segment (trên X-axis và Y-axis) có boundary property riêng
    if boundary_pairs:
        print(f"Creating anti-periodic boundaries ({len(boundary_pairs)} pairs)...")
        for pair in boundary_pairs:
            name = pair["name"]
            # Tạo boundary property (BdryFormat=5 = Anti-periodic)
            femm.mi_addboundprop(name, 0, 0, 0, 0, 0, 0, 0, 0, 5)

            # Select segment trên trục X - dùng midpoint của segment
            edge_x = pair["x_axis"]
            mid_x = (edge_x["x1"] + edge_x["x2"]) / 2
            mid_y = (edge_x["y1"] + edge_x["y2"]) / 2
            try:
                femm.mi_selectsegment(mid_x, mid_y)
                femm.mi_setsegmentprop(name, 0, 1, 0, 1)
                femm.mi_clearselected()
                ok_x = True
            except:
                ok_x = False

            # Select segment trên trục Y - dùng midpoint của segment
            edge_y = pair["y_axis"]
            mid_x = (edge_y["x1"] + edge_y["x2"]) / 2
            mid_y = (edge_y["y1"] + edge_y["y2"]) / 2
            try:
                femm.mi_selectsegment(mid_x, mid_y)
                femm.mi_setsegmentprop(name, 0, 1, 0, 1)
                femm.mi_clearselected()
                ok_y = True
            except:
                ok_y = False

            status_x = "OK" if ok_x else "FAIL"
            status_y = "OK" if ok_y else "FAIL"
            print(f"    {name}: X-axis={status_x}, Y-axis={status_y}")

    # Zoom fit và refresh
    femm.mi_zoomnatural()
    femm.mi_refreshview()

    # Lưu file mới
    femm.mi_saveas(output_fem.as_posix())

    print(f"\nDa tao file moi: {output_fem}")
    print(f"  - AIR blocks: {len(air_points)}")
    print(f"  - IRON blocks: {len(iron_points)} (M350_50A, mesh=4)")
    print(f"  - Anti-periodic boundaries: {len(boundary_pairs)}")
    return output_fem


if __name__ == "__main__":
    create_model_with_rotor(
        base_fem="basic.FEM",
        rotor_dxf="combined_regions.dxf",
        centroids_json="centroids.json",
    )
    input("Nhan Enter de dong FEMM...")
