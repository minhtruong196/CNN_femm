# plot_fem.py - Import DXF vào FEMM và tự động đặt block labels
import json
import math
from pathlib import Path
from typing import List, Tuple

import femm

# Global flag để track FEMM đã mở chưa (mỗi process có flag riêng khi dùng multiprocessing)
_femm_opened = False


def reset_femm_state():
    """Reset FEMM state cho worker mới (dùng trong multiprocessing)."""
    global _femm_opened
    _femm_opened = False


def calculate_phase_currents(Im=11, ini=-15, adv=35):
    angle_a = ini + 90 + adv
    angle_b = ini + 90 + adv - 120
    angle_c = ini + 90 + adv + 120

    i_a = Im * math.cos(math.radians(angle_a))
    i_b = Im * math.cos(math.radians(angle_b))
    i_c = Im * math.cos(math.radians(angle_c))

    return i_a, i_b, i_c


def ensure_femm_open():
    """Mở FEMM nếu chưa mở."""
    global _femm_opened
    if not _femm_opened:
        femm.openfemm(1)  # 1 = minimized
        _femm_opened = True


def close_femm():
    """Đóng FEMM."""
    global _femm_opened
    if _femm_opened:
        femm.closefemm()
        _femm_opened = False


def setup_model(
    base_fem: str,
    rotor_dxf: str,
    centroids_json: str,
    output_fem: str = None,
    verbose: bool = True,
) -> Tuple[str, dict]:
    """
    Setup model: import DXF, đặt block labels, tạo boundaries.
    KHÔNG chạy analyze - chỉ setup geometry.

    Returns:
        Tuple[output_fem_path, centroids_data]
    """
    base_path = Path(base_fem).absolute()
    dxf_path = Path(rotor_dxf).absolute()
    json_path = Path(centroids_json).absolute()

    if output_fem is None:
        output_fem = base_path.parent / (base_path.stem + "_with_rotor.FEM")
    else:
        output_fem = Path(output_fem).absolute()

    # Load centroids
    if verbose:
        print("Loading centroids...")
    with open(json_path, "r") as f:
        centroids_data = json.load(f)

    air_points = centroids_data.get("air", [])
    iron_points = centroids_data.get("iron", [])
    boundary_pairs = centroids_data.get("boundaries", [])
    if verbose:
        print(f"  AIR: {len(air_points)} points, IRON: {len(iron_points)} points")
        print(f"  Boundaries: {len(boundary_pairs)} pairs")

    # Mở FEMM
    ensure_femm_open()

    # Đóng document cũ nếu còn mở (tránh chồng lấn)
    try:
        femm.mi_close()
    except:
        pass

    # Mở file gốc
    femm.opendocument(base_path.as_posix())

    # Import DXF
    if verbose:
        print(f"Importing DXF: {dxf_path}")
    femm.mi_readdxf(dxf_path.as_posix())

    # Đặt block labels cho AIR - gán vào group 2 để tính torque
    if verbose:
        print("Placing AIR block labels (group 2)...")
    for pt in air_points:
        x, y = pt["x"], pt["y"]
        femm.mi_addblocklabel(x, y)
        femm.mi_selectlabel(x, y)
        femm.mi_setblockprop("Air", 0, 4, "<None>", 0, 2, 0)
        femm.mi_clearselected()

    # Đặt block labels cho IRON
    if verbose:
        print("Placing IRON block labels (M350_50A, group 2)...")
    for pt in iron_points:
        x, y = pt["x"], pt["y"]
        femm.mi_addblocklabel(x, y)
        femm.mi_selectlabel(x, y)
        femm.mi_setblockprop("M350_50A", 0, 4, "<None>", 0, 2, 0)
        femm.mi_clearselected()

    # Tạo Anti-periodic boundary conditions
    if boundary_pairs and verbose:
        print(f"Creating anti-periodic boundaries ({len(boundary_pairs)} pairs)...")
    for pair in boundary_pairs:
        name = pair["name"]
        femm.mi_addboundprop(name, 0, 0, 0, 0, 0, 0, 0, 0, 5)

        edge_x = pair["x_axis"]
        mid_x = (edge_x["x1"] + edge_x["x2"]) / 2
        mid_y = (edge_x["y1"] + edge_x["y2"]) / 2
        try:
            femm.mi_selectsegment(mid_x, mid_y)
            femm.mi_setsegmentprop(name, 0, 1, 0, 1)
            femm.mi_clearselected()
        except:
            pass

        edge_y = pair["y_axis"]
        mid_x = (edge_y["x1"] + edge_y["x2"]) / 2
        mid_y = (edge_y["y1"] + edge_y["y2"]) / 2
        try:
            femm.mi_selectsegment(mid_x, mid_y)
            femm.mi_setsegmentprop(name, 0, 1, 0, 1)
            femm.mi_clearselected()
        except:
            pass

    # Lưu file mới
    femm.mi_saveas(str(output_fem))

    if verbose:
        print(f"Model saved: {output_fem}")

    # Đóng document để tránh chồng lấn khi gọi lại
    femm.mi_close()

    return str(output_fem), centroids_data


def evaluate_torque(
    fem_file: str,
    adv: float,
    Im: float = 11,
    ini: float = -15,
    verbose: bool = True,
) -> float:
    """
    Chạy simulation với góc adv cho trước và trả về torque.

    Args:
        fem_file: File FEM đã setup (có geometry)
        adv: Góc advance (độ)
        Im: Biên độ dòng điện (A)
        ini: Góc ban đầu (độ)
        verbose: In thông tin

    Returns:
        Torque (N.m)
    """
    ensure_femm_open()

    # Đóng document cũ nếu còn mở (tránh chồng lấn khi có error trước đó)
    try:
        femm.mi_close()
    except:
        pass

    # Mở file
    femm.opendocument(str(Path(fem_file).absolute()))

    # Tính dòng điện 3 pha
    i_a, i_b, i_c = calculate_phase_currents(Im, ini, adv)

    if verbose:
        print(f"  adv={adv}°: i_a={i_a:.3f}, i_b={i_b:.3f}, i_c={i_c:.3f}", end="")

    # Gán dòng điện
    femm.mi_modifycircprop("PhaseA", 1, i_a)
    femm.mi_modifycircprop("PhaseB", 1, i_b)
    femm.mi_modifycircprop("PhaseC", 1, i_c)

    # Analyze
    femm.mi_analyze(1)  # 1 = minimize window
    femm.mi_loadsolution()

    # Tính torque
    femm.mo_groupselectblock(2)
    torque = femm.mo_blockintegral(22)
    femm.mo_clearblock()

    if verbose:
        print(f" -> torque={torque:.4f} N.m")

    # Đóng solution và document
    femm.mo_close()
    femm.mi_close()

    return torque


def sweep_adv_for_max_torque(
    fem_file: str,
    adv_list: List[float] = None,
    Im: float = 11,
    ini: float = -15,
    verbose: bool = True,
) -> Tuple[float, float, List[Tuple[float, float]]]:
    """
    Quét nhiều góc adv và tìm torque max.

    Args:
        fem_file: File FEM đã setup
        adv_list: Danh sách góc adv (mặc định [35, 40, 45, 50, 55])
        Im: Biên độ dòng điện
        ini: Góc ban đầu
        verbose: In thông tin

    Returns:
        Tuple[best_adv, max_torque, all_results]: góc tốt nhất, torque max, và tất cả kết quả
    """
    if adv_list is None:
        adv_list = [35, 40, 45, 50, 55]

    if verbose:
        print(f"\nSweeping adv = {adv_list}...")

    results = []
    for adv in adv_list:
        torque = evaluate_torque(fem_file, adv, Im, ini, verbose=verbose)
        results.append((adv, torque))

    # Tìm max
    best_adv, max_torque = max(results, key=lambda x: x[1])

    if verbose:
        print(f"\nBest: adv={best_adv}° -> torque={max_torque:.4f} N.m")

    return best_adv, max_torque, results


def create_model_with_rotor(
    base_fem="basic.FEM",
    rotor_dxf="combined_regions.dxf",
    centroids_json="centroids.json",
    output_fem=None,
):
    """
    Mở file FEM gốc, import rotor từ DXF, đặt block labels tự động.
    File gốc không bị thay đổi. (Legacy function - giữ để tương thích)

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

    # Tính và set dòng điện 3 pha theo phương trình
    Im = 11      # Biên độ dòng điện (A)
    ini = -15    # Góc ban đầu (độ)
    adv = 35     # Control angle (độ)

    i_a, i_b, i_c = calculate_phase_currents(Im, ini, adv)
    print(f"\n=== DONG DIEN 3 PHA ===")
    print(f"  Im = {Im} A, ini = {ini}°, adv = {adv}°")
    print(f"  i_a = {i_a:.4f} A")
    print(f"  i_b = {i_b:.4f} A")
    print(f"  i_c = {i_c:.4f} A")

    # Gán dòng điện vào các circuit (propnum=1 = total current)
    femm.mi_modifycircprop("PhaseA", 1, i_a)
    femm.mi_modifycircprop("PhaseB", 1, i_b)
    femm.mi_modifycircprop("PhaseC", 1, i_c)
    print("  -> Da gan dong dien vao circuit PhaseA, PhaseB, PhaseC")

    # Vật liệu Air và M350_50A đã có sẵn trong file FEM gốc
    # Không cần gọi mi_getmaterial

    # Import DXF
    print(f"Importing DXF: {dxf_path}")
    femm.mi_readdxf(dxf_path.as_posix())

    # Đặt block labels cho AIR - gán vào group 2 để tính torque
    # Tọa độ trong JSON đã là mm (cùng đơn vị với DXF và FEMM)
    print("Placing AIR block labels (group 2)...")
    for pt in air_points:
        x, y = pt["x"], pt["y"]
        femm.mi_addblocklabel(x, y)
        femm.mi_selectlabel(x, y)
        # mi_setblockprop(blockname, automesh, meshsize, incircuit, magdir, group, turns)
        femm.mi_setblockprop("Air", 0, 6, "<None>", 0, 2, 0)
        femm.mi_clearselected()

    # Đặt block labels cho IRON (M350_50A, mesh size 4) - gán vào group 2
    print("Placing IRON block labels (M350_50A, group 2)...")
    for pt in iron_points:
        x, y = pt["x"], pt["y"]
        femm.mi_addblocklabel(x, y)
        femm.mi_selectlabel(x, y)
        # mi_setblockprop(blockname, automesh, meshsize, incircuit, magdir, group, turns)
        femm.mi_setblockprop("M350_50A", 0, 4, "<None>", 0, 2, 0)
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

    # Chạy analyze và tính torque cho group 2
    print("\n=== TINH TORQUE ===")
    print("Running analysis...")
    femm.mi_analyze()
    femm.mi_loadsolution()

    # Chọn tất cả block trong group 2 và tính torque
    femm.mo_groupselectblock(2)

    # Tính torque bằng weighted stress tensor (type 22)
    torque = femm.mo_blockintegral(22)
    print(f"\nTorque (group 2) = {torque} N.m")

    # Clear selection
    femm.mo_clearblock()

    return output_fem, torque


def evaluate_individual_worker(args):
    """
    Worker function để đánh giá một individual trong subprocess riêng.
    Dùng cho multiprocessing - mỗi worker có FEMM instance riêng.

    Args:
        args: tuple (worker_id, weights, tra_path, base_fem, output_dir, adv_list, Im, ini)

    Returns:
        tuple (worker_id, fitness, best_adv) hoặc (worker_id, 0.0, 0.0) nếu lỗi
    """
    worker_id, weights, tra_path, base_fem, output_dir, adv_list, Im, ini = args

    # Import ở đây để tránh circular import
    from dxf_part import generate_geometry

    try:
        # Reset FEMM state cho worker mới
        reset_femm_state()

        # Tạo tên file riêng cho worker này
        output_prefix = f"worker_{worker_id}"

        # 1. Tạo geometry từ weights
        dxf_path, json_path = generate_geometry(
            weights=weights,
            tra_path=tra_path,
            output_dir=output_dir,
            verbose=False,
            output_prefix=output_prefix
        )

        # 2. Setup FEMM model với output file riêng
        output_fem = str(Path(output_dir) / f"{output_prefix}.FEM")
        fem_file, _ = setup_model(
            base_fem=base_fem,
            rotor_dxf=dxf_path,
            centroids_json=json_path,
            output_fem=output_fem,
            verbose=False
        )

        # 3. Quét adv để tìm max torque
        best_adv, max_torque, _ = sweep_adv_for_max_torque(
            fem_file=fem_file,
            adv_list=adv_list,
            Im=Im,
            ini=ini,
            verbose=False
        )

        # Đóng FEMM sau khi xong
        close_femm()

        return (worker_id, max_torque, best_adv)

    except Exception as e:
        print(f"Worker {worker_id} ERROR: {e}")
        try:
            close_femm()
        except:
            pass
        return (worker_id, 0.0, 0.0)


if __name__ == "__main__":
    output_file, torque = create_model_with_rotor(
        base_fem="basic.FEM",
        rotor_dxf="combined_regions.dxf",
        centroids_json="centroids.json",
    )
    print(f"\n=== KET QUA ===")
    print(f"Output file: {output_file}")
    print(f"Torque: {torque} N.m")
    input("\nNhan Enter de dong FEMM...")
