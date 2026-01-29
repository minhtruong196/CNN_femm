import femm
from pathlib import Path

# Simple DXF import using FEMM's built-in importer
femm.openfemm()
femm.newdocument(0)  # magnetics
femm.mi_probdef(0, "millimeters", "planar", 1e-8, 0, 30)

# Prefer the LINE-based DXF and avoid backslash escapes by using POSIX paths.
rotor_dxf = Path(__file__).with_name("rotor.dxf")
stator_Coil_dxf = Path(__file__).with_name("sta_coil.dxf")
femm.mi_readdxf(rotor_dxf.as_posix())
femm.mi_readdxf(stator_Coil_dxf.as_posix())
femm.mi_zoomnatural()
femm.mi_refreshview()

print("Da import DXF. Hay xem cua so FEMM.")
input("Nhan Enter de dong FEMM...")
femm.closefemm()
