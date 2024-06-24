
from dataclasses import dataclass

import gdsfactory as gf
from gdsfactory.technology import (
    LayerLevel,
    LayerMap,
    LayerStack,
    LayerView,
    LayerViews,
)
from gdsfactory.typings import Layer

gf.config.rich_output()
nm = 1e-3

class LayerMapElectronic(LayerMap):

    # Common (1)
    POLY: Layer = (1, 0)
    N_WELL: Layer = (2, 0)
    P_WELL: Layer = (3, 0)

    # BJT-specific (10)
    BJT_COLLECTOR: Layer = (10,0)
    BJT_COLLECTOR_DEEP: Layer = (11,0)
    BJT_EMITTER: Layer = (12,0)
    BJT_BASE: Layer = (13,0)
    BJT_ISOLATION: Layer = (14,0)

    # JFET-specific (20)


    # MOSFET-specific (30)

    # Caps (40)
    MIM_NITRIDE: Layer = (40, 0)

    # BEOL (50-70)
    VIAC: Layer = (60, 0)
    M1: Layer = (51, 0)
    VIA1: Layer = (61, 0)
    M2: Layer = (52, 0)
    VIA2: Layer = (62, 0)
    M3: Layer = (53, 0)
    VIA3: Layer = (63, 0)
    MTOP: Layer = (50, 0)
    PAD: Layer = (70, 0)

    # Misc
    LABEL_INSTANCE: Layer = (66, 0)
    DEVREC: Layer = (68, 0)
    PORT: Layer = (1, 10)
    PORTE: Layer = (1, 11)
    TE: Layer = (203, 0)
    TM: Layer = (204, 0)
    TEXT: Layer = (66, 0)
    WAFER: Layer = (99999, 0)

    class Config:
        frozen = True
        extra = "forbid"


LAYER = LayerMapElectronic()

# This is something you usually define in KLayout
class ElectronicLayerViews(LayerViews):
    POLY: LayerView = LayerView(color="#FF0000")  # Red
    N_WELL: LayerView = LayerView(color="#00FF00", pattern="dotted")  # Green
    P_WELL: LayerView = LayerView(color="#0000FF", pattern="dashed")  # Blue

    class BEOLGroup(LayerView):
        M1: LayerView = LayerView(color="#FFFF00")  # Yellow
        VIA1: LayerView = LayerView(color="#FF00FF")  # Magenta
        M2: LayerView = LayerView(color="#00FFFF")  # Cyan
        VIA2: LayerView = LayerView(color="#FFA500")  # Orange
        M3: LayerView = LayerView(color="#A52A2A")  # Brown
        VIA3: LayerView = LayerView(color="#800080")  # Purple
        MTOP: LayerView = LayerView(color="#808080")  # Grey
        PAD: LayerView = LayerView(color="#808080", pattern="solid")  # Black

    BEOL: LayerView = BEOLGroup()

    class SimulationGroup(LayerView):
        TE: LayerView = LayerView(color="green")
        PORT: LayerView = LayerView(color="green", alpha=0)

    Simulation: LayerView = SimulationGroup()


LAYER_VIEWS = ElectronicLayerViews(layers=LAYER)


@dataclass
class LayerThicknessDefaults:
    thickness_mim_nitride: int = 200 * nm
    thickness_viac: int = 1000 * nm
    thickness_m1: int = 2000 * nm
    thickness_via1: int = 1000 * nm
    thickness_m2: int = 2000 * nm
    thickness_via2: int = 1000 * nm
    thickness_m3: int = 2000 * nm
    thickness_via3: int = 1000 * nm
    thickness_mtop: int = 4000 * nm
    substrate_thickness: int = 6000 * nm

def get_layer_stack_electronic(thickness_mim_nitride = LayerThicknessDefaults.thickness_mim_nitride,
                               thickness_viac = LayerThicknessDefaults.thickness_viac,
                               thickness_m1 = LayerThicknessDefaults.thickness_m1,
                               thickness_via1 = LayerThicknessDefaults.thickness_via1,
                               thickness_m2 = LayerThicknessDefaults.thickness_m2,
                               thickness_via2 = LayerThicknessDefaults.thickness_via2,
                               thickness_m3 = LayerThicknessDefaults.thickness_m3,
                               thickness_via3 = LayerThicknessDefaults.thickness_via3,
                               thickness_mtop = LayerThicknessDefaults.thickness_mtop,
                               substrate_thickness = LayerThicknessDefaults.substrate_thickness
                               ):

    return LayerStack(
        layers=dict(
            mim_nitride=LayerLevel(
                layer=LAYER.MIM_NITRIDE,
                thickness=thickness_mim_nitride,
                zmin=0,
                material="silicon_nitride",
                mesh_order=1
            ),
            viac=LayerLevel(
                layer=LAYER.VIAC,
                thickness=thickness_viac,
                zmin=0,
                material="copper",
                mesh_order=4
            ),
            m1=LayerLevel(
                layer=LAYER.M1,
                thickness=thickness_m1,
                zmin=thickness_viac,
                material="copper",
                mesh_order=2
            ),
            via1=LayerLevel(
                layer=LAYER.VIA1,
                thickness=thickness_via1,
                zmin=(thickness_viac + thickness_m1),
                material="copper",
                mesh_order=5
            ),
            m2=LayerLevel(
                layer=LAYER.M2,
                thickness=thickness_m2,
                zmin=(thickness_viac + thickness_m1 + thickness_via1),
                material="copper",
                mesh_order=3
            ),
            via2=LayerLevel(
                layer=LAYER.VIA2,
                thickness=thickness_via2,
                zmin=(thickness_viac + thickness_m1 + thickness_via1 + thickness_m2),
                material="copper",
                mesh_order=6
            ),
            m3=LayerLevel(
                layer=LAYER.M3,
                thickness=thickness_m3,
                zmin=(thickness_viac + thickness_m1 + thickness_via1 + thickness_m2 + thickness_via2),
                material="copper",
                mesh_order=7
            ),
            via3=LayerLevel(
                layer=LAYER.VIA3,
                thickness=thickness_via3,
                zmin=(thickness_viac + thickness_m1 + thickness_via1 + thickness_m2 + thickness_via2 + thickness_m3),
                material="copper",
                mesh_order=8
            ),
            mtop=LayerLevel(
                layer=LAYER.MTOP,
                thickness=thickness_mtop,
                zmin=(thickness_viac + thickness_m1 + thickness_via1 + thickness_m2 + thickness_via2 + thickness_m3 + thickness_via3),
                material="silicon_dioxide",
                mesh_order=10
            ),
            pad=LayerLevel(
                layer=LAYER.PAD,
                thickness=thickness_mtop,
                zmin=(thickness_viac + thickness_m1 + thickness_via1 + thickness_m2 + thickness_via2 + thickness_m3 + thickness_via3),
                material="silicon_dioxide",
                mesh_order=9
            ),
            # WAFER layers
            cladding=LayerLevel(
                layer=LAYER.WAFER,
                thickness=(thickness_viac + thickness_m1 + thickness_via1 + thickness_m2 + thickness_via2 + thickness_m3 + thickness_via3 + thickness_mtop + substrate_thickness),
                zmin=-substrate_thickness,
                material="silicon_dioxide",
                mesh_order=100
            ),
            substrate=LayerLevel(
                layer=LAYER.WAFER,
                thickness=substrate_thickness,
                zmin=-substrate_thickness,
                material="silicon_insulator",
                mesh_order=99
            ),
        )
    )

LAYER_STACK = get_layer_stack_electronic()

cross_sections = dict()
cells = dict()

PDK_electronic = gf.Pdk(
    name="electronic",
    cells=cells,
    cross_sections=cross_sections,
    layers=dict(LAYER),
    layer_views=LAYER_VIEWS,
    layer_stack=LAYER_STACK,
)
gf.clear_cache()


if __name__ == "__main__":

    PDK_electronic.activate()
