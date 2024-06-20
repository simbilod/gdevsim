from gdsfactory.technology import LayerLevel, LayerMap, LayerStack

Layer = tuple[int, int]


class GenericLayerMap(LayerMap):
    """Generic silicon photonic layermap based on the book:

    Lukas Chrostowski, Michael Hochberg, "Silicon Photonics Design",
    Cambridge University Press 2015, page 353

    You will need to create a new LayerMap with your specific foundry layers.
    """

    WAFER: Layer = (99999, 0)

    WG: Layer = (1, 0)
    WGCLAD: Layer = (111, 0)
    SLAB150: Layer = (2, 0)
    SHALLOW_ETCH: Layer = (2, 6)
    SLAB90: Layer = (3, 0)
    DEEP_ETCH: Layer = (3, 6)
    DEEPTRENCH: Layer = (4, 0)
    GE: Layer = (5, 0)
    UNDERCUT: Layer = (6, 0)
    WGN: Layer = (34, 0)
    WGN_CLAD: Layer = (36, 0)

    N: Layer = (20, 0)
    NP: Layer = (22, 0)
    NPP: Layer = (24, 0)
    P: Layer = (21, 0)
    PP: Layer = (23, 0)
    PPP: Layer = (25, 0)
    GEN: Layer = (26, 0)
    GEP: Layer = (27, 0)

    HEATER: Layer = (47, 0)
    M1: Layer = (41, 0)
    M2: Layer = (45, 0)
    M3: Layer = (49, 0)
    MTOP: Layer = (49, 0)
    VIAC: Layer = (40, 0)
    VIA1: Layer = (44, 0)
    VIA2: Layer = (43, 0)


LAYER = GenericLayerMap()


def get_layer_stack_photonic(
    thickness_wg=0.22,
    thickness_slab=0.09,
    thickness_ge=0.5,
    thickness_via=1.5,
    thickness_m1=1.0,
    thickness_box=1,
    thickness_clad=3,
) -> LayerStack:
    return LayerStack(
        layers=dict(
            core=LayerLevel(
                layer=LAYER.WG,
                thickness=thickness_wg,
                zmin=0,
                material="silicon",
                mesh_order=1,
            ),
            slab=LayerLevel(
                layer=LAYER.SLAB90,
                thickness=thickness_slab,
                zmin=0,
                material="silicon",
                mesh_order=2,
            ),
            ge=LayerLevel(
                layer=LAYER.GE,
                thickness=thickness_ge,
                zmin=thickness_wg,
                material="germanium",
                mesh_order=3,
                z_to_bias=((0, 1), (0, -1)),
            ),
            via=LayerLevel(
                layer=LAYER.VIAC,
                thickness=thickness_via + thickness_m1 / 2,
                zmin=thickness_slab,
                material="aluminum",
                mesh_order=4,
                sidewall_angle=-5,
            ),
            box=LayerLevel(
                layer=LAYER.WAFER,
                thickness=thickness_box,
                zmin=-thickness_box,
                material="silicon_dioxide",
                mesh_order=5,
            ),
            clad=LayerLevel(
                layer=LAYER.WAFER,
                thickness=thickness_clad,
                zmin=0,
                material="silicon_dioxide",
                mesh_order=6,
            ),
            n=LayerLevel(
                layer=LAYER.N,
                layer_type="doping",
                into=("core", "slab"),
                info={
                    "impulse_profiles": (
                        {
                            "function": "gaussian_impulse",
                            "parameters": {
                                "range": 0.1,
                                "vertical_straggle": 0.5,
                                "lateral_straggle": 0.02,
                            },
                        },
                    ),
                    "ion_types": ("donor",),
                    "peak_concentrations": (1e18,),
                },
            ),
            np=LayerLevel(
                layer=LAYER.NP,
                layer_type="doping",
                into=("core", "slab"),
                info={
                    "impulse_profiles": (
                        {
                            "function": "gaussian_impulse",
                            "parameters": {
                                "range": 0.0,
                                "vertical_straggle": 0.5,
                                "lateral_straggle": 0.02,
                            },
                        },
                    ),
                    "ion_types": ("donor",),
                    "peak_concentrations": (1e18,),
                },
            ),
            npp=LayerLevel(
                layer=LAYER.NPP,
                layer_type="doping",
                into=("core", "slab"),
                info={
                    "impulse_profiles": (
                        {
                            "function": "gaussian_impulse",
                            "parameters": {
                                "range": 0.0,
                                "vertical_straggle": 0.5,
                                "lateral_straggle": 0.02,
                            },
                        },
                    ),
                    "ion_types": ("donor",),
                    "peak_concentrations": (1e20,),
                },
            ),
            p=LayerLevel(
                layer=LAYER.P,
                layer_type="doping",
                into=("core", "slab"),
                info={
                    "impulse_profiles": (
                        {
                            "function": "gaussian_impulse",
                            "parameters": {
                                "range": 0.1,
                                "vertical_straggle": 0.5,
                                "lateral_straggle": 0.02,
                            },
                        },
                    ),
                    "ion_types": ("acceptor",),
                    "peak_concentrations": (1e18,),
                },
            ),
            pp=LayerLevel(
                layer=LAYER.PP,
                layer_type="doping",
                into=("core", "slab"),
                info={
                    "impulse_profiles": (
                        {
                            "function": "gaussian_impulse",
                            "parameters": {
                                "range": 0.0,
                                "vertical_straggle": 0.5,
                                "lateral_straggle": 0.02,
                            },
                        },
                    ),
                    "ion_types": ("acceptor",),
                    "peak_concentrations": (1e18,),
                },
            ),
            ppp=LayerLevel(
                layer=LAYER.PPP,
                layer_type="doping",
                into=("core", "slab"),
                info={
                    "impulse_profiles": (
                        {
                            "function": "gaussian_impulse",
                            "parameters": {
                                "range": 0.0,
                                "vertical_straggle": 0.5,
                                "lateral_straggle": 0.02,
                            },
                        },
                    ),
                    "ion_types": ("acceptor",),
                    "peak_concentrations": (1e20,),
                },
            ),
            gen=LayerLevel(
                layer=LAYER.GEN,
                layer_type="doping",
                into=("ge",),
                info={
                    "impulse_profiles": (
                        {
                            "function": "gaussian_impulse",
                            "parameters": {
                                "range": 0.0,
                                "vertical_straggle": 0.03,
                                "lateral_straggle": 0.02,
                            },
                        },
                    ),
                    "ion_types": ("donor",),
                    "peak_concentrations": (1e18,),
                },
            ),
            gep=LayerLevel(
                layer=LAYER.GEP,
                layer_type="doping",
                into=("ge",),
                info={
                    "impulse_profiles": (
                        {
                            "function": "gaussian_impulse",
                            "parameters": {
                                "range": 0.0,
                                "vertical_straggle": 0.03,
                                "lateral_straggle": 0.02,
                            },
                        },
                    ),
                    "ion_types": ("acceptor",),
                    "peak_concentrations": (1e18,),
                },
            ),
        )
    )
