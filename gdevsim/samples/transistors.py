import gdsfactory as gf
from gdsfactory import Component
from gdsfactory.typings import Layer

from gdevsim.samples.layers_electronic import LAYER


@gf.cell
def vertical_bjt(
    via_layer: Layer = LAYER.VIAC,
    emitter_doping_layer: Layer = LAYER.BJT_EMITTER,
    size_emitter: tuple[float, float] = (5.0, 5.0),
    emitter_base_via_distance: float = 10,
    collector_base_via_distance: float = 10,
    collector_doping_layer: Layer = LAYER.BJT_COLLECTOR,
    base_doping_layer: Layer = LAYER.BJT_BASE,
    isolation_doping_layer: Layer = LAYER.BJT_ISOLATION,
    base_emitter_overplot: float = 1.0,
    collector_base_overplot: float = 1.0,
    isolation_collector_overplot: float = 1.0,
    via_emitter_underplot: float = 1.0,
) -> Component:
    """This function creates a vertical bipolar junction transistor (BJT) cell.

    The BJT is defined from vertical dopings.

    Arguments:
        emitter_doping_layer: Layer, The layer of the emitter doping. Default is LAYER.BJT_EMITTER.
        size_emitter: tuple[float, float], The size of the emitter doping. Default is (5.0, 5.0).
        emitter_base_via_distance: float, The distance between the emitter and the base via. Default is 10.
        collector_base_via_distance: float, The distance between the collector and the base via. Default is 10.
        collector_doping_layer: Layer, The layer of the collector doping. Default is LAYER.BJT_COLLECTOR.
        base_doping_layer: Layer, The layer of the base doping. Default is LAYER.BJT_BASE.
        isolation_doping_layer: Layer, The layer of the isolation doping. Default is LAYER.BJT_ISOLATION.
        base_emitter_overplot: float, The width of the base-emitter overplot. Default is 1.0.
        collector_base_overplot: float, The width of the collector-base overplot. Default is 1.0.
        isolation_collector_overplot: float, The width of the isolation-collector overplot. Default is 1.0.
        via_emitter_underplot: float, The width of the emitter via underplot. Default is 1.0.
    """

    c = Component()

    # Vias
    emitter_via = c << gf.components.rectangle(size=(size_emitter[0] - 2 * via_emitter_underplot, size_emitter[1] - 2 * via_emitter_underplot), layer=via_layer, centered=True)
    emitter_via.movex(-emitter_base_via_distance)

    collector_via = c << gf.components.rectangle(size=(size_emitter[0] - 2 * via_emitter_underplot, size_emitter[1] - 2 * via_emitter_underplot), layer=via_layer, centered=True)
    collector_via.movex(collector_base_via_distance)

    base_via = c << gf.components.rectangle(size=(size_emitter[0] - 2 * via_emitter_underplot, size_emitter[1] - 2 * via_emitter_underplot), layer=via_layer, centered=True)

    # Dopings
    emitter_doping = c << gf.components.rectangle(size=size_emitter, layer=emitter_doping_layer, centered=True)
    emitter_doping.x = emitter_via.x

    base_doping_dx = base_via.xmax - emitter_doping.xmin + base_emitter_overplot + via_emitter_underplot
    base_doping = c << gf.components.rectangle(size=(base_doping_dx, size_emitter[1] + 2 * base_emitter_overplot), layer=base_doping_layer, centered=True)
    base_doping.xmin = emitter_doping.xmin - base_emitter_overplot

    collector_doping_dx = collector_via.xmax - base_doping.xmin + collector_base_overplot + via_emitter_underplot
    collector_doping = c << gf.components.rectangle(size=(collector_doping_dx, size_emitter[1] + 2 * base_emitter_overplot + 2 * collector_base_overplot), layer=collector_doping_layer, centered=True)
    collector_doping.xmin = base_doping.xmin - collector_base_overplot

    isolation_doping_dx = collector_doping_dx + 2 * isolation_collector_overplot
    isolation_doping = c << gf.components.rectangle(size=(isolation_doping_dx, size_emitter[1] + 2 * base_emitter_overplot + 2 * collector_base_overplot + 2 * isolation_collector_overplot), layer=isolation_doping_layer, centered=True)
    isolation_doping.xmin = collector_doping.xmin - isolation_collector_overplot


    # Ports
    c.add_port(name="emitter", port=emitter_via.ports["e1"])
    c.add_port(name="collector", port=collector_via.ports["e1"])
    c.add_port(name="base", port=base_via.ports["e1"])

    return c


# @gf.cell
# def vertical_jfet(
#     size_gate: tuple[float, float] = (5.0, 5.0),
#     source_drain_overplot_from_gate_x: float = 5,
#     gate_channel_overplot_y: float = 5,
#     channel_doping_layer: Layer = LAYER.JFET_CHANNEL,
#     gate_doping_layer: Layer = LAYER.JFET_GATE,
#     via_layer: Layer = LAYER.VIAC,
# ) -> Component:
#     """This function creates a vertical junction field-effect (JFET) cell.

#     See e.g. https://www.allaboutcircuits.com/textbook/semiconductors/chpt-2/junction-field-effect-transistors/

#     Arguments:

#     """

#     c = Component()

#     # Dopings
#     c << gf.components.rectangle(size=(size_gate[0], size_gate[1] + 2 * gate_channel_overplot_y), layer=gate_doping_layer, centered=True)
#     c << gf.components.rectangle(size=(size_gate[0] + 2 * source_drain_overplot_from_gate_x, size_gate[1]), layer=channel_doping_layer, centered=True)

#     # Vias
#     c << gf.components.rectangle(size=(size_emitter[0] - 2 * via_emitter_underplot, size_emitter[1] - 2 * via_emitter_underplot), layer=emitter_doping_layer, centered=True)


#     # Ports
#     c.add_port(name="emitter", port=emitter_via.ports["e1"])
#     c.add_port(name="collector", port=collector_via.ports["e1"])
#     c.add_port(name="base", port=base_via.ports["e1"])

#     return c



if __name__ == "__main__":
    c = vertical_bjt()
    c.show()
