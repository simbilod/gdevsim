import gdsfactory as gf
from gdsfactory import Component
from gdsfactory.components import straight_pn, via_stack
from gdsfactory.components.via import viac
from gdsfactory.generic_tech import LAYER
from gdsfactory.typings import ComponentSpec


@gf.cell
def vertical_ge_detector(
    length: float = 100.0,
    si_width: float = 15.0,
    ge_width: float = 8.0,
    n_via_width: float = 1,
    n_doping_width: float = 2,
    p_via_width: float = 1,
    p_contact_doping_width: float = 1.5,
    p_via_offset: float = 6,
    taper: ComponentSpec | None = None,
) -> Component:
    """Germanium detector with n-contact over the germanium and side silicon p-contacts

    Arguments
        length: float, The length of the detector. Default is 100.0.
        si_width: float, The width of the silicon. Default is 15.0.
        ge_width: float, The width of the germanium. Default is 8.0.
        n_via_width: float, The width of the n-via. Default is 1.
        n_via_offset: float, The offset of the n-via. Default is 1.
        n_doping_width: float, The width of the n-doping. Default is 2.
        p_via_width: float, The width of the p-via. Default is 1.
        p_contact_doping_width: float, The width of the p-contact doping. Default is 1.5.
        p_via_offset: float, The offset of the p-via. Default is 6.
        taper: ComponentSpec | None, The taper of the component. Default is None.
    """

    c = Component()

    # Structure
    silicon = c << gf.components.rectangle(
        size=[length, si_width], layer=LAYER.WG, centered=True
    )
    _germanium = c << gf.components.rectangle(
        size=[length, ge_width], layer=LAYER.GE, centered=True
    )

    # N-doping + contact
    _n_ge = c << gf.components.rectangle(
        size=[length, n_doping_width], layer=LAYER.GEN, centered=True
    )
    n_via = c << gf.components.rectangle(
        size=[length, n_via_width], layer=LAYER.VIAC, centered=True
    )

    # P-dopings + contacts
    _p_slab = c << gf.components.rectangle(
        size=[length, si_width],
        layer=LAYER.PP,
        centered=True,
    )
    # p_contact1 = c << gf.components.rectangle(
    #     size=[length, p_contact_doping_width], layer=LAYER.PPP, centered=True
    # )
    # p_contact1.movey(p_via_offset)
    # p_contact2 = c << gf.components.rectangle(
    #     size=[length, p_contact_doping_width], layer=LAYER.PPP, centered=True
    # )
    # p_contact2.movey(-p_via_offset)
    p1_via = c << gf.components.rectangle(
        size=[length, p_via_width], layer=LAYER.VIAC, centered=True
    )
    p1_via.movey(p_via_offset)
    p2_via = c << gf.components.rectangle(
        size=[length, p_via_width], layer=LAYER.VIAC, centered=True
    )
    p2_via.movey(-p_via_offset)

    # Ports
    c.add_port("e_n", port=n_via.ports["e1"])
    c.add_port("e_p1", port=p1_via.ports["e1"])
    c.add_port("e_p2", port=p2_via.ports["e1"])

    if taper:
        taper.connect("o2", destination=silicon.ports["e1"])

    return c


@gf.cell
def lateral_ge_detector(
    length: float = 100.0,
    si_width: float = 15.0,
    ge_width: float = 8.0,
    n_via_width: float = 0.5,
    n_offset: float = 1.5,
    n_doping_width: float = 1,
    p_via_width: float = 0.5,
    p_doping_width: float = 1.0,
    p_offset: float = -1.5,
    taper: ComponentSpec | None = None,
) -> Component:
    """Germanium detector with p and n-doping in the germanium."""

    c = Component()

    # Structure
    silicon = c << gf.components.rectangle(
        size=[length, si_width], layer=LAYER.WG, centered=True
    )
    _germanium = c << gf.components.rectangle(
        size=[length, ge_width], layer=LAYER.GE, centered=True
    )

    # N-doping + contact
    n_ge = c << gf.components.rectangle(
        size=[length, n_doping_width], layer=LAYER.GEN, centered=True
    )
    n_via = c << gf.components.rectangle(
        size=[length, n_via_width], layer=LAYER.VIAC, centered=True
    )
    n_ge.movey(n_offset)
    n_via.movey(n_offset)

    # P-doping + contact
    p_ge = c << gf.components.rectangle(
        size=[length, p_doping_width], layer=LAYER.GEN, centered=True
    )
    p_via = c << gf.components.rectangle(
        size=[length, p_via_width], layer=LAYER.VIAC, centered=True
    )
    p_ge.movey(p_offset)
    p_via.movey(p_offset)

    if taper:
        taper.connect("o2", destination=silicon.ports["e1"])

    return c

@gf.cell
def straight_pn_via_ports(component: ComponentSpec):
   """Process the component to define ports on the VIAC layer."""
   c = gf.Component()

   mod_ref = c << component

   # Get top and bottom vias location
   vias = component.extract(layers=[LAYER.VIAC])
   top_via_y = vias.ymax
   bot_via_y = vias.ymin

   # Define ports there
   c.add_port(name="e_p", center=(mod_ref.xsize/2, top_via_y - 0.01), layer=LAYER.VIAC, width=vias.xsize, orientation=90)
   c.add_port(name="e_n", center=(mod_ref.xsize/2, bot_via_y + 0.01), layer=LAYER.VIAC, width=vias.xsize, orientation=270)

   return c



straight_pn_simulation = straight_pn_via_ports(straight_pn(length=15,
                                        via_stack=gf.partial(via_stack,
                                                                layers = (None, LAYER.M1,),
                                                                vias = (viac,),
                                                            ),
                                        via_stack_width=2,
                                        taper=None)
                                        )


if __name__ == "__main__":
    c = lateral_ge_detector()
    c.show()
