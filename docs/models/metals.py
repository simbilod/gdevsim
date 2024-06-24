# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# # Metals
#
# Metals are defined as materials with a Fermi level within the conduction band, and with associated carrier distributions also always within the band across simulation conditions. Hence, they always support a fixed density of mobile charge. Therefore, compared to the semiconductors, there are no local charge gradients, and the proportionality of current to electric field reduces to a (possibly temperature-dependent) constant:
#
# * **Parameters**:
#     * Resistivity $\rho (T)$ (Ohm*cm)
#     * Temperature coefficient of resistivity $\alpha_\rho$ ($K^{-1}$)
# * **Node solutions**:
#     * Fermi potential $\varphi_F (\bm{x})$ (V)
# * **Equations**
#     * Drift current (Ohm's law) $\bm{J} = - \rho^{-1} \nabla \varphi_F$
#     * Charge conservation (re-expressed, at DC) $ \nabla \cdot \left( \rho^{-1}\nabla\varphi_F \right) = 0 $
# * **Interfaces**
#     * Contacts
#         * $\varphi_F = V_{applied}$
#     * Metals
#         * Assume perfect resistance-free ohmic contact
#             * $\varphi_F^1 = \varphi_F^2$
#             *
#     * Insulators
#         * Continuity
#     * Semiconductors
#         * Assume perfect resistance-free ohmic contact
#
# <div class="alert alert-success">
# Notes:
#
# * Fermi potential is used to distinguish from electrostatic potential that extends into insulators and semiconductors
# * All interfaces with other metals and semiconductors are assumed ohmic
# * The above is only exact at DC frequency; at AC frequencies, a complex permittivity involving $\epsilon$ and $\sigma$ should be considered for insulators and metals.
# </div>

#
