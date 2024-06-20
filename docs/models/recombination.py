# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: devsim
#     language: python
#     name: python3
# ---

# # Recombination models
#
# Recombination is an effective "sink" for electron and hole densities in the semiconductor equations.
#
# This notebook details some of the recombination models in gdevsim.

# # `bulkSRH`
#
# Bulk Shockley-Read-Hall recombination is implemented as:
#
# $$ R_{bulk\_SRH} = \frac{np - n_i^2}{\tau_p (n+n_1) + \tau_n (p+p_1)} $$
#
# with
#
# $$ n_1 = n_{i,eff} e^{\frac{E_{trap} - E_t}{kT}} $$
# $$ p_1 = n_{i,eff} e^{-\frac{E_{trap} - E_t}{kT}} $$
#
# $E_{trap}$ is the difference between the defect level and the intrinsic level.
#
# The carrier recombination lifetimes have doping and temperature dependence as follows:
#
# $$ \tau = \tau_0 \frac{1}{1 + \left( \frac{N_A + N_D}{N_{ref}} \right)^{\alpha_{SRH_N}}} \left( \frac{T}{T_{ref}} \right)^{\alpha_{SRH_T}} $$
#
#

# # `surfaceSRH`
#
# Recombination at surfaces is treated similarly to bulk SRH, but to account with increased trap density at interfaces, with a different "surface recombination velocity" in lieu of lifetime:
#
# $$ R_{surf\_SRH} = \frac{np - n_i^2}{s_p^{-1} (n+n_1) + s_n^{-1}(p+p_1)} $$
#
# The temperature dependence of the recombination velocities is approximated by the same power law as usual:
#
# $$ s = s_{0} \left( \frac{T}{T_{ref}} \right)^{\alpha_{s}} $$
#



#
