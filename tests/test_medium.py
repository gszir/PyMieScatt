"""Regression tests for particles in a medium (nMedium != 1). Run from the repository root:

    python -m unittest discover -s tests -t .

The wavelength argument is the vacuum wavelength: the size parameter is
x = pi*d*nMedium/wavelength and the relative index is m/nMedium.
"""
import unittest
import warnings

import numpy as np

import PyMieScatt as ps

WATER = 1.331
WAVELENGTH = 660.0


def direct_efficiencies(m, wavelength, diameter, nMedium):
    """Qext and Qsca summed directly from Mie_ab at the relative index and medium size parameter."""
    x = np.pi * diameter * nMedium / wavelength
    an, bn = ps.Mie_ab(m / nMedium, x)
    n = np.arange(1, len(an) + 1)
    qext = 2 / x**2 * np.sum((2 * n + 1) * (an.real + bn.real))
    qsca = 2 / x**2 * np.sum((2 * n + 1) * (np.abs(an)**2 + np.abs(bn)**2))
    return qext, qsca


class MediumTests(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore", message="Note: the use of nMedium")

    def test_mieq_uses_the_wavelength_in_the_medium(self):
        for m in (1.4564 + 0j, 1.5 + 0.1j):
            for diameter in (150.0, 300.0, 1000.0):
                with self.subTest(m=m, diameter=diameter):
                    q = ps.MieQ(m, WAVELENGTH, diameter, nMedium=WATER)
                    np.testing.assert_allclose(q[:2], direct_efficiencies(m, WAVELENGTH, diameter, WATER), rtol=1e-12)
                    scaled = ps.MieQ(m / WATER, WAVELENGTH / WATER, diameter, nMedium=1.0)
                    np.testing.assert_allclose(q, scaled, rtol=1e-12)

    def test_core_shell_is_continuous_at_its_shortcuts(self):
        # A zero core or equal indices go through MieQ; a negligible core uses the coated-sphere branch.
        m_shell = 1.4564 + 0j
        coated = ps.MieQCoreShell(1.587 + 0j, m_shell, WAVELENGTH, 1e-4, 300.0, nMedium=WATER)
        for shortcut in (ps.MieQCoreShell(1.587 + 0j, m_shell, WAVELENGTH, 0.0, 300.0, nMedium=WATER),
                         ps.MieQCoreShell(m_shell, m_shell, WAVELENGTH, 150.0, 300.0, nMedium=WATER)):
            np.testing.assert_allclose(shortcut[:2], coated[:2], rtol=1e-9)


if __name__ == "__main__":
    unittest.main()
