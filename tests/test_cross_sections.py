"""Regression tests for asCrossSection results. Run from the repository root:

    python -m unittest discover -s tests -t .
"""
import unittest

import numpy as np

import PyMieScatt as ps


class CrossSectionTests(unittest.TestCase):
    def test_ratio_is_dimensionless(self):
        m, wavelength = 1.5 + 0.1j, 532.0
        for name, function, diameter in (("MieQ", ps.MieQ, 300.0), ("RayleighMieQ", ps.RayleighMieQ, 5.0),
                                         ("LowFrequencyMieQ", ps.LowFrequencyMieQ, 20.0)):
            with self.subTest(function=name):
                efficiencies = function(m, wavelength, diameter, nMedium=1.0, asDict=True)
                cross_sections = function(m, wavelength, diameter, nMedium=1.0, asDict=True, asCrossSection=True)
                area = np.pi * (diameter / 2)**2
                self.assertAlmostEqual(cross_sections["Cext"] / (efficiencies["Qext"] * area), 1, places=12)
                self.assertAlmostEqual(cross_sections["Cratio"], cross_sections["Cback"] / cross_sections["Csca"], places=12)
                self.assertAlmostEqual(cross_sections["Cratio"], efficiencies["Qratio"], places=12)


if __name__ == "__main__":
    unittest.main()
