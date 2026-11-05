"""Regression tests for the graphical inversion. Run from the repository root:

    python -m unittest discover -s tests -t .
"""
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import PyMieScatt as ps


class ContourIntersectionTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_recovers_the_refractive_index(self):
        m = 1.5 + 0.1j
        qsca, qabs, qback = ps.fastMieQ(m, 532, 300)
        for kwargs in ({}, {"Qback": qback}, {"n": 1.5}):
            with self.subTest(**{k: float(v) for k, v in kwargs.items()}):
                solutions = ps.ContourIntersection(qsca, qabs, 532, 300, gridPoints=60, **kwargs)[0]
                self.assertTrue(np.any(np.abs(solutions - m) < 2e-3), solutions)

    def test_size_distribution_recovers_the_refractive_index(self):
        m = 1.5 + 0.1j
        dp = np.linspace(100, 500, 40)
        ndp = 1e3 * np.exp(-0.5 * ((dp - 300) / 50)**2)
        bsca, babs, _ = ps.fastMie_SD(m, 532, dp, ndp)
        solutions = ps.ContourIntersection_SD(bsca, babs, 532, dp, ndp, gridPoints=30)[0]
        self.assertTrue(np.any(np.abs(solutions - m) < 2e-3), solutions)


if __name__ == "__main__":
    unittest.main()
