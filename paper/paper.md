---
title: 'wolensing: A Python package to compute wave optics amplification factor for gravitational wave'
tags:
  - Python
  - gravitational lensing
  - gravitational waves
authors:
  - name: Simon M.C. Yeung
    orcid: 
    affiliation: "1"
  - name: Mark H.Y. Cheung
    orcid: 
    affiliation: "2"
  - name: Miguel Zulmalacarregui
    orcid: 
    affiliation: "3"
  - name: Otto A. Hannuksela
    orcid: 
    affiliation: "4"
affiliations:
  - name: University of Wisconsin-Milwaukee, Milwaukee, WI 53201, USA
    index: 1
  - name: William H. Miller III Department of Physics and Astronomy, Johns Hopkins University, 3400 North Charles Street, Baltimore, Maryland, 21218, USA
    index: 2
  - name: Max Planck Institute for Gravitational Physics (Albert Einstein Institute), Am Muhlenburg 1, D-14476 Potsdam-Golm, Germany
    index: 3
  - name: Department of Physics, The Chinese University of Hong Kong, Shatin, N.T., Hong Kong
    index: 4
date: 1 December 2023
license: MIT
bibliography: paper.bib
---

# Summary

The `wolensing` Python package offers a solution for gravitational wave lensing computations within the full wave-optics regime. This tool is primarily designed to calculate the gravitational lensing amplification factor including diffractive effects, an essential component for generating accurate lensed gravitational wave waveforms. These waveforms are integral to astrophysical and cosmological studies related to gravitational-wave lensing.

Integrating with lensingGW [@Pagano_2020], wolensing provides solutions for image positions in the high-frequency regime where wave and geometrical optics converge. This functionality allows the amplification factor to be applicable across a wider frequency range. Another key feature of `wolensing` is its ability to plot time delay contours on the lens plane, offering researchers a visual tool to better understand the relationship between the lens system and the amplification factor. 

`wolensing` is compatible with various lens models in `lenstronomy` [@Birrer_2021]. There are also built-in lens models including point mass, singular isothermal sphere (SIS), and nonsingular isothermal ellipsoid (NIE) with `jax` supporting GPU computation. Users can accommodate different lens models in the code with `jax`.

`wolensing` is available as an open-source package on `PyPI` and can be installed via `pip`.  


# Statement of need

Gravitational wave lensing studies have traditionally concentrated on strong lensing, utilizing the geometrical optics approximation. This approach predicts images with varying time delays and magnifications, while maintaining uniform frequency evolution across these images. However, for lens masses around or below 3 solar masses, the scale of the Schwarzschild radius becomes comparable to the wavelength of the gravitational wave. This introduces significant diffraction effects, necessitating a shift from geometrical to wave optics for accurate modeling. In this regime, the frequency evolution of gravitational waves is influenced by the amplification factor, which is determined using a diffraction integral, resulting in a marked increase in computational complexity and cost.

The `wolensing` package addresses this challenge by providing efficient computation of the amplification factor for general lenses. To optimize computational speed, it includes built-in simple lens models that leverage `jax` for enhanced performance. Furthermore, wolensing integrates geometrical optics for high-frequency scenarios, reducing the computational cost in that regime. 

The core component of `wolensing` is a 2-dimensional integrator that estimates the area between neighboring contour lines of the lensing time delay function [@Diego_2019].
The integration method implemented works well for general lens systems and fine tuning of the settings is not required when changing the lens model.
Other than scenarios with a single lens, `wolensing` can also be used to study systems with multiple lenses.
Notably, [@Cheung_2020] and [@Yeung_2021] employed the package to analyze microlensing effects on top of type-I and type-II images produced by a Singular Isothermal Sphere (SIS) galaxy.

# Acknowledgements

