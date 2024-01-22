---
title: 'wolensing: A Python package to compute wave optics amplification factor for gravitational wave'
tags:
  - Python
  - gravitational lensing
  - gravitational waves
authors:
  - name: Simon M.C. Yeung
    orcid: 0000-0002-7926-4935  #to be modified
    affiliation: "1"
  - name: Mark H.Y. Cheung
    orcid: 0000-0002-2700-4605
    affiliation: "2"
affiliations:
  - name: University of Wisconsin-Milwaukee, Milwaukee, WI 53201, USA
    index: 1
  - name: William H. Miller III Department of Physics and Astronomy, Johns Hopkins University, 3400 North Charles Street, Baltimore, Maryland, 21218, USA
    index: 2
date: 1 December 2023
license: MIT
bibliography: paper.bib
---

# Summary

`wolensing` is a Python package for gravitational lensing in wave optics. 

The package computes the amplification factor in full wave optics, which could be used in obtaining lensed gravitational wave waveforms for further analyses of astrophysics and cosmology.  `wolensing`  also utilizes `lensingGW` [@Pagano_2020] to solve the image positions with geometrical optics which wave optics converges in high frequency regime. This allows the amplification factor to be extended beyond the wave optics regime to cover an extensive frequency range. 

`wolensing` also allows the user to plot the time delay contours of the lens plane, giving a better understanding of the interplay between the lens system and the amplification factor. 

`wolensing` is compatible with various lens models in `lenstronomy` [@Birrer_2021]. There are also built-in lens models including Point Mass, Singular Isothermal Sphere (SIS), and Nonsingular Isothermal Ellipsoid (NIE) with `jax` supporting GPU computation. Users can accommodate different lens models in the code with `jax`.

The package is available on `pip` and open source.  


# Statement of need

Most gravitational wave lensing studies are focused around strong lensing which employs geometrical optics. In the geometrical optics approximation, images of different time delays, magnifications are created, and the frequency evolutions of the images are the same. For lenses of smaller masses of order of 3 solar mass or lower, the Schwarzschild radius is comparable to the wavelength of gravitational wave, diffraction effect takes place. Frequency evolution can be described by amplification factor with diffraction integral, and the computational cost has significantly increased. 

`wolensing` computes the amplification factor for general lenses. To enhance the computational speed, the package has built-in simple lens models with `jax`. Combining with geometrical optics in high frequency regime, the amplification factor enables detailed analyses of different lens systems. Embedded lens system is also supported. [@Cheung_2021, @Yeung_2023] analysed microlensing on type-I and type-II images created by SIS galaxy.

# Acknowledgements

