---
title: 'Gala: A Python package for galactic dynamics'
tags:
  - Python
  - astronomy
  - gravitational waves
  - gravitational lensing
  - geometrical optics
  - wave optics
authors:
  - name: Simon M.C. Yeung 
    orcid: 
    equal-contrib: true
    affiliation: 1
  - name: Mark H.Y. Cheung
    equal-contrib: true 
    affiliation: 2
  - name: Miguel
    affiliation: 3
affiliations:
 - name: University of Wisconsin-Milwaukee, Milwaukee, WI 53201, USA
   index: 1
 - name: William H. Miller III Department of Physics and Astronomy, Johns Hopkins University, 3400 North Charles Street, Baltimore, Maryland, 21218, USA
   index: 2
 - name: Max Planck Institute for Gravitational Physics (Albert Einstein Institute), Am M ̈uhlenberg 1, D-14476 Potsdam, Germany
   index: 3
date: 13 August 2017
bibliography: paper.bib

---

# Summary

Gravitational wave is lensed when it encounters a massive object. 
Effects of magnification and multple images take place when the gravitational wave is strongly-lensed by galaxies. 
Wave optics arsies when the gravitatioinal wave is microlensed by stellar object, which enables studies of the distribution of stars and the structure of dark matter halos.
To compute the amplification factor for the lensed waveform, diffraction integral is evaluated.
Solving the integral is an important step for the studies of different lensing scenarios and applicaitons of lensed gravitational waves.

# Statement of need

`Wolensing` is a Python package for computing wave optics amplification factor. 





`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# Method

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
