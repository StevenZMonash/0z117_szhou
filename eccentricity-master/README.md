# pycbc-waveform-SEOBNRE

Usage:

1. Compile the source c code to generate the SEOBNRE library libSEOBNRE.so

   ```bash
   gcc -fPIC -shared src/main-interface.c src/SEOBNREforPythonMain.c -L/opt/local/lib -lgsl -lgslcblas -lm -o libSEOBNRE.so
   ```

   , then put the library in your environment variable LD_LIBRARY_PATH.

2. Install the python code

   ```bash
   rm -rf build
   python setup.py install --prefix=YOUR_PATH
   ```

   , write the python path into your environment variable PYTHONPATH

3. A example to call SEOBNRE through pycbc:

   ```python
   import SEOBNRE
   from pycbc.waveform import get_td_waveform
   import numpy as np
   import lal
   import lalsimulation as lalsim
   import pylab
   
   kwargs = {'mass1':10,
             'mass2':10,
             'spin1z':0.1,
             'spin2z':0.1,
             'distance':400,
             'delta_t':1./4096,
             'eccentricity':0,
             'coa_phase':0,
             'f_lower':20,
             'long_asc_nodes':0}
   hp, hc = get_td_waveform(approximant='SEOBNRE',
                                    **kwargs)
   
   pylab.plot(hp.sample_times, hp, label='hplus')
   pylab.plot(hc.sample_times, hc, label='hcross')
   pylab.legend()
   pylab.show()
   ```

   A comparison with other SEOBNRv(1,2,3,4) waveforms can be found in [test/prod1_wfoffset_SEOBcomp.ipynb](https://gitlab.aei.uni-hannover.de/yifan.wang/eccentricity/-/blob/master/test/prod1_wfoffset_SEOBcomp.ipynb) (Note that SEOBNRE waveforms are different with others up to a coalescence phase offset).



References: 

* [https://arxiv.org/abs/1910.00784](https://arxiv.org/abs/1910.00784) (Validating the Effective-One-Body Numerical-Relativity Waveform Models for Spin-aligned Binary Black Holes along Eccentric Orbits)
* [https://journals.aps.org/prd/abstract/10.1103/PhysRevD.96.044028](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.96.044028) (Waveform model for an eccentric binary black hole based on the effective-one-body-numerical-relativity formalism)

