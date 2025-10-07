# Agent-based-simulations-of-lung-tumour-evolution
A repository of the code used to simulate bulk sequencing data from lung tumours evolving under genetic selection pressures, from our upcoming paper "Agent-based simulations of lung tumour evolution suggests ongoing cell competition drives realistic clonal expansions". Here we compare three alternative models.

## Usage
-model_1.py contains the code used to run Model 1, which is loosely based on SCIMET and describes weak competition (for proliferation only) during continual expansion.
-model_2.py contains the code used to run Model 2, which describes competition for survival and proliferation during sequential rounds of expansion.
-model_3.py contains the code used to run Model 3, which describes competition for survival and proliferation in a fixed-size tumour.
-compute-summary-statistics.ipynb contains the notebook used to extract summary statistics from simulated tumours.
