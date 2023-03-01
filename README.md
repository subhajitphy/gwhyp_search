# gwhyp_search
# The goal of the project is following:
### Step1: 
We generate the fake toas SEPARATELY for multiple psrs using tempo2, (available inside ```./tempo2_fake_tim```)\
The fake toa generation can be accomplished with the following command:
``` tempo2 -gr fake -f *.par -nobsd 1 -ndobs 14 -randha y -ha 8 -start 50000 -end 53000 -rms 1e-4 ```\
I used NanoGrav open mdc par files and generated fake toas using tempo2 to begin with. 
### Step2: 
then inject the hyp model parameters into it to get injected 
toas+pars (SEPARATELY for individual PSRS) using libstempo. (script is available here: ```./run/runall.py```) 
### Step3:  
Next We need to run similar to GWA analysis (using multiple psrs with model: 
white noise  +tm +hyp) to do the final parameter estimation. The script is ```hyp_search_GWA.py```\
Injected toas with gwhyp models are available at: ```./run/gwhyp_sims_try```, and the injected plots are inside: ```./run/injected_plots```  

