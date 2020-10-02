#echo "$1"
#python h0rton/generate_summary.py ${1} simple_mc_default
python h0rton/infer_h0_simple_mc_truth.py experiments/v0/simple_mc_default.json
python h0rton/summarize.py 0 simple_mc_default
python h0rton/infer_h0_mcmc_default.py experiments/v1/mcmc_default.json
python h0rton/infer_h0_mcmc_default.py experiments/v2/mcmc_default.json
python h0rton/infer_h0_mcmc_default.py experiments/v3/mcmc_default.json
python h0rton/infer_h0_mcmc_default.py experiments/v4/mcmc_default.json
python h0rton/summarize.py 1 mcmc_default
python h0rton/summarize.py 2 mcmc_default
python h0rton/summarize.py 3 mcmc_default
python h0rton/summarize.py 4 mcmc_default
python h0rton/combine_lenses.py 0 
python h0rton/combine_lenses.py 1 
python h0rton/combine_lenses.py 2 
python h0rton/combine_lenses.py 3 
python h0rton/combine_lenses.py 4 

