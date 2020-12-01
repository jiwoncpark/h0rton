# Get BNN samples for each dropout on the validation set
# (2 HST orbits)
python h0rton/infer_h0_mcmc_default.py experiments/v2/mcmc_default_samples_drop=0.001.json
python h0rton/infer_h0_mcmc_default.py experiments/v2/mcmc_default_samples_drop=0.005.json
python h0rton/infer_h0_mcmc_default.py experiments/v2/mcmc_default_samples_no_dropout.json
# (1 HST orbit)
python h0rton/infer_h0_mcmc_default.py experiments/v3/mcmc_default_samples_drop=0.001.json
python h0rton/infer_h0_mcmc_default.py experiments/v3/mcmc_default_samples_drop=0.005.json
python h0rton/infer_h0_mcmc_default.py experiments/v3/mcmc_default_samples_no_dropout.json
# (0.5 HST orbit)
python h0rton/infer_h0_mcmc_default.py experiments/v4/mcmc_default_samples_drop=0.001.json
python h0rton/infer_h0_mcmc_default.py experiments/v4/mcmc_default_samples_drop=0.005.json
python h0rton/infer_h0_mcmc_default.py experiments/v4/mcmc_default_samples_no_dropout.json

# Get BNN samples on the test set
python h0rton/infer_h0_mcmc_default.py experiments/v2/mcmc_default_test.json

# Run forward modeling on four lenses, 37, 43, 63, 86
python h0rton/infer_h0_forward_modeling.py experiments/v2/forward_modeling_37.json
python h0rton/infer_h0_forward_modeling.py experiments/v2/forward_modeling_43.json
python h0rton/infer_h0_forward_modeling.py experiments/v2/forward_modeling_63.json
python h0rton/infer_h0_forward_modeling.py experiments/v2/forward_modeling_86.json
