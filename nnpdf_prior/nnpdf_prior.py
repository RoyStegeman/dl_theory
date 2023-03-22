#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from matplotlib import pyplot as plt
import numpy as np
np.random.seed(1)

from n3fit.model_gen import pdfNN_layer_generator
from n3fit.io.writer import XGRID
from n3fit.msr import gen_integration_input
from n3fit.backends import operations as op


from validphys.api import API


# In[ ]:


# Number of networks to initialize with random parameters
number_of_networks = 5


# In[ ]:


# input xgrid for which to produce the corresponding outputs
input_xgrid = XGRID.reshape(1,-1)
integration_xgrid, _ = gen_integration_input(2000)


# In[ ]:


# read basis settings from a fit runcard (optional, but basis_info is useful)
fit_info = API.fit(fit="NNPDF40_nnlo_as_01180_1000").as_input()

basis_info = fit_info["fitting"]["basis"]

nodes = fit_info["parameters"]["nodes_per_layer"]
activations = fit_info["parameters"]["activation_per_layer"]
initializer_name = fit_info["parameters"]["initializer"]
layer_type = fit_info["parameters"]["layer_type"]
dropout = fit_info["parameters"]["dropout"]


# In[ ]:


integration_input = op.numpy_to_input(integration_xgrid, name="integration_grid")


# In[ ]:


nn_outputs = []
for i in range(number_of_networks):
    print(i)
    pdf_model = pdfNN_layer_generator(
        inp=2,
        nodes=[25, 20, 8],
        activations=['tanh','tanh','linear'],
        initializer_name="glorot_normal",
        layer_type="dense",
        flav_info=basis_info,
        fitbasis="EVOL",
        out=14,
        seed=np.random.randint(0, pow(2, 31)),
        dropout=0.0,
        regularizer=None,
        regularizer_args=None,
        impose_sumrule="All",
        scaler=None,
        parallel_models=1,
    )
    out = pdf_model[0].predict( {"pdf_input": input_xgrid, "integrator_input": integration_input})

    # outputs basis: sigma, g, v, v3, v8, t3, t8, t15
    out = out[0,:,[1,2,3,4,5,9,10,11]]

    nn_outputs.append(out)
nn_outputs = np.array(nn_outputs)


# In[ ]:


pdf_model[0].summary()


# In[ ]:


pdf_model[0].layers


# In[ ]:


pdf_names = ["\Sigma", "g", "V", "V3", "V8", "T3", "T8", "T15"]

fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(9, 18))
for i, (name, ax) in enumerate(zip(pdf_names, axs.flatten())):
    ax.plot(input_xgrid[0], input_xgrid.T*nn_outputs[:,i,:].T, color='C0', lw=0.2)
    pdf_cv = nn_outputs[:,i,:].mean(axis=0)
    ax.plot(input_xgrid[0], input_xgrid[0]*pdf_cv, color='C0', lw=2)
    ax.set_xscale("log")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(rf"$x{name}(x)$")
    ax.set_xlim(1e-5,1)

fig.tight_layout()

