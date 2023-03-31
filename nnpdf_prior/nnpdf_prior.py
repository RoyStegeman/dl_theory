from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from n3fit.model_gen import pdfNN_layer_generator
from n3fit.io.writer import XGRID

from validphys.api import API

tf.get_logger().setLevel('ERROR')

tf.autograph.set_verbosity(0)
np.random.seed(0)

# Number of networks to initialize with random parameters
number_of_networks = 50


# input xgrid for which to produce the corresponding outputs
input_xgrid = XGRID.reshape(1, -1)


# read basis settings from a fit runcard (optional, but basis_info is useful)
fit_info = API.fit(fit="NNPDF40_nnlo_as_01180_1000").as_input()

basis_info = fit_info["fitting"]["basis"]

nodes = fit_info["parameters"]["nodes_per_layer"]
activations = fit_info["parameters"]["activation_per_layer"]
initializer_name = fit_info["parameters"]["initializer"]
layer_type = fit_info["parameters"]["layer_type"]
dropout = fit_info["parameters"]["dropout"]


nn_outputs = []
for sumrule in ["all", False]:
    res = []
    for i in tqdm(range(number_of_networks)):
        # Initialize the NNPDF model with given hyperparameters
        pdf_model = pdfNN_layer_generator(
            inp=2,
            nodes=[25, 20, 8],
            activations=["tanh", "tanh", "linear"],
            initializer_name="glorot_normal",
            layer_type="dense",
            flav_info=basis_info,
            fitbasis="EVOL",
            out=14,
            seed=np.random.randint(0, pow(2, 31)),
            dropout=0.0,
            regularizer=None,
            regularizer_args=None,
            impose_sumrule=sumrule,
            scaler=None,
            parallel_models=1,
        )

        # Generate predictions in 14-flavor basis
        out = pdf_model[0].predict({"pdf_input": input_xgrid}, verbose=False)

        # transform to 8 flavor basis: sigma, g, v, v3, v8, t3, t8, t15
        out = out[0, :, [1, 2, 3, 4, 5, 9, 10, 11]]
        res.append(out)
    nn_outputs.append(np.array(res))


################################################################################
## Plots

# plot settings
pdf_names = ["\Sigma", "g", "V", "V3", "V8", "T3", "T8", "T15"]
colors = ["C0", "C1"]
labels = ["MSR and VSR", "w/o sumrules"]
xscale = "log"


plt.clf()
handles = []
fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(9, 18))
for i, (name, ax) in enumerate(zip(pdf_names, axs.flatten())):
    for out, color in zip(nn_outputs, colors):
        ax.plot(input_xgrid[0], input_xgrid.T * out[:, i, :].T, color=color, lw=0.2)
        pdf_cv = out[:, i, :].mean(axis=0)
        line = ax.plot(input_xgrid[0], input_xgrid[0] * pdf_cv, color=color, lw=2)
        handles.append(line[0])
    ax.set_xscale(xscale)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(rf"$x{name}(x)$")
    ax.set_xlim(1e-5, 1)
fig.legend([handles[0],handles[1]],[labels[0],labels[1]])
fig.suptitle("distribution of pdfs at initialization\n")
fig.tight_layout()
fig.savefig("nnpdf_prior_replicaplot.pdf")


plt.clf()
handles = []
fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(9, 18))
for i, (name, ax) in enumerate(zip(pdf_names, axs.flatten())):
    for out, color in zip(nn_outputs, colors):
        pdf_cv = out[:, i, :].mean(axis=0)
        pdf_std = out[:, i, :].std(axis=0)
        cl_high = np.nanpercentile(out[:, i, :], 84, axis=0)
        cl_low = np.nanpercentile(out[:, i, :], 16, axis=0)

        # plot rep0 PDF
        line = ax.plot(input_xgrid[0], input_xgrid[0] * pdf_cv, color=color, lw=2, label="replica0")
        handles.append(line[0])

        # 68%c.l. band
        ax.fill_between(
            input_xgrid[0],
            input_xgrid[0] * cl_low,
            input_xgrid[0] * cl_high,
            alpha=0.4,
            color=color,
            label=r"1$\sigma$",
        )

        # 1 std lines
        ax.plot(
            input_xgrid[0], input_xgrid[0] * (pdf_cv - pdf_std), alpha=0.4, ls="dashed", color=color
        )
        ax.plot(
            input_xgrid[0],
            input_xgrid[0] * (pdf_cv + pdf_std),
            alpha=0.4,
            ls="dashed",
            color=color,
            label="68%c.l.",
        )

    ax.set_xscale(xscale)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(rf"$x{name}(x)$")
    ax.set_xlim(1e-5, 1)
fig.legend([handles[0],handles[1]],[labels[0],labels[1]])
fig.suptitle("distribution of pdfs at initialization\n")
fig.tight_layout()
fig.savefig("nnpdf_prior_bandplot.pdf")
