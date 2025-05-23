
from typing import Dict, Tuple, Callable
import anesthetic
import matplotlib.pyplot as plt
import numpy as np
import swyft
from anesthetic import make_2d_axes
from PolySwyft.PolySwyft_Settings import PolySwyft_Settings
from PolySwyft.utils import compute_KL_compression, compute_KL_divergence_truth



def plot_analysis_of_NSNRE(root: str, network_storage: Dict[int, swyft.SwyftModule],
                           samples_storage: Dict[int, anesthetic.Samples], dkl_storage: Dict[int, Tuple[float, float]],
                           polyswyftSettings: PolySwyft_Settings,
                           obs: swyft.Sample, true_posterior: anesthetic.Samples = None, deadpoints_processing: Callable = None):
    """
    Plot the analysis of the NSNRE.
    :param root: A string of the root directory to save the plots
    :param root_storage: A dictionary of roots for each round
    :param network_storage: A dictionary of networks for each round
    :param samples_storage: A dictionary of samples for each round
    :param dkl_storage: A dictionary of KL divergences for each round
    :param polyswyftSettings: A PolySwyft_Settings object
    :param obs: A Swyft sample of the observed data
    :param true_posterior: An anesthetic samples object of the true posterior if available
    :return:
    """
    from matplotlib import rc
    rc('text', usetex=True)
    rc('font', family='serif')

    # set up labels for plotting
    params_idx = [i for i in range(0, polyswyftSettings.num_features)]
    params_labels = {i: rf"${polyswyftSettings.targetKey}_{i}$" for i in range(polyswyftSettings.num_features)}

    dkl_storage_true = {}
    dkl_compression_storage = {}

    # triangle plot
    if polyswyftSettings.plot_triangle_plot:
        kinds = {'lower': 'kde_2d', 'diagonal': 'kde_1d', 'upper': "scatter_2d"}
        fig, axes = make_2d_axes(params_idx, labels=params_labels, lower=True, diagonal=True, upper=True,
                                 ticks="outer")
        if polyswyftSettings.triangle_start == 0:
            first_round_samples = samples_storage[0]
            # load prior from last round
            prior = first_round_samples.prior()
            prior.plot_2d(axes=axes, alpha=0.4, label="prior", kinds=kinds)
        for rd in range(polyswyftSettings.triangle_start, polyswyftSettings.NRE_num_retrain_rounds + 1):
            nested = samples_storage[rd]
            nested.plot_2d(axes=axes, alpha=0.4, label=fr"$p_{rd}(\theta|D)$",
                           kinds=kinds)
        if true_posterior is not None:
            true_posterior.plot_2d(axes=axes, alpha=0.9, label="true", color="red",
                                   kinds=kinds)

        axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes) / 2, len(axes)), loc='lower center',
                                ncols=polyswyftSettings.NRE_num_retrain_rounds + 2)
        fig.savefig(f"{root}/NRE_triangle_posterior.pdf")

    # KL divergence plot
    if polyswyftSettings.plot_KL_divergence:
        for rd in range(0, polyswyftSettings.NRE_num_retrain_rounds + 1):
            DKL = compute_KL_compression(samples_storage[rd], polyswyftSettings)
            dkl_compression_storage[rd] = DKL
            if true_posterior is not None:
                current_network = network_storage[rd]
                KDL_true = compute_KL_divergence_truth(polyswyftSettings=polyswyftSettings,
                                                       network=current_network.eval(),
                                                       true_posterior=true_posterior.copy(), obs=obs,
                                                       samples=samples_storage[rd])
                dkl_storage_true[rd] = KDL_true
        plt.figure(figsize=(3.5, 3.5))

        ### plot KL(P_i||P_{i-1})
        plt.errorbar(x=[i for i in range(1, polyswyftSettings.NRE_num_retrain_rounds + 1)],
                     y=[dkl_storage[i][0] for i in range(1, polyswyftSettings.NRE_num_retrain_rounds + 1)],
                     yerr=[dkl_storage[i][1] for i in range(1, polyswyftSettings.NRE_num_retrain_rounds + 1)],
                     label=r"$\mathrm{KL} (\mathcal{P}_i||\mathcal{P}_{i-1})$")
        ### plot KL(P_i||Pi)
        plt.errorbar(x=[i for i in range(0, polyswyftSettings.NRE_num_retrain_rounds + 1)],
                     y=[dkl_compression_storage[i][0] for i in range(0, polyswyftSettings.NRE_num_retrain_rounds + 1)],
                     yerr=[dkl_compression_storage[i][1] for i in range(0, polyswyftSettings.NRE_num_retrain_rounds + 1)],
                     label=r"$\mathrm{KL}(\mathcal{P}_i||\pi)$")

        if true_posterior is not None and polyswyftSettings.model is not None:
            ### plot KL(P_true||P_i)
            plt.errorbar(x=[i for i in range(0, polyswyftSettings.NRE_num_retrain_rounds + 1)],
                         y=[dkl_storage_true[i][0] for i in range(0, polyswyftSettings.NRE_num_retrain_rounds + 1)],
                         yerr=[dkl_storage_true[i][1] for i in range(0, polyswyftSettings.NRE_num_retrain_rounds + 1)],
                         label=r"$\mathrm{KL}(\mathcal{P}_{\mathrm{True}}||\mathcal{P}_i)$", linestyle="--")
            ### plot KL(P_true||Pi)
            true_posterior_samples = true_posterior.iloc[:, :polyswyftSettings.num_features].to_numpy()
            logPi = polyswyftSettings.model.prior().logpdf(true_posterior_samples)
            logP = polyswyftSettings.model.posterior(obs[polyswyftSettings.obsKey].squeeze()).logpdf(true_posterior_samples)
            dkl_compression_truth = 1/logP.shape[0]*((logP - logPi).sum())
            plt.hlines(y=dkl_compression_truth, xmin=0, xmax=polyswyftSettings.NRE_num_retrain_rounds, color="red",
                               label=r"$\mathrm{KL}(\mathcal{P}_{\mathrm{True}}||\pi)$",linestyle="--")

        plt.legend()
        plt.xlabel("retrain round")
        plt.ylabel("KL divergence")
        plt.savefig(f"{root}/kl_divergence.pdf", dpi=300, bbox_inches='tight')
        plt.close()


    if polyswyftSettings.plot_statistical_power:
        initial_size = polyswyftSettings.n_training_samples
        stats_power = np.empty(shape=(polyswyftSettings.NRE_num_retrain_rounds + 1,))
        for rd in range(0, polyswyftSettings.NRE_num_retrain_rounds + 1):
            samples = samples_storage[rd]
            if deadpoints_processing is not None:
                samples = deadpoints_processing(samples, rd)
            size = samples.shape[0]
            stats_power[rd] = size / initial_size
            initial_size += size

        plt.figure()
        plt.plot(stats_power)
        plt.xlabel("retrain round")
        plt.ylabel("num new samples / num training samples")
        plt.savefig(f"{root}/statistical_power.pdf")
        plt.close()
