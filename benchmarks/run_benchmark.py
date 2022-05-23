"""
Benchmark for regression and uncertainty quantification (UQ) on small molecular datasets. UQ metrics assume
standardised y-values.
Author: Ryan-Rhys Griffiths 2022
"""

import argparse
import logging

from botorch import fit_gpytorch_model
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch

from benchmark_models import TanimotoGP, ScalarProductGP
from gprotorch.dataloader import DataLoaderMP
from gprotorch.dataloader.data_utils import transform_data
from gpytorch_metrics import negative_log_predictive_density, mean_standardized_log_loss, quantile_coverage_error


# Remove Graphein warnings
logging.getLogger("graphein").setLevel("ERROR")

gp_models = {'Tanimoto': 'Tanimoto', 'Scalar Product': 'Scalar Product'}
dataset_names = {'Photoswitch': 'Photoswitch', 'ESOL': 'ESOL', 'FreeSolv': 'FreeSolv', 'Lipophilicity': 'Lipophilicity'}
dataset_paths = {'Photoswitch':'../data/property_prediction/photoswitches.csv',
                 'ESOL': '../data/property_prediction/ESOL.csv',
                 'FreeSolv': '../data/property_prediction/FreeSolv.csv',
                 'Lipophilicity': '../data/property_prediction/Lipophilicity.csv'}


def main(n_trials, test_set_size, dataset_name, dataset_path, featurisation, gp_model):
    """

    Args:
        n_trials: Number of random train/test splits for the datasets. Default is 20
        test_set_size: Size of the test set for evaluation. Default is 0.2
        dataset_name: Benchmark dataset to use. One of ['Photoswitch', 'ESOL', 'FreeSolv', 'Lipophilicity']
        dataset_path: Benchmark dataset path. One of ['../data/property_prediction/photoswitches.csv',
                                                       ../data/property_prediction/ESOL.csv',
                                                       '../data/property_prediction/FreeSolv.csv',
                                                       '../data/property_prediction/Lipophilicity.csv']
        featurisation: Choice of features. One of ['fingerprints', 'fragments', 'fragprints']
        gp_model: Choice of model. One of ['Tanimoto', 'Scalar Product']

    Returns: Evaluation of model/representation on the benchmark

    """

    if dataset_name not in dataset_names.values():
        raise ValueError(f"The specified dataset choice ({dataset_name}) is not a valid option. "
                            f"Choose one of {list(dataset_names.keys())}.")
    if dataset_path not in dataset_paths.values():
        raise ValueError(f"The specified dataset path ({dataset_path}) is not a valid option. "
                            f"Choose one of {list(dataset_paths.values())}.")

    # Load the benchmark dataset
    loader = DataLoaderMP()
    loader.load_benchmark(dataset_name, dataset_path)

    # Choose the featurisation
    loader.featurize(featurisation)
    X = loader.features
    y = loader.labels

    # initialise performance metric lists for regression
    r2_list = []
    rmse_list = []
    mae_list = []

    # initialise performance metric lists for UQ
    nlpd_list = []
    msll_list = []
    qce_list = []

    for i in range(0, n_trials):

        print(f'Trial {i} of {n_trials}')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_size, random_state=i)

        #  We standardise the outputs but leave the inputs unchanged
        _, y_train, _, y_test, y_scaler = transform_data(X_train, y_train, X_test, y_test)

        # Specify the precision. GPyTorch has issues with large datasets and float64.
        if y.size > 1000:
            precision = np.float32
        else:
            precision = np.float64

        # Convert numpy arrays to PyTorch tensors and flatten the label vectors
        X_train = torch.tensor(X_train.astype(precision))
        X_test = torch.tensor(X_test.astype(precision))
        y_train = torch.tensor(y_train.astype(precision)).flatten()
        y_test = torch.tensor(y_test.astype(precision)).flatten()

        # initialise GP likelihood and model
        likelihood = GaussianLikelihood()

        if gp_model == 'Tanimoto':
            model = TanimotoGP(X_train, y_train, likelihood)
        elif gp_model == 'Scalar Product':
            model = ScalarProductGP(X_train, y_train, likelihood)
        else:
            raise ValueError(f"The specified model choice ({gp_model}) is not a valid option. "
                            f"Choose one of {list(gp_models.keys())}.")

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # "Loss" for GPs - the marginal log likelihood
        mll = ExactMarginalLogLikelihood(likelihood, model)

        # Use the BoTorch utility for fitting GPs in order to use the LBFGS-B optimiser (recommended)
        fit_gpytorch_model(mll)

        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()

        # full GP predictive distribution
        trained_pred_dist = likelihood(model(X_test))

        # Compute NLPD on the Test set
        nlpd = negative_log_predictive_density(trained_pred_dist, y_test)

        # Compute MSLL on Test set
        msll = mean_standardized_log_loss(trained_pred_dist, y_test)

        # Compute quantile coverage error on test set
        qce = quantile_coverage_error(trained_pred_dist, y_test, quantile=95)

        print(f'NLPD: {nlpd:.2f}')
        print(f'MSLL: {msll:.2f}')
        print(f'QCE: {qce:.2f}')

        # mean and variance GP prediction
        f_pred = model(X_test)

        y_pred = f_pred.mean

        # Transform back to real data space to compute metrics and detach gradients
        y_pred = y_scaler.inverse_transform(y_pred.detach().unsqueeze(dim=1))
        y_test = y_scaler.inverse_transform(y_test.detach().unsqueeze(dim=1))

        # Output Standardised RMSE and RMSE on Train Set
        y_train = y_train.detach()
        y_pred_train = model(X_train).mean.detach()
        train_rmse_stan = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_rmse = np.sqrt(
            mean_squared_error(y_scaler.inverse_transform(y_train.unsqueeze(dim=1)),
                               y_scaler.inverse_transform(y_pred_train.unsqueeze(dim=1))))
        print("\nStandardised Train RMSE: {:.3f}".format(train_rmse_stan))
        print("Train RMSE: {:.3f}".format(train_rmse))

        # Compute R^2, RMSE and MAE on Test set
        score = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        print("\nR^2: {:.3f}".format(score))
        print("RMSE: {:.3f}".format(rmse))
        print("MAE: {:.3f}".format(mae))

        nlpd_list.append(nlpd)
        msll_list.append(msll)
        qce_list.append(qce)

        r2_list.append(score)
        rmse_list.append(rmse)
        mae_list.append(mae)

    nlpd_list = torch.tensor(nlpd_list)
    msll_list = torch.tensor(msll_list)
    qce_list = torch.tensor(qce_list)

    r2_list = np.array(r2_list)
    rmse_list = np.array(rmse_list)
    mae_list = np.array(mae_list)

    print("\nmean NLPD: {:.4f} +- {:.4f}".format(torch.mean(nlpd_list), torch.std(nlpd_list) / torch.sqrt(torch.tensor(n_trials))))
    print("\nmean MSLL: {:.4f} +- {:.4f}".format(torch.mean(msll_list), torch.std(msll_list) / np.sqrt(torch.tensor(n_trials))))
    print("\nmean QCE: {:.4f} +- {:.4f}".format(torch.mean(qce_list), torch.std(qce_list) / np.sqrt(torch.tensor(n_trials))))

    print("\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list), np.std(r2_list) / np.sqrt(len(r2_list))))
    print("mean RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list), np.std(rmse_list) / np.sqrt(len(rmse_list))))
    print("mean MAE: {:.4f} +- {:.4f}\n".format(np.mean(mae_list), np.std(mae_list) / np.sqrt(len(mae_list))))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--n_trials', type=int, default=20,
                        help='int specifying number of random train/test splits to use')
    parser.add_argument('-ts', '--test_set_size', type=float, default=0.2,
                        help='float in range [0, 1] specifying fraction of dataset to use as test set')
    parser.add_argument('-d', '--dataset', type=str, default='ESOL',
                        help='Dataset to use. One of [Photoswitch, ESOL, FreeSolv, Lipophilicity]')
    parser.add_argument('-p', '--path', type=str, default="../data/property_prediction/ESOL.csv",
                        help='Path to the dataset file. One of [../data/property_prediction/photoswitches.csv, '
                             '../data/property_prediction/ESOL.csv, '
                             '../data/property_prediction/FreeSolv.csv, '
                             '../data/property_prediction/Lipophilicity.csv]')
    parser.add_argument('-r', '--featurisation', type=str, default='fragments',
                        help='str specifying the molecular featurisation. '
                             'One of [fingerprints, fragments, fragprints].')
    parser.add_argument('-m', '--model', type=str, default='Tanimoto',
                        help='Model to use. One of [Tanimoto, Scalar Product,].')

    args = parser.parse_args()
    main(args.n_trials, args.test_set_size, args.dataset, args.path, args.featurisation, args.model)
