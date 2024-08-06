import importlib
from copy import copy, deepcopy
from typing import Any, Optional

import torch
from gpytorch import settings
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.models import ExactGP
from gpytorch.models.exact_prediction_strategies import prediction_strategy
from loguru import logger


def load_class(module_name: str, class_name: str) -> Any:
    """
    Dynamically load a class from a given module with error handling.

    Args:
        module_name (str): The name of the module.
        class_name (str): The name of the class to load.

    Returns:
        class: The dynamically loaded class.

    Raises:
        ImportError: If the module cannot be loaded.
        AttributeError: If the class cannot be found in the module.
    """
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except ImportError as err:
        raise ImportError(
            f"Module {module_name} could not be loaded: {str(err)}"
        ) from err
    except AttributeError as err:
        raise AttributeError(
            f"Class {class_name} not found in {module_name}: {str(err)}"
        ) from err


class NonTensorialInputs:
    def __init__(self, data):
        self.data = data

    def append(self, new_data):
        self.data.extend(new_data.data)

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __deepcopy__(self, memo):
        return NonTensorialInputs(copy(self.data))


class SIGP(ExactGP):
    """
    A reimplementation of gpytorch's ExactGP that allows for non-tensorial inputs.
    The inputs to this class may be a gauche.NonTensorialInputs instance, with graphs
    stored within the object's .data attribute.

    In the longer term, if ExactGP can be refactored such that the validation checks ensuring
    that the inputs are torch.Tensors are optional, this class should subclass ExactGP without
    performing those checks.
    """

    def __init__(self, train_inputs, train_targets, likelihood):
        if (
            train_inputs is not None
            and type(train_inputs) is NonTensorialInputs
        ):
            train_inputs = (train_inputs,)
        if not isinstance(likelihood, _GaussianLikelihoodBase):
            raise RuntimeError("SIGP can only handle Gaussian likelihoods")

        super(ExactGP, self).__init__()
        if train_inputs is not None:
            self.train_inputs = tuple(
                (
                    i.unsqueeze(-1)
                    if torch.is_tensor(i) and i.ndimension() == 1
                    else i
                )
                for i in train_inputs
            )
            self.train_targets = train_targets
        else:
            self.train_inputs = None
            self.train_targets = None
        self.likelihood = likelihood

        self.prediction_strategy = None

    def __call__(self, *args, **kwargs):
        train_inputs = (
            list(self.train_inputs) if self.train_inputs is not None else []
        )

        inputs = [
            (
                i.unsqueeze(-1)
                if torch.is_tensor(i) and i.ndimension() == 1
                else i
            )
            for i in args
        ]

        # Training mode: optimizing
        if self.training:
            if self.train_inputs is None:
                raise RuntimeError(
                    "train_inputs, train_targets cannot be None in training mode. "
                    "Call .eval() for prior predictions, or call .set_train_data() to add training data."
                )
            res = super(ExactGP, self).__call__(*inputs, **kwargs)
            return res

        # Prior mode
        elif (
            settings.prior_mode.on()
            or self.train_inputs is None
            or self.train_targets is None
        ):
            full_inputs = args
            full_output = super(ExactGP, self).__call__(*full_inputs, **kwargs)
            if settings.debug().on():
                if not isinstance(full_output, MultivariateNormal):
                    raise RuntimeError(
                        "SIGP.forward must return a MultivariateNormal"
                    )
            return full_output

        # Posterior mode
        else:
            # Get the terms that only depend on training data
            if self.prediction_strategy is None:
                train_output = super(ExactGP, self).__call__(
                    *train_inputs, **kwargs
                )

                # Create the prediction strategy for
                self.prediction_strategy = prediction_strategy(
                    train_inputs=train_inputs,
                    train_prior_dist=train_output,
                    train_labels=self.train_targets,
                    likelihood=self.likelihood,
                )

            # Concatenate the input to the training input
            full_inputs = []
            if torch.is_tensor(train_inputs[0]):
                batch_shape = train_inputs[0].shape[:-2]
                for train_input, input in zip(train_inputs, inputs):
                    # Make sure the batch shapes agree for training/test data
                    if batch_shape != train_input.shape[:-2]:
                        batch_shape = torch.broadcast_shapes(
                            batch_shape, train_input.shape[:-2]
                        )
                        train_input = train_input.expand(
                            *batch_shape, *train_input.shape[-2:]
                        )
                    if batch_shape != input.shape[:-2]:
                        batch_shape = torch.broadcast_shapes(
                            batch_shape, input.shape[:-2]
                        )
                        train_input = train_input.expand(
                            *batch_shape, *train_input.shape[-2:]
                        )
                        input = input.expand(*batch_shape, *input.shape[-2:])
                    full_inputs.append(torch.cat([train_input, input], dim=-2))
            else:
                # from IPython.core.debugger import set_trace; set_trace()
                full_inputs = deepcopy(train_inputs)
                full_inputs[0].append(inputs[0])

            # Get the joint distribution for training/test data
            full_output = super(ExactGP, self).__call__(*full_inputs, **kwargs)
            if settings.debug().on():
                if not isinstance(full_output, MultivariateNormal):
                    raise RuntimeError(
                        "SIGP.forward must return a MultivariateNormal"
                    )
            full_mean, full_covar = (
                full_output.loc,
                full_output.lazy_covariance_matrix,
            )

            # Determine the shape of the joint distribution
            batch_shape = full_output.batch_shape
            joint_shape = full_output.event_shape
            tasks_shape = joint_shape[1:]  # For multitask learning
            test_shape = torch.Size(
                [
                    joint_shape[0] - self.prediction_strategy.train_shape[0],
                    *tasks_shape,
                ]
            )

            # Make the prediction
            with settings.cg_tolerance(settings.eval_cg_tolerance.value()):
                (
                    predictive_mean,
                    predictive_covar,
                ) = self.prediction_strategy.exact_prediction(
                    full_mean, full_covar
                )

            # Reshape predictive mean to match the appropriate event shape
            predictive_mean = predictive_mean.view(
                *batch_shape, *test_shape
            ).contiguous()
            return full_output.__class__(predictive_mean, predictive_covar)

    @classmethod
    def save(
        cls,
        model: "SIGP",
        optimizer: Optional[torch.optim.Optimizer] = None,
        filename: str = "model.pth",
    ) -> None:
        """
        Saves the model state, optimizer state, training data, and other configurations to a file.

        Args:
            model (SIGP): The model instance to save.
            optimizer (Optional[torch.optim.Optimizer]): The optimizer associated with the model. Default is None.
            filename (str): The filename where the model state will be saved. Default is "model.pth".

        Returns:
            None
        """
        logger.info(f"Saving model state to {filename}")
        model_state = {
            "version": "0.1.0",
            "model_state_dict": model.state_dict(),
            "optimizer_class": (
                optimizer.__class__.__name__ if optimizer is not None else None
            ),
            "optimizer_state_dict": (
                optimizer.state_dict() if optimizer is not None else None
            ),
            "train_inputs": model.train_inputs,
            "train_targets": model.train_targets,
            "likelihood_state_dict": model.likelihood.state_dict(),
            "covariance_state_dict": model.covariance.state_dict(),
            "mean_module_state_dict": model.mean.state_dict(),
            "model_class": model.__class__.__name__,
            "likelihood_class": model.likelihood.__class__.__name__,
            "covar_module_class": model.covariance.__class__.__name__,
            "mean_module_class": model.mean.__class__.__name__,
            "covar_module_args": (
                {"node_label": model.covariance.node_label}
                if hasattr(model.covariance, "node_label")
                else {}
            ),
        }
        torch.save(model_state, filename)

    @classmethod
    def load(cls, filename: str = "model.pth") -> "SIGP":
        """
        Load the model state and other configurations from a file for inference.

        Args:
            filename (str): The filename from which to load the model state. Default is "model.pth".

        Returns:
            SIGP: The loaded model instance.

        Raises:
            ValueError: If the model class specified in the file is not found.
        """
        logger.info(f"Loading model state from {filename}")
        model_state = torch.load(filename)

        if model_state.get("version", "0.0.0") != "0.1.0":
            logger.warning(
                f"Loading model version {model_state.get('version', '0.0.0')}. Current version is 0.1.0."
            )

        # Dynamically get the class from globals() or a predefined mapping if not available directly
        ModelClass = globals().get(model_state["model_class"], None)
        if ModelClass is None:
            raise ValueError(
                f"Model class {model_state['model_class']} not found."
            )

        LikelihoodClass = load_class(
            "gpytorch.likelihoods", model_state["likelihood_class"]
        )
        if LikelihoodClass is None:
            raise ValueError(
                f"Likelihood class {model_state['likelihood_class']} not found."
            )
        CovarModuleClass = load_class(
            "gauche.kernels.graph_kernels", model_state["covar_module_class"]
        )
        if CovarModuleClass is None:
            raise ValueError(
                f"Covariance module class {model_state['covar_module_class']} not found."
            )
        MeanModuleClass = load_class(
            "gpytorch.means", model_state["mean_module_class"]
        )

        model = ModelClass(
            train_x=model_state["train_inputs"],
            train_y=model_state["train_targets"],
            likelihood=LikelihoodClass(),
            covar_module=CovarModuleClass(**model_state["covar_module_args"]),
            mean_module=MeanModuleClass(),
        )
        model.load_state_dict(model_state["model_state_dict"])
        model.likelihood.load_state_dict(model_state["likelihood_state_dict"])
        model.covariance.load_state_dict(model_state["covariance_state_dict"])
        model.mean.load_state_dict(model_state["mean_module_state_dict"])

        model.eval()
        return model

    @staticmethod
    def load_optimizer(
        filename: str = "model.pth",
    ) -> Optional[torch.optim.Optimizer]:
        """
        Load the optimizer state from a file.

        Args:
            filename (str): The filename from which to load the optimizer state. Default is "model.pth".

        Returns:
            Optional[torch.optim.Optimizer]: The loaded optimizer if available, None otherwise.
        """
        logger.info(f"Loading optimizer state from {filename}")
        model_state = torch.load(filename)

        optimizer_class_name = model_state.get("optimizer_class")
        optimizer_state_dict = model_state.get("optimizer_state_dict")

        if optimizer_class_name is None or optimizer_state_dict is None:
            logger.warning("No optimizer information found in the saved file.")
            return None

        OptimizerClass = getattr(torch.optim, optimizer_class_name, None)
        if OptimizerClass is None:
            logger.warning(
                f"Optimizer class {optimizer_class_name} not found in torch.optim."
            )
            return None

        # Create a dummy optimizer with a single parameter
        optimizer = OptimizerClass([torch.nn.Parameter(torch.empty(1))])
        optimizer.load_state_dict(optimizer_state_dict)

        return optimizer
