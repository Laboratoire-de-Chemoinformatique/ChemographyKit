import contextlib
import logging
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn
from tqdm.auto import tqdm


class DataStandardizer:
    """
    A utility class for standardizing data using mean and standard deviation.

    This class provides functionality to center and scale data by removing the mean
    and dividing by the standard deviation. It handles NaN values appropriately
    and provides warnings for numerical issues.

    Attributes:
        with_mean (bool): Whether to center the data by subtracting the mean
        with_std (bool): Whether to scale the data by dividing by standard deviation
        data_mean (Optional[torch.Tensor]): The mean of the fitted data
        data_std (Optional[torch.Tensor]): The standard deviation of the fitted data
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True) -> None:
        """
        Initialize the DataStandardizer.

        Args:
            with_mean: Whether to center the data by subtracting the mean
            with_std: Whether to scale the data by dividing by standard deviation
        """
        self.with_mean = with_mean
        self.with_std = with_std
        self.data_mean: Optional[torch.Tensor] = None
        self.data_std: Optional[torch.Tensor] = None

    @classmethod
    def nanstd(
        cls, x: torch.Tensor, dim: Union[int, Tuple[int, ...]], keepdim: bool = False
    ) -> torch.Tensor:
        """
        Compute the standard deviation ignoring NaN values.

        This method calculates the standard deviation of a tensor along specified
        dimensions while ignoring NaN values. It's equivalent to numpy's nanstd
        function but implemented using PyTorch operations.

        Args:
            x: Input tensor
            dim: Dimension(s) along which to compute the standard deviation
            keepdim: Whether to keep the reduced dimensions

        Returns:
            torch.Tensor: Standard deviation tensor with NaN values ignored
        """
        result = torch.sqrt(
            torch.nanmean(
                torch.pow(torch.abs(x - torch.nanmean(x, dim=dim).unsqueeze(dim)), 2),
                dim=dim,
            )
        )

        if keepdim:
            result = result.unsqueeze(dim)

        return result

    def fit_transform(
        self, X: Union[torch.Tensor, np.ndarray], axis: int = 0
    ) -> torch.Tensor:
        """
        Fit the standardizer to the data and transform it.

        This method computes the mean and standard deviation of the input data
        and then applies standardization (centering and scaling) to the data.
        It handles NaN values and provides warnings for numerical issues.

        Args:
            X: Input data to be standardized. Can be a torch.Tensor or numpy array
            axis: Axis along which to compute the mean and standard deviation

        Returns:
            torch.Tensor: Standardized data tensor

        Warns:
            UserWarning: If numerical issues are encountered during centering or scaling
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float64)

        if self.with_mean:
            mean_ = torch.nanmean(X, dim=axis, keepdim=True)
            self.data_mean = mean_

            # Check for extreme values that might cause numerical issues
            if torch.any(torch.abs(X) > 1e5):
                warnings.warn(
                    "Extreme values detected in the data. "
                    "This may cause numerical issues during standardization.",
                    UserWarning,
                )

        if self.with_std:
            scale_ = self.nanstd(X, dim=axis, keepdim=True)
            self.data_std = scale_

            # Check for zero or near-zero variance features
            if torch.any(scale_ < 1e-8):
                warnings.warn(
                    "Features with zero or near-zero variance detected. "
                    "This may cause numerical issues during standardization.",
                    UserWarning,
                )

        # Center the data if required
        if self.with_mean:
            X = X - self.data_mean
            mean_1 = torch.nanmean(X, dim=axis)
            if not torch.allclose(mean_1, torch.zeros_like(mean_1), atol=1e-8):
                warnings.warn(
                    "Numerical issues were encountered when centering the data. "
                    "Dataset may contain very large values. You may need to prescale your features."
                )
                X = X - mean_1

        # Scale the data if required
        if self.with_std:
            scale_ = torch.clamp(self.data_std, min=1e-8)  # Handle zeros in scale
            X = X / scale_

            if self.with_mean:
                mean_2 = torch.nanmean(X, dim=axis)
                if not torch.allclose(mean_2, torch.zeros_like(mean_2), atol=1e-8):
                    warnings.warn(
                        "Numerical issues were encountered when scaling the data. "
                        "The standard deviation of the data is probably very close to 0."
                    )
                    X = X - mean_2

        return X


class BaseGTM(ABC, nn.Module):
    """
    Abstract base class for Generative Topographic Mapping (GTM) implementations.

    This class defines the interface that all GTM implementations must follow.
    GTM is a probabilistic dimensionality reduction technique that creates a
    non-linear mapping from a high-dimensional data space to a low-dimensional
    latent space using a generative model.

    Attributes:
        device (torch.device): The device (CPU/GPU) on which computations are performed
    """

    def __init__(self, device: str = "cpu") -> None:
        """
        Initialize the BaseGTM.

        Args:
            device: Device to use for computations ("cpu" or "cuda")
        """
        super().__init__()
        self.device = torch.device(device)

    @abstractmethod
    def _init_weights(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Initialize the weight matrix for the GTM model.

        Returns:
            torch.Tensor: Initialized weight matrix
        """
        raise NotImplementedError()

    @abstractmethod
    def _init_beta(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Initialize the precision parameter (beta) for the GTM model.

        Returns:
            torch.Tensor: Initialized beta parameter
        """
        raise NotImplementedError()

    @abstractmethod
    def _init_grid(self, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, ...]:
        """
        Initialize the grid structure for the latent space.

        Returns:
            Tuple[torch.Tensor, ...]: Grid-related tensors (nodes, mu, basis_width, etc.)
        """
        raise NotImplementedError()

    @abstractmethod
    def _init_rbfs(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Initialize the radial basis functions (RBFs) for the GTM model.

        Returns:
            torch.Tensor: RBF matrix
        """
        raise NotImplementedError()

    @abstractmethod
    def kernel(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Compute the kernel (distance) matrix between two sets of points.

        Returns:
            torch.Tensor: Distance matrix
        """
        raise NotImplementedError()

    @abstractmethod
    def e_step(
        self, data: torch.Tensor, distances: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform the E-step of the EM algorithm.

        Args:
            data: Input data tensor
            distances: Distance matrix between data points and mixture centers

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Responsibilities and log-likelihoods
        """
        raise NotImplementedError()

    @abstractmethod
    def m_step(
        self, data: torch.Tensor, responsibilities: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform the M-step of the EM algorithm.

        Args:
            data: Input data tensor
            responsibilities: Responsibility matrix from E-step

        Returns:
            torch.Tensor: Updated distance matrix
        """
        raise NotImplementedError()

    @abstractmethod
    def fit(self, data: torch.Tensor) -> None:
        """
        Fit the GTM model to the data.

        Args:
            data: Training data tensor
        """
        raise NotImplementedError()

    @abstractmethod
    def project(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project data points to the latent space.

        Args:
            data: Data to project

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Responsibilities and log-likelihoods
        """
        raise NotImplementedError()

    @abstractmethod
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Transform data using the fitted GTM model.

        Args:
            data: Data to transform

        Returns:
            torch.Tensor: Transformed data
        """
        raise NotImplementedError()

    @abstractmethod
    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Fit the GTM model and transform the data in one step.

        Args:
            data: Training data tensor

        Returns:
            torch.Tensor: Transformed data
        """
        raise NotImplementedError()


class VanillaGTM(BaseGTM, ABC):
    """
    Vanilla implementation of Generative Topographic Mapping (GTM).

    This class implements the standard GTM algorithm as described in the original
    paper by Bishop et al. GTM is a probabilistic dimensionality reduction technique
    that creates a non-linear mapping from high-dimensional data to a low-dimensional
    latent space using a generative model with radial basis functions.

    Attributes:
        num_nodes (int): Number of nodes in the latent space grid
        num_basis_functions (int): Number of radial basis functions
        basis_width (float): Width parameter for the radial basis functions
        reg_coeff (float): Regularization coefficient for the weight matrix
        standardize (bool): Whether to standardize the input data
        max_iter (int): Maximum number of EM iterations
        tolerance (float): Convergence tolerance for the EM algorithm
        n_components (int): Number of latent space dimensions (2 or 3)
        use_cholesky (bool): Whether to use Cholesky decomposition for numerical stability
        seed (int): Random seed for reproducibility
        topology (str): Topology of the latent space grid ("square", "hexagonal", etc.)
        device (str): Device to use for computations ("cpu" or "cuda")
        nodes (torch.Tensor): Grid nodes in the latent space
        mu (torch.Tensor): Centers of the radial basis functions
        phi (torch.Tensor): RBF matrix
        weights (torch.Tensor): Weight matrix for the generative model
        beta (torch.Tensor): Precision parameter for the noise model
        data_mean (torch.Tensor): Mean of the training data
        data_std (torch.Tensor): Standard deviation of the training data
    """

    def __init__(
        self,
        num_nodes: int,
        num_basis_functions: int,
        basis_width: float,
        reg_coeff: float,
        standardize: bool = True,
        max_iter: int = 100,
        tolerance: float = 1e-3,
        n_components: int = 2,
        use_cholesky: bool = False,
        seed: int = 1234,
        topology: str = "square",
        device: str = "cpu",
    ) -> None:
        """
        Initialize the VanillaGTM model.

        Args:
            num_nodes: Number of nodes in the latent space grid. Must be a perfect
                      square for 2D or perfect cube for 3D latent space
            num_basis_functions: Number of radial basis functions. Must be a perfect
                               square for 2D or perfect cube for 3D latent space
            basis_width: Width parameter for the radial basis functions
            reg_coeff: Regularization coefficient for the weight matrix
            standardize: Whether to standardize the input data
            max_iter: Maximum number of EM iterations
            tolerance: Convergence tolerance for the EM algorithm
            n_components: Number of latent space dimensions (2 or 3)
            use_cholesky: Whether to use Cholesky decomposition for numerical stability
            seed: Random seed for reproducibility
            topology: Topology of the latent space grid ("square", "hexagonal", etc.)
            device: Device to use for computations ("cpu" or "cuda")

        Raises:
            AssertionError: If num_nodes or num_basis_functions are not perfect squares/cubes
        """
        super().__init__(device=device)
        self.to(self.device)
        torch.manual_seed(seed)
        self.n_components = n_components
        if self.n_components == 2:
            assert np.sqrt(num_nodes).is_integer(), "num_nodes must be square"
            assert np.sqrt(
                num_basis_functions
            ).is_integer(), "num_basis_functions must be square"
        elif self.n_components == 3:
            assert round(
                np.cbrt(num_nodes)
            ).is_integer(), f"{num_nodes} {type(num_nodes)} must be a cube"
            assert round(
                np.cbrt(num_basis_functions)
            ).is_integer(), "num_basis_functions must be a cube"
        self.num_nodes: int = num_nodes
        self.num_basis_functions: int = num_basis_functions
        self.basis_width: float = basis_width
        self.reg_coeff: float = reg_coeff
        self.standardize: bool = standardize
        self.max_iter: int = max_iter
        self.tolerance: float = tolerance

        self.use_cholesky: bool = use_cholesky
        self.topology: str = topology

        # Here we will add parameters of GTM
        nodes, mu, scaled_basis_width = self._init_grid()
        self.nodes: torch.Tensor = nodes.to(self.device).double()
        self.mu: torch.Tensor = mu.to(self.device).double()
        self._scaled_basis_width: float = scaled_basis_width
        # always initialised after grid
        self.phi: torch.Tensor = self._init_rbfs()

        self.data_mean: Optional[torch.Tensor] = None
        self.data_std: Optional[torch.Tensor] = None
        self.weights: Optional[torch.Tensor] = None
        self.beta: Optional[torch.Tensor] = None

    def _init_weights(
        self, data: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """
        Initialize the weight matrix for the GTM model.

        The weight matrix has dimensions (num_basis_functions + 1, data_dimensions)
        where the extra row accounts for the bias term.

        Args:
            data: Input data tensor to determine the output dimensionality
            *args: Additional positional arguments (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            torch.Tensor: Zero-initialized weight matrix
        """
        return torch.zeros(
            self.num_basis_functions + 1,
            data.shape[-1],
            dtype=torch.float64,
            device=self.device,
        )

    def _init_beta(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Initialize the precision parameter (beta) for the GTM model.

        Beta represents the inverse variance of the noise in the generative model.
        It's initialized to 1.0 as a default value.

        Args:
            *args: Additional positional arguments (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            torch.Tensor: Initialized beta parameter
        """
        return torch.tensor(1.0, dtype=torch.float64, device=self.device)

    def _rectangular_grid(self, num_points_x: int, num_points_y: int) -> torch.Tensor:
        """
        Initialize a rectangular grid structure for the latent space.

        This method generates a regular rectangular grid pattern with coordinates
        ranging from -1 to 1 in both x and y dimensions. The grid is flattened
        into a 2D tensor where each row represents a grid point.

        Args:
            num_points_x (int): Number of points along the x-axis
            num_points_y (int): Number of points along the y-axis

        Returns:
            torch.Tensor: A 2D tensor of shape (num_points_x * num_points_y, 2)
                         containing the rectangular grid coordinates, where each
                         row is [x, y] coordinates of a grid point
        """
        x_points = torch.linspace(-1, 1, steps=num_points_x)
        y_points = torch.linspace(-1, 1, steps=num_points_y)
        x, y = torch.meshgrid(x_points, y_points, indexing="ij")
        return torch.stack((x.flatten(), y.flatten()), dim=1)

    def _init_grid(self) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Initialize the grid structure for the latent space.

        This method creates the grid nodes and RBF centers for the GTM model.
        It uses rectangular grids for both the latent space nodes and the
        RBF centers, with appropriate scaling.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, float]: A tuple containing:
                - nodes: Grid nodes in the latent space
                - mu: RBF centers
                - basis_width: Scaled basis width parameter
        """
        nodes = self._rectangular_grid(
            int(np.sqrt(self.num_nodes)), int(np.sqrt(self.num_nodes))
        )
        mu = self._rectangular_grid(
            int(np.sqrt(self.num_basis_functions)),
            int(np.sqrt(self.num_basis_functions)),
        )

        mu = mu * (self.num_basis_functions / (self.num_basis_functions - 1))
        basis_width = self.basis_width * float(mu[0, 1] - mu[1, 1])
        return nodes, mu, basis_width

    def _init_rbfs(self) -> torch.Tensor:
        """
        Initialize the radial basis functions (RBFs) for the GTM model.

        This method computes the RBF matrix by calculating the distances between
        grid nodes and RBF centers, then applying a Gaussian kernel. A bias
        term (column of ones) is added to the matrix.

        Returns:
            torch.Tensor: RBF matrix of shape (num_nodes, num_basis_functions + 1)
                         where the last column contains ones for the bias term
        """

        dist_nodes_rbfs = (
            torch.cdist(
                self.nodes, self.mu, compute_mode="donot_use_mm_for_euclid_dist"
            ).to(self.device)
            ** 2
        )

        phi = torch.exp(-0.5 * dist_nodes_rbfs / self._scaled_basis_width**2)
        return torch.cat(
            (phi, torch.ones(phi.size(0), 1, dtype=torch.float64, device=self.device)),
            dim=1,
        )

    def _standardize(
        self, x: torch.Tensor, with_mean: bool = True, with_std: bool = True
    ) -> torch.Tensor:
        """
        Standardize the input data using mean and standard deviation.

        Args:
            x: Input data tensor
            with_mean: Whether to center the data by subtracting the mean
            with_std: Whether to scale the data by dividing by standard deviation

        Returns:
            torch.Tensor: Standardized data tensor
        """
        standardizer = DataStandardizer(with_mean, with_std)
        x = standardizer.fit_transform(x)
        return x

    @staticmethod
    def _log_matrix_stats(matrix: torch.Tensor, message: str = "") -> None:
        """
        Log statistics about a matrix for debugging purposes.

        Args:
            matrix: The tensor to analyze
            message: Optional message to include in the log
        """
        logging.debug(
            f"{message}: "
            f"max - {matrix.max()} "
            f"min - {matrix.min()} "
            f"mean - {matrix.mean()} "
            f"std - {matrix.std()} "
        )

    def kernel(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute the squared Euclidean distance matrix between two sets of points.

        Args:
            a: First set of points
            b: Second set of points

        Returns:
            torch.Tensor: Squared distance matrix
        """
        return (
            torch.cdist(a, b, compute_mode="donot_use_mm_for_euclid_dist").to(
                self.device
            )
            ** 2
        )

    def e_step(
        self, data: torch.Tensor, distances: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform the E-step of the EM algorithm.

        This step computes the responsibilities (posterior probabilities) of each
        data point belonging to each mixture component, and calculates the
        log-likelihood of the data under the current model.

        Args:
            data: Input data tensor of shape (num_data_points, data_dimensions)
            distances: A KxN matrix of squared distances between data points and mixture centers
                      where K is the number of latent nodes and N is the number of data points

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - responsibilities: A num_latents x num_data matrix of responsibilities computed using softmax
                - llhs: Log-likelihood of data under the Gaussian mixture
        """
        exponent_terms = -(self.beta / 2) * distances
        responsibilities = torch.softmax(exponent_terms, dim=0, dtype=torch.float64)
        resp_logsum = torch.logsumexp(exponent_terms, dim=0)

        llhs = (
            resp_logsum
            + (data.shape[-1] / 2) * torch.log(self.beta / (2.0 * torch.pi))
            - torch.log(torch.tensor(self.num_nodes, dtype=torch.float64))
        )
        return responsibilities, llhs

    def m_step(
        self, data: torch.Tensor, responsibilities: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform the M-step of the EM algorithm.

        This step updates the model parameters (weights and beta) by maximizing
        the expected log-likelihood. It solves a regularized least squares problem
        to update the weights and computes the new precision parameter beta.

        Args:
            data: Input data tensor of shape (num_data_points, data_dimensions)
            responsibilities: Responsibility matrix from the E-step

        Returns:
            torch.Tensor: Updated distance matrix between data points and mixture centers
        """
        G = torch.diag(responsibilities.sum(dim=1))
        # A is phi'* G * phi + lambda * I / beta
        A = self.phi.T @ G @ self.phi + self.reg_coeff / self.beta * torch.eye(
            self.num_basis_functions + 1, dtype=torch.double, device=self.device
        )
        B = self.phi.T @ (responsibilities @ data)
        if self.use_cholesky:
            # here we can use Cholesky decomposition for numerical stability
            L = torch.linalg.cholesky(A)
            Y = torch.linalg.solve(L, B)
            self.weights = torch.linalg.solve(L.T, Y)
        else:
            self.weights = torch.linalg.solve(A, B)
        distance = self.kernel(self.phi @ self.weights, data)
        self.beta = (data.shape[0] * data.shape[1]) / (
            responsibilities * distance
        ).sum()
        return distance

    def _fit_loop(self, data: torch.Tensor) -> None:
        """
        Main training loop for the GTM model using the EM algorithm.

        This method iteratively performs E-step and M-step until convergence
        or maximum iterations are reached. It includes progress tracking and
        convergence monitoring.

        Args:
            data: Training data tensor
        """
        # Initial llh
        llh_old = torch.tensor(0).double()

        # Initialize the distance matrix
        init_space_posit = self.phi @ self.weights  # Initial space positions (Y-matrix)
        self.init_space_posit = deepcopy(init_space_posit)
        distances = self.kernel(init_space_posit, data)
        # Calculate the distance matrix in the data space
        self._log_matrix_stats(distances, "First distances RBFs-data in N-dimensions")

        pbar = tqdm(range(self.max_iter))
        for index, _ in enumerate(pbar):
            responsibilities, llhs = self.e_step(data, distances)
            llh = torch.mean(llhs)  # normalisation by data
            llh_diff = torch.abs(llh_old - llh)

            # Logging part
            info = {
                "LLh": float(torch.round(llh, decimals=5)),
                "deltaLLh": float(torch.round(llh_diff, decimals=5)),
                "beta": float(torch.round(self.beta, decimals=5)),
            }
            logging.info(" ".join([f"{k}: {v}" for k, v in info.items()]))
            pbar.set_postfix(info)

            # Convergence check part
            if llh_diff < self.tolerance:  # Helena checks for several cycles
                break
            llh_old = llh
            if index < self.max_iter - 1:
                distances = self.m_step(data, responsibilities)

    def fit(self, x: torch.Tensor) -> None:
        """
        Fit the GTM model to the training data.

        This method initializes the model parameters and runs the EM algorithm
        to learn the optimal mapping from the latent space to the data space.

        Args:
            x: Training data tensor of shape (num_samples, num_features)
        """
        x = x.to(self.device)
        # Calculate mean and standard deviation along each column (axis 0)

        if self.standardize:
            # Scale the tensor using mean and standard deviation
            x = self._standardize(x, with_mean=True, with_std=True)

        self.data_mean = torch.mean(x, dim=0)
        self.data_std = torch.std(x, dim=0)

        # initialise weights and beta from the data
        self.weights = self._init_weights(x)
        self.weights[-1, :] = self.data_mean
        self.beta = self._init_beta()
        self._fit_loop(x)

    def project(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project data points to the latent space.

        This method maps high-dimensional data points to the latent space
        by computing their responsibilities (posterior probabilities) for
        each latent node.

        Args:
            x: Input data tensor of shape (num_samples, num_features)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - responsibilities: Responsibility matrix of shape (num_nodes, num_samples)
                - llhs: Log-likelihood values for each data point
        """
        x = x.to(self.device)
        if self.standardize:
            x = self._standardize(x, with_mean=True, with_std=True)
        distance = self.kernel(self.phi @ self.weights, x)
        responsibilities, llhs = self.e_step(x, distance)
        return responsibilities.T, llhs

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Transform data using the fitted GTM model.

        This method projects data points to the latent space and returns
        the mean of the posterior distribution (responsibilities weighted
        by latent node positions).

        Args:
            data: Input data tensor of shape (num_samples, num_features)

        Returns:
            torch.Tensor: Transformed data in the latent space
        """
        responsibilities, _ = self.project(data)
        # Return the mean of the posterior distribution
        return responsibilities @ self.nodes

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Fit the GTM model and transform the data in one step.

        Args:
            data: Training data tensor of shape (num_samples, num_features)

        Returns:
            torch.Tensor: Transformed data in the latent space
        """
        self.fit(data)
        return self.transform(data)


class GTM(VanillaGTM):
    """
    This class extends VanillaGTM with PCA-based initialization as described
    in Bishop et al.'s original paper. It uses PCA to initialize the weight
    matrix and beta parameter for better convergence.

    Attributes:
        pca_engine (str): PCA implementation to use ("sklearn" or "torch")
        pca_scale (bool): Whether to scale eigenvectors by sqrt of eigenvalues
        pca_lowrank (bool): Whether to use low-rank PCA approximation
    """

    def __init__(
        self,
        pca_engine: str = "sklearn",
        pca_scale: bool = True,
        pca_lowrank: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the GTM model.

        Args:
            pca_engine: PCA implementation to use ("sklearn" or "torch")
            pca_scale: Whether to scale eigenvectors by sqrt of eigenvalues
            pca_lowrank: Whether to use low-rank PCA approximation
            *args: Additional positional arguments passed to VanillaGTM
            **kwargs: Additional keyword arguments passed to VanillaGTM
        """
        self.pca_engine: str = pca_engine
        self.pca_scale: bool = pca_scale
        self.pca_lowrank: bool = pca_lowrank
        super(GTM, self).__init__(*args, **kwargs)

    def _init_beta_mixture_components(self) -> torch.Tensor:
        """
        Initialize beta parameter based on mixture component distances.

        This method computes the initial beta value by analyzing the distances
        between mixture components in the latent space. It finds the average
        nearest neighbor distance and uses half of this value as the initial beta.

        Returns:
            torch.Tensor: Initial beta parameter based on mixture component distances
        """
        y = self.phi @ self.weights
        lat_space_dist = (
            torch.cdist(y, y, compute_mode="donot_use_mm_for_euclid_dist").to(
                self.device
            )
            ** 2
        )
        self._log_matrix_stats(
            lat_space_dist, "Distances between nodes in N-dimensions"
        )

        # Add a large number to the diagonal (similar to realmax in MATLAB)
        lat_space_dist.fill_diagonal_(torch.finfo(lat_space_dist.dtype).max)

        # Find the average distance between nearest neighbors
        mean_nn = torch.min(lat_space_dist, dim=1).values.mean()

        # Calculate options for the initial beta
        beta = mean_nn / 2
        return beta

    def _pca_torch(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform PCA using PyTorch's implementation.

        This method computes the principal components using either full SVD
        or low-rank approximation, with optional scaling of eigenvectors.

        Args:
            data: Input data tensor

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - eigenvectors: Principal component vectors
                - eigenvalues: Corresponding eigenvalues
        """
        if self.standardize:
            # Assuming data is already standardized or you want to standardize it here
            pca_data = data
        else:
            pca_data = data - data.mean(dim=0)

        if self.pca_lowrank:
            _, singular_values, eigenvectors = torch.pca_lowrank(
                pca_data, q=20, center=False
            )
            eigenvectors = eigenvectors[:, : self.n_components + 1].T
        else:
            _, singular_values, eigenvectors = torch.linalg.svd(
                pca_data, full_matrices=False
            )
            eigenvectors = eigenvectors[: self.n_components + 1, :]

        eigenvalues = singular_values[: self.n_components + 1].reshape(-1, 1) ** 2 / (
            data.shape[0] - 1
        )

        if self.pca_scale:
            # Scale by sqrt(eigenvalues)
            eigenvectors = eigenvectors * torch.sqrt(eigenvalues)
        return eigenvectors, eigenvalues

    def _pca_sklearn(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform PCA using scikit-learn's implementation.

        This method computes the principal components using scikit-learn's PCA,
        with optional scaling of eigenvectors by the square root of eigenvalues.

        Args:
            data: Input data tensor

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - eigenvectors: Principal component vectors
                - eigenvalues: Corresponding eigenvalues
        """
        if self.standardize:
            # If you need standardization, do it here, else just use data as is.
            pca_data = data
        else:
            pca_data = data - data.mean(dim=0)

        pca = PCA(n_components=self.n_components + 1)
        pca.fit(pca_data.cpu().numpy())

        eigenvectors = pca.components_
        eigenvalues = pca.explained_variance_

        if self.pca_scale:
            # The provided code scales eigenvectors by sqrt of singular values
            # Torch code scales eigenvectors by sqrt of eigenvalues.
            # Here we use the given scaling for sklearn (as stated in the code):
            eigenvectors = eigenvectors * np.sqrt(
                eigenvalues[:, np.newaxis]
            )  # np.sqrt(pca.singular_values_[:, np.newaxis])

        eigenvectors = torch.from_numpy(eigenvectors).double()
        eigenvalues = torch.from_numpy(eigenvalues).double()

        return eigenvectors, eigenvalues

    def _get_pca(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get PCA results using the specified engine.

        This method dispatches to the appropriate PCA implementation based on
        the pca_engine setting. It supports sklearn, torch, or custom dictionaries.

        Args:
            data: Input data tensor

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - eigenvectors: Principal component vectors
                - eigenvalues: Corresponding eigenvalues

        Raises:
            ValueError: If pca_engine is not recognized
        """
        if self.pca_engine == "sklearn":
            eigenvectors, eigenvalues = self._pca_sklearn(data)
        elif self.pca_engine == "torch":
            eigenvectors, eigenvalues = self._pca_torch(data)
        elif isinstance(self.pca_engine, dict):
            eigenvectors, eigenvalues = (
                self.pca_engine["eigenvectors"],
                self.pca_engine["eigenvalues"],
            )
        else:
            raise ValueError(f"Unknown pca_engine: {self.pca_engine}")
        eigenvectors = eigenvectors.to(data.dtype).to(self.device)
        eigenvalues = eigenvalues.to(data.dtype).to(self.device)
        return eigenvectors, eigenvalues

    def _init_weights(self, eigenvectors: torch.Tensor) -> torch.Tensor:
        """
        Initialize weights using PCA-based projection.

        This method initializes the weight matrix by projecting the latent space
        nodes onto the principal components and solving a least squares problem.

        Args:
            eigenvectors: PCA eigenvectors from the data

        Returns:
            torch.Tensor: Initialized weight matrix

        Note:
            Can't be run on GPU. see https://github.com/pytorch/pytorch/issues/71222
            TODO: check if shapes are good
        """
        ## Can't be run on GPU. see https://github.com/pytorch/pytorch/issues/71222
        self.nodes = (self.nodes - self.nodes.mean(dim=0)) / (self.nodes.std(dim=0))
        nodes_pca_projection = (
            self.nodes @ eigenvectors[: self.n_components]
        )  # TODO: check if shapes are good
        return torch.linalg.lstsq(
            self.phi, nodes_pca_projection, driver="gels"
        ).solution  # .cuda()

    def _init_beta(self, eigenvalues: torch.Tensor) -> torch.Tensor:
        """
        Initialize beta parameter using PCA eigenvalues.

        This method computes the initial beta value by taking the maximum of
        two estimates: one from mixture component distances and one from
        PCA eigenvalues.

        Args:
            eigenvalues: PCA eigenvalues from the data

        Returns:
            torch.Tensor: Initial beta parameter
        """
        # Calculating the initial beta
        beta_1 = self._init_beta_mixture_components()
        beta_2 = eigenvalues[self.n_components]

        logging.debug(f"Beta from distances: {beta_1}")
        logging.debug(f"Beta from PCA: {beta_2}")
        return max(beta_1, beta_2)

    def fit(self, x: torch.Tensor) -> None:
        """
        Fit the GTM model to the training data.

        This method initializes the model parameters using PCA-based initialization
        and runs the EM algorithm to learn the optimal mapping from the latent
        space to the data space.

        Args:
            x: Training data tensor of shape (num_samples, num_features)
        """
        x = x.to(self.device)

        if self.standardize:
            # Calculate mean and standard deviation along each column (axis 0)
            # Scale the tensor using mean and standard deviation
            x = self._standardize(x, with_mean=True, with_std=True)
        self.data_mean = torch.mean(x, dim=0)
        self.data_std = torch.std(x, dim=0)
        # initialise weights and beta from the data
        eigenvectors, eigenvalues = self._get_pca(x)
        self.weights = self._init_weights(eigenvectors)
        self.weights[-1, :] = self.data_mean
        self.beta = self._init_beta(eigenvalues)
        self.beta_init = deepcopy(self.beta)
        self._fit_loop(x)
