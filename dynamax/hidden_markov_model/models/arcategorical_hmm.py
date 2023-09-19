from typing import NamedTuple, Optional, Tuple, Union

import jax.numpy as jnp
import jax.random as jr
from jax import lax, tree_map, vmap
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax.nn import one_hot
from jaxtyping import Array, Float

from dynamax.hidden_markov_model.models.abstractions import HMM, HMMEmissions, HMMParameterSet
from dynamax.hidden_markov_model.models.initial import ParamsStandardHMMInitialState
from dynamax.hidden_markov_model.models.initial import StandardHMMInitialState
from dynamax.hidden_markov_model.models.transitions import ParamsStandardHMMTransitions
from dynamax.hidden_markov_model.models.transitions import StandardHMMTransitions
from dynamax.parameters import ParameterProperties, ParameterSet, PropertySet
from dynamax.types import Scalar
from dynamax.utils.utils import pytree_sum


class ParamsLinearAutoregressiveCategoricalHMMEmissions(NamedTuple):
    probs: Union[Float[Array, "state_dim num_classes num_classes"], ParameterProperties]


class ParamsLinearAutoregressiveCategoricalHMM(NamedTuple):
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsLinearAutoregressiveCategoricalHMMEmissions


class LinearAutoregressiveCategoricalHMMEmissions(HMMEmissions):

    def __init__(self,
                 num_states,
                 emission_dim,
                 num_classes,
                 num_lags=1,
                 emission_prior_concentration=1.1):
        """_summary_

        Args:
            emission_probs (_type_): _description_
        """
        self.num_states = num_states
        self.emission_dim = emission_dim
        self.num_classes = num_classes
        self.num_lags = num_lags
        self.emission_prior_concentration = emission_prior_concentration  * jnp.ones(num_classes)

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    def distribution(self, params, state, inputs=None):
        return tfd.Independent(
            tfd.Categorical(probs=params.probs[state, jnp.int32(inputs)]),
            reinterpreted_batch_ndims=1)

    def log_prior(self, params):
        return tfd.Dirichlet(self.emission_prior_concentration).log_prob(params.probs).sum()

    def initialize(self, key=jr.PRNGKey(0), method="prior", emission_probs=None):
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.
        Parameters will then be sampled from the prior (if `method==prior`).

        Note: in the future we may support more initialization schemes, like K-Means.

        Args:
            key (PRNGKey, optional): random number generator for unspecified parameters. Must not be None if there are any unspecified parameters. Defaults to jr.PRNGKey(0).
            method (str, optional): method for initializing unspecified parameters. Currently, only "prior" is allowed. Defaults to "prior".
            emission_probs (array, optional): manually specified emission probabilities. Defaults to None.

        Returns:
            params: nested dataclasses of arrays containing model parameters.
            props: a nested dictionary of ParameterProperties to specify parameter constraints and whether or not they should be trained.
        """
        # Initialize the emission probabilities
        if emission_probs is None:
            if method.lower() == "prior":
                prior = tfd.Dirichlet(self.emission_prior_concentration)
                emission_probs = prior.sample(seed=key, sample_shape=(self.num_states, self.num_classes))
            elif method.lower() == "kmeans":
                raise NotImplementedError("kmeans initialization is not yet implemented!")
            else:
                raise Exception("invalid initialization method: {}".format(method))
        else:
            assert emission_probs.shape == (self.num_states, self.num_classes, self.num_classes)
            assert jnp.all(emission_probs >= 0)
            assert jnp.allclose(emission_probs.sum(axis=2), 1.0)

        # Add parameters to the dictionary
        params = ParamsLinearAutoregressiveCategoricalHMMEmissions(probs=emission_probs)
        props = ParamsLinearAutoregressiveCategoricalHMMEmissions(probs=ParameterProperties(constrainer=tfb.SoftmaxCentered()))
        return params, props

    def collect_suff_stats(self, params, posterior, emissions, inputs=None):
        expected_states = posterior.smoothed_probs
        x = one_hot(emissions[:, 0], self.num_classes)
        s = one_hot(inputs[:, 0], self.num_classes)

        return dict(sum_x=jnp.einsum("tk,ts,td->ksd", expected_states, s, x))

    def initialize_m_step_state(self, params, props):
        return None

    def m_step(self, params, props, batch_stats, m_step_state):
        if props.probs.trainable:
            emission_stats = pytree_sum(batch_stats, axis=0)
            probs = tfd.Dirichlet(self.emission_prior_concentration + emission_stats['sum_x']).mode()
            print("probs", probs)
            params = params._replace(probs=probs)
        return params, m_step_state


class LinearAutoregressiveCategoricalHMM(HMM):
    r"""An HMM with conditionally independent categorical emissions.

    Let $y_t \in \{1,\ldots,C\}^N$ denote a vector of $N$ conditionally independent
    categorical emissions from $C$ classes at time $t$. In this model,the emission
    distribution is,

    $$p(y_t \mid z_t, \theta) = \prod_{n=1}^N \mathrm{Cat}(y_{tn} \mid \theta_{z_t,n})$$
    $$p(\theta) = \prod_{k=1}^K \prod_{n=1}^N \mathrm{Dir}(\theta_{k,n}; \gamma 1_C)$$

    with $\theta_{k,n} \in \Delta_C$ for $k=1,\ldots,K$ and $n=1,\ldots,N$ are the
    *emission probabilities* and $\gamma$ is their prior concentration.

    :param num_states: number of discrete states $K$
    :param emission_dim: number of conditionally independent emissions $N$
    :param num_classes: number of multinomial classes $C$
    :param initial_probs_concentration: $\alpha$
    :param transition_matrix_concentration: $\beta$
    :param transition_matrix_stickiness: optional hyperparameter to boost the concentration on the diagonal of the transition matrix.
    :param emission_prior_concentration: $\gamma$

    """
    def __init__(self, num_states: int,
                 emission_dim: int,
                 num_classes: int,
                 num_lags: int=1,
                 initial_probs_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_stickiness: Scalar=0.0,
                 emission_prior_concentration=1.1):
        self.emission_dim = emission_dim
        self.num_lags = num_lags
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(num_states, concentration=transition_matrix_concentration, stickiness=transition_matrix_stickiness)
        emission_component = LinearAutoregressiveCategoricalHMMEmissions(num_states, emission_dim, num_classes, emission_prior_concentration=emission_prior_concentration)
        super().__init__(num_states, initial_component, transition_component, emission_component)

    def initialize(self,
                   key: jr.PRNGKey=jr.PRNGKey(0),
                   method: str="prior",
                   initial_probs: Optional[Float[Array, "num_states"]]=None,
                   transition_matrix: Optional[Float[Array, "num_states num_states"]]=None,
                   emission_probs: Optional[Float[Array, "num_states emission_dim num_classes"]]=None
    ) -> Tuple[ParameterSet, PropertySet]:
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.
        Parameters will then be sampled from the prior (if `method==prior`).

        Note: in the future we may support more initialization schemes, like K-Means.

        Args:
            key (PRNGKey, optional): random number generator for unspecified parameters. Must not be None if there are any unspecified parameters. Defaults to None.
            method (str, optional): method for initializing unspecified parameters. Currently, only "prior" is allowed. Defaults to "prior".
            initial_probs (array, optional): manually specified initial state probabilities. Defaults to None.
            transition_matrix (array, optional): manually specified transition matrix. Defaults to None.
            emission_probs (array, optional): manually specified emission probabilities. Defaults to None.

        Returns:
            Model parameters and their properties.
        """
        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_probs=emission_probs)
        return ParamsLinearAutoregressiveCategoricalHMM(**params), ParamsLinearAutoregressiveCategoricalHMM(**props)
    
    def sample(self,
               params: HMMParameterSet,
               key: jr.PRNGKey,
               num_timesteps: int,
               prev_emissions: Optional[Float[Array, "num_lags emission_dim"]]=None,
    ) -> Tuple[Float[Array, "num_timesteps state_dim"], Float[Array, "num_timesteps emission_dim"]]:
        r"""Sample states $z_{1:T}$ and emissions $y_{1:T}$ given parameters $\theta$.

        Args:
            params: model parameters $\theta$
            key: random number generator
            num_timesteps: number of timesteps $T$
            prev_emissions: (optionally) preceding emissions $y_{-L+1:0}$. Defaults to zeros.

        Returns:
            latent states and emissions

        """
        if prev_emissions is None:
            # Default to zeros
            prev_emissions = jnp.zeros((self.num_lags, self.emission_dim))

        def _step(carry, key):
            prev_state, prev_emissions = carry
            key1, key2 = jr.split(key, 2)
            state = self.transition_distribution(params, prev_state).sample(seed=key2)
            emission = self.emission_distribution(params, state, inputs=jnp.ravel(prev_emissions)).sample(seed=key1)
            next_prev_emissions = jnp.vstack([emission, prev_emissions[:-1]])
            return (state, next_prev_emissions), (state, emission)

        # Sample the initial state
        key1, key2, key = jr.split(key, 3)
        initial_state = self.initial_distribution(params).sample(seed=key1)
        initial_emission = self.emission_distribution(params, initial_state, inputs=jnp.ravel(prev_emissions)).sample(seed=key2)
        initial_prev_emissions = jnp.vstack([initial_emission, prev_emissions[:-1]])

        # Sample the remaining emissions and states
        next_keys = jr.split(key, num_timesteps - 1)
        _, (next_states, next_emissions) = lax.scan(
            _step, (initial_state, initial_prev_emissions), next_keys)

        # Concatenate the initial state and emission with the following ones
        expand_and_cat = lambda x0, x1T: jnp.concatenate((jnp.expand_dims(x0, 0), x1T))
        states = tree_map(expand_and_cat, initial_state, next_states)
        emissions = tree_map(expand_and_cat, initial_emission, next_emissions)
        return states, emissions

    @property
    def inputs_shape(self):
        """Return a pytree matching the pytree of tuples specifying the shape(s)
        of a single time step's inputs.
        """
        return (self.num_lags * self.emission_dim,)
    
    def compute_inputs(self,
                    emissions: Float[Array, "num_batch num_timesteps emission_dim"],
                    prev_emissions: Optional[Float[Array, "num_batch num_lags emission_dim"]] = None
    ) -> Float[Array, "num_batch num_timesteps emission_dim_times_num_lags"]:
        r"""
        Helper function to compute the matrix of lagged emissions.

        Args:
            emissions: $(B \times T \times N)$ array of emissions
            prev_emissions: $(B \times L \times N)$ array of previous emissions. Defaults to zeros.

        Returns:
            $(B \times T \times N \cdot L)$ array of lagged emissions. These are the inputs to the fitting functions.
        """
        if prev_emissions is None:
            # Default to zeros
            prev_emissions = jnp.zeros((emissions.shape[0], self.num_lags, self.emission_dim))

        padded_emissions = jnp.hstack((prev_emissions, emissions))
        num_timesteps = emissions.shape[1]
        return jnp.column_stack([padded_emissions[:, lag:lag+num_timesteps]
                                 for lag in reversed(range(self.num_lags))])
    
    def compute_inputs_old(self,
                       emissions: Float[Array, "num_timesteps emission_dim"],
                       prev_emissions: Optional[Float[Array, "num_lags emission_dim"]]=None
    ) -> Float[Array, "num_timesteps emission_dim_times_num_lags"]:
        r"""Helper function to compute the matrix of lagged emissions.

        Args:
            emissions: $(T \times N)$ array of emissions
            prev_emissions: $(L \times N)$ array of previous emissions. Defaults to zeros.

        Returns:
            $(T \times N \cdot L)$ array of lagged emissions. These are the inputs to the fitting functions.
        """
        if prev_emissions is None:
            # Default to zeros
            prev_emissions = jnp.zeros((self.num_lags, self.emission_dim))

        padded_emissions = jnp.vstack((prev_emissions, emissions))
        num_timesteps = len(emissions)
        return jnp.column_stack([padded_emissions[lag:lag+num_timesteps]
                                 for lag in reversed(range(self.num_lags))])
    
    def m_step(self, params, props, batch_stats, m_step_state):
        batch_initial_stats, batch_transition_stats, batch_emission_stats = batch_stats
        initial_m_step_state, transitions_m_step_state, emissions_m_step_state = m_step_state

        initial_params, initial_m_step_state = self.initial_component.m_step(params.initial, props.initial, batch_initial_stats, initial_m_step_state)
        transition_params, transitions_m_step_state = self.transition_component.m_step(params.transitions, props.transitions, batch_transition_stats, transitions_m_step_state)
        emission_params, emissions_m_step_state = self.emission_component.m_step(params.emissions, props.emissions, batch_emission_stats, emissions_m_step_state)
        params = params._replace(initial=initial_params, transitions=transition_params, emissions=emission_params)
        m_step_state = initial_m_step_state, transitions_m_step_state, emissions_m_step_state
        return params, m_step_state
