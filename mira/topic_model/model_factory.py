from mira.topic_model.modality_mixins.accessibility_model import AccessibilityModel
from mira.topic_model.modality_mixins.expression_model import ExpressionModel

from mira.topic_model.CODAL.covariate_model import CovariateModel
from mira.topic_model.generative_models.dirichlet_process import \
        ExpressionDirichletProcessModel, AccessibilityDirichletProcessModel
from mira.topic_model.generative_models.lda_generative import \
        ExpressionDirichletModel, AccessibilityDirichletModel

from mira.topic_model.base import BaseModel, logger
from mira.topic_model.base import TopicModel as mira_topic_model
import numpy as np



def TopicModel(
    n_samples, n_features,*,
    feature_type,
    highly_variable_key = None,
    exogenous_key = None,
    endogenous_key = None,
    counts_layer = None,
    categorical_covariates = None,
    continuous_covariates = None,
    covariates_keys = None,
    extra_features_keys = None,
    latent_space = 'dirichlet',
    **model_parameters,
):
    '''
    Instantiates a topic model, which learns regulatory "topics" from 
    single-cell RNA-seq or ATAC-seq data. Topics capture 
    patterns of covariance between gene or cis-regulatory elements. 
    Each cell is represented by a composition over topics, and each 
    topic corresponds with activations of co-regulated elements.

    You may use enrichment analysis of topics to understand signaling 
    and transcription factor drivers of cell states, and embedding of 
    cell-topic distributions to visualize and cluster cells, 
    and to perform pseudotime trajectory inference.

    When working with batched data, the parameters of the topic model are
    optimized using the novel CODAL (COvariate Disentangling Augmented Loss) objective,
    which shows State of the Art performance for detection of batch confounded cell types. 

    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset, used to choose hyperparameters for the model
    n_features : int
        Number of features in the dataset, used to choose hyperparamters for the model
    feature_type : {'expression','accessibilty'}
        Modality of the data being modeled.
    highly_variable_key : str, default = None
        Column in AnnData that marks features to be modeled. 
        These features should include all elements used for enrichment analysis of topics. 
        For expression data, this should be highly variable genes releveant to your system
        (the top ~4000 appears to work well). For accessibility data, all called peaks may be used.
    exogenous_key : str, default=None
        Same as `highly_variable_key`, included for backwards compatibility.
    endogenous_key : str, default=None
        Column in AnnData that marks features to be used for encoder neural network. 
        These features should prioritize elements that distinguish 
        between populations, like highly-variable genes. If "None", then the model will
        use the features supplied to "exogenous_key".
    counts_layer : str, default=None
        Layer in AnnData that countains raw counts for modeling.
    categorical_covariates : str, list[str], np.ndarray[str], or None, default=None
        Categorical covariates in the dataset. For example, batch of origin, donor, 
        assay chemistry, sequencing machine, etc.
    continuous_covariates : str, list[str], np.ndarray[str], or None
        Continuous covariates in the dataset. For example, FRIP score (ATAC-seq), 
        percent reads mitochrondria (RNA-seq), or other QC metrics.
    extra_features_keys : str, list[str], np.ndarray[str], or None
        Columns in anndata.obs which contain extra features for the encoder neural network.
    latent_space : "dirichlet" or "dirichlet-process"
        Which prior to use for the topic model latent space. "dirichlet" assumes the same
        latent space as the LDA algorithm. The dimensionality of the latent space must be 
        tuned. The "dirichlet_process" latent space automatically infers the number of topics
        during training, but requires large datasets to work well (>40000 cells).
    
    **Model Parameters**
    
    cost_beta : float>0, default = 1.
        Multiplier of the regularization loss terms (KL divergence and mutual information 
        regularization) versus the reconstruction loss term. Smaller datasets (<10K cells )
        sometimes require larger **cost_beta** (1.25 -2.), while larger datasets (>10K cells) 
        always work well with cost_beta=1. This parameter is automatically set to a reasonable
        value based on the size of the dataset provided to this function. 
    num_topics : int, default=16
        Number of topics to learn from data.
    hidden : int, default=128
        Number of nodes to use in hidden layers of encoder network
    num_layers: int, default=3
        Number of layers to use in encoder network, including output layer
    num_epochs: int, default=40
        Number of epochs to train topic model. The One-cycle learning rate policy
        requires a pre-defined training length, and 40 epochs is 
        usually an overestimate of the optimal number of epochs to train for.
    decoder_dropout : float (0., 1.), default=0.2
        Dropout rate for the decoder network. Prevents node collapse.
    encoder_dropout : float (0., 1.), default=0.2
        Dropout rate for the encoder network. Prevents overfitting.
    use_cuda : boolean, default=True
        Try using CUDA GPU speedup while training.
    seed : int, default=None
        Random seed for weight initialization. 
        Enables reproduceable initialization of model.
    min_learning_rate : float, default=1e-6
        Start learning rate for One-cycle learning rate policy.
    max_learning_rate : float, default=1e-1
        Peak learning rate for One-cycle policy. 
    batch_size : int, default=64
        Minibatch size for stochastic gradient descent while training. 
        Larger batch sizes train faster, but may produce less optimal models.
    initial_pseudocounts : int, default=50
        Initial pseudocounts allocated to approximated hierarchical dirichlet prior.
        More pseudocounts produces smoother topics, 
        less pseudocounts produces sparser topics. 
    nb_parameterize_logspace : boolean, default=True
        Parameterize negative-binomial distribution using log-space probability 
        estimates of gene expression. Is more numerically stable.
    embedding_size : int > 0 or None, default=None
        Number of nodes in first encoder neural network layer. Default of *None*
        gives an embedding size of `hidden`.
    kl_strategy : {'monotonic','cyclic'}, default='cyclic'
        Whether to anneal KL term using monotonic or cyclic strategies. Cyclic
        may produce slightly better models.

    **CODAL models only**

    dependence_lr : float>0, default=1e-4
        Learning rate for tuning the mutual information estimator
    dependence_hidden : int>0, default=64
        Hidden size of mutual information estimator
    weight_decay : float>0, default=0.001
        Weight decay of topic model weight optimizer
    min_momentum : float>0, default=0.85
        Min momentum for 1-cycle learning rate policy
    max_momentum : float>0, default=0.95
        Max momentum for 1-cycle learning rate policy
    covariates_hidden : int>0, default=32
        Number of nodes for single layer of technical effect network
    covariates_dropout : float>0, default=0.05
        Dropout applied to the technical effect network.
    mask_dropout : float>0, default=0.05
        Bernoulli coruption rate of technical effect predictions during training.
    marginal_estimation_size : int>0, default=256
        Number of pairings used to estimate mutual information at each step.
    dependence_beta : float>0, default=1.
        The weight of the mutual information cost at each step is `cost_beta`*`dependence_beta`.
        Changing this value to more than 1 weights mutual information regularization more highly
        than KL-divergence regularization of the loss. 

    **Accessibility models only**

    embedding_dropout : float>0, default=0.05
        Bernoulli corruption of bag of peaks input to DAN encoder.
    skipconnection_atac_encoder : boolean, default=True
        Whether to add a skip connection to the ATAC-seq encoder model. This improves performance
        in CODAL models.    

    Returns
    -------

    topic model : 
        A CODAL (if there are technical covariates in the dataset) or MIRA topic model.
        Hyperparameters of the topic model are chosen based on the supplied dataset
        properties. 

    Attributes
    ----------
    features : np.ndarray[str]
        Array of exogenous feature names, all features used in learning topics
    highly_variable : np.ndarray[boolean]
        Boolean array marking which features were 
        "highly_variable"/endogenous, used to train encoder
    encoder : torch.nn.Sequential
        Encoder neural network
    decoder : torch.nn.Sequential
        Decoder neural network
    num_exog_features : int
        Number of exogenous features to predict using decoder network
    num_endog_features : int
        Number of endogenous feature used for encoder network
    device : torch.device
        Device on which model is allocated
    enrichments : dict
        Results from enrichment analysis of topics. For expression topic model,
        this gives geneset enrichments from Enrichr. For accessibility topic
        model, this gives motif enrichments.
    topic_cols : list
        The names of the columns for the topics added by the
        `predict` method to an anndata object. Useful for quickly accessing
        topic columns for plotting.

    Examples
    --------

    ..code-block :: python

        >>> model = mira.topics.TopicModel(
            ...    *rna_data.shape,
            ...    feature_type = 'expression',
            ...    highly_variable = 'highly_variable', 
            ...    counts_layer = 'rawcounts',
            ...    categorical_covariates = ['batch','donor'],
            ...    continuous_covariates = ['FRIP']
            ... )
    '''

    assert(latent_space in ['dp', 'dirichlet'])
    assert(feature_type in ['expression','accessibility'])

    if latent_space == 'dp':
        if n_samples < 40000:
            logger.warn(
                'The dirichlet process model is intended for atlas-level experiments.\n'
                'For smaller datasets, please use the "dirichlet" latent space, and use the tuner'
                'to find the optimal number of topics.'
            )
            
        latent_space = 'dirichlet-process'

    basename = 'model'
    if not all([c is None for c in [categorical_covariates, continuous_covariates, covariates_keys]]):
        basename = 'covariate-model'
        baseclass = CovariateModel
    else:
        baseclass = BaseModel

    if feature_type == 'expression':
        feature_model = ExpressionModel
    elif feature_type == 'accessibility':
        feature_model = AccessibilityModel

    generative_map = {
        ('expression','dirichlet') : ExpressionDirichletModel,
        ('expression','dirichlet-process') : ExpressionDirichletProcessModel,
        ('accessibility', 'dirichlet') : AccessibilityDirichletModel,
        ('accessibility', 'dirichlet-process') : AccessibilityDirichletProcessModel,
    }

    generative_model = generative_map[(feature_type, latent_space)]

    _class = type(
        '_'.join([latent_space, feature_type, basename]),
        (generative_model, feature_model, baseclass, mira_topic_model),
        {}
    )

    def none_or_1d(x):
        if x is None:
            return None
        else:
            return list(np.atleast_1d(x))

    assert not (highly_variable_key is not None and exogenous_key is not None), \
        'Only one of `highly_variable_key` or `exogenous_key` can be supplied at one time'

    instance = _class(
        exogenous_key = highly_variable_key or exogenous_key,
        endogenous_key = endogenous_key,
        counts_layer = counts_layer,
        categorical_covariates = none_or_1d(categorical_covariates),
        continuous_covariates = none_or_1d(continuous_covariates),
        covariates_keys = none_or_1d(covariates_keys),
        extra_features_keys = none_or_1d(extra_features_keys),
    )

    parameter_recommendations = \
        instance.recommend_parameters(n_samples, n_features)
    
    parameter_recommendations.update(model_parameters)
    instance.set_params(**parameter_recommendations)

    return instance