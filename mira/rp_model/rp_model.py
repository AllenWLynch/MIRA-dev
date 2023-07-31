import numpy as np
import logging
from mira.adata_interface.rp_model import wraps_rp_func, add_isd_results, \
    add_predictions, fetch_TSS_from_adata
from mira.adata_interface.core import add_layer
import h5py as h5
import os
import glob
from ._gene_model import GeneModel
from joblib import Parallel, delayed
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def _generate_features(model, expr_adata, atac_adata):
    pass


def _parallel_apply(func, n_jobs = 1, bar_desc = '',*, gene_models, expr_adata, atac_adata):

    return Parallel(n_jobs=n_jobs, verbose=0, pre_dispatch='2*n_jobs', max_nbytes = None)\
                ( delayed(func)(model, features) 
                  for model, features in tqdm(
                        map(lambda model : _generate_features(model, expr_adata, atac_adata), gene_models), 
                        desc = bar_desc, total = gene_models
                    )
                )


class BaseModel:

    @classmethod
    def load_dir(cls,counts_layer = None,*,expr_model, accessibility_model, prefix):
        '''
        Load directory of RP models. Adds all available RP models into a container.

        Parameters
        ----------
        expr_model: mira.topics.ExpressionTopicModel
            Trained MIRA expression topic model.
        accessibility_model : mira.topics.AccessibilityTopicModel
            Trained MIRA accessibility topic model.
        counts_layer : str, default=None
            Layer in AnnData that countains raw counts for modeling.
        prefix : str
            Prefix under which RP models were saved.

        Examples
        --------

        .. code-block :: python

            >>> litemodel = mira.rp.LITE_Model.load_dir(
            ...     counts_layer = 'counts',
            ...     expr_model = rna_model, 
            ...     accessibility_model = atac_model,
            ...     prefix = 'path/to/rpmodels/'
            ... )

        '''

        paths = glob.glob(prefix + '*.pth')

        genes = [os.path.basename(x.split('_')[-1].split('.')[0]) 
                for x in paths]

        model = cls(expr_model = expr_model, 
                    seed = 0,
                    accessibility_model = accessibility_model,
                    counts_layer = counts_layer, 
                    genes = genes).load(prefix)

        return model


    def __init__(self,seed = 0,*,
        expr_model, 
        accessibility_model, 
        genes,
        counts_layer = None,
        n_jobs = 1):
        '''
        Parameters
        ----------

        expr_model: mira.topics.ExpressionTopicModel
            Trained MIRA expression topic model.
        accessibility_model : mira.topics.AccessibilityTopicModel
            Trained MIRA accessibility topic model.
        genes : np.ndarray[str], list[str]
            List of genes for which to learn RP models.
        learning_rate : float>0
            Learning rate for L-BGFS optimizer.
        counts_layer : str, default=None
            Layer in AnnData that countains raw counts for modeling.
        initialization_model : mira.rp.LITE_Model, mira.rp.NITE_Model, None
            Initialize parameters of RP model using the provided model before
            further optimization with L-BGFS. This is used when training the NITE
            model, which is initialized with the LITE model parameters learned 
            for the same genes, then retrained to optimized the NITE model's 
            extra parameters. This procedure speeds training.

        Attributes
        ----------
        genes : np.ndarray[str]
            Array of gene names for models
        features : np.ndarray[str]
            Array of gene names for models
        models : list[mira.rp.GeneModel]
            List of trained RP models
        model_type : {"NITE", "LITE"}
        
        Examples
        --------

        Setup requires RNA and ATAC AnnData objects with shared cell barcodes
        and trained topic models for both modes:

        .. code-block:: python
            
            >>> rp_args = dict(expr_adata = rna_data, atac_adata = atac_data)

        '''

        self.seed = seed
        self.n_jobs = n_jobs,
        self.expr_model = expr_model
        self.accessibility_model = accessibility_model
        self.counts_layer = counts_layer

        self.models = []
        for gene in genes:

            self.models.append(
                GeneModel(gene)
            )


    def subset(self, genes):
        '''
        Return a subset container of RP models.

        Parameters
        ----------

        genes : np.ndarray[str], list[str]
            List of genes to subset from RP model

        Examples
        --------

        .. code-block :: python

            >>> less_models = litemodel.subset(['LEF1','WNT3'])

        
        '''
        assert(isinstance(genes, (list, np.ndarray)))
        for gene in genes:
            if not gene in self.genes:
                raise ValueError('Gene {} is not in RP model'.format(str(gene)))        


        return self.__class__(
            expr_model = self.expr_model,
            accessibility_model = self.accessibility_model, 
            counts_layer=self.counts_layer, 
            models = [model for model in self.models if model.gene in genes]
        )
    
    def join(self, rp_model):
        '''
        Merge RP models from two model containers.

        Parameters
        ----------

        rp_model : mira.rp.LITE_Model, mira.rp.NITE_Model
            RP model container from which to append new RP models

        Examples
        --------

        .. code-block :: python

            >>> model1.genes
            ... ['LEF1','WNT3']
            >>> model2.genes
            ... ['CTSC','EDAR']
            >>> merged_model = model1.join(model2)
            >>> merged_model.genes
            ... ['LEF1','WNT3','CTSC','EDAR']

        '''

        assert(isinstance(rp_model, BaseModel))

        add_models = np.setdiff1d(rp_model.genes, self.genes)

        for add_gene in add_models:
            self.models.append(
                rp_model.get_model(add_gene)
            )
        
        return self

    def __getitem__(self, gene):
        '''
        Alias for `get_model(gene)`.

        Examples
        --------

        >>> rp_model["LEF1"]
        ... <mira.rp_model.rp_model.GeneModel at 0x7fa07af1cf10>

        '''
        return self.get_model(gene)

    @property
    def genes(self):
        return np.array([model.gene for model in self.models])

    @property
    def features(self):
        return self.genes

    def _get_masks(self, tss_distance):

        upstream_mask = tss_distance > 0
        downstream_mask = tss_distance >= 0

        return upstream_mask, downstream_mask

    @staticmethod
    def bn(x, mu, var, eps):
            return (x - mu)/np.sqrt( var + eps)

    def _get_region_weights(self, NITE_features, softmax_denom, idx):
        
        model = self.accessibility_model

        rate = model._get_gamma()[idx] * self.bn(
            NITE_features.dot(model._get_beta()[:, idx]),
            model._get_bn_mean()[idx],
            model._get_bn_var()[idx],
            model.decoder.bn.eps
        ) + model._get_bias()[idx]

        region_probabilities = np.exp(rate)/softmax_denom[:, np.newaxis]
        return region_probabilities.cpu().numpy()

    def save(self, prefix):
        '''
        Save RP models.

        Parameters
        ----------

        prefix : str
            Prefix under which to save RP models. May be filename prefix
            or directory. RP models will save with format:
            **{prefix}_{LITE/NITE}_{gene}.pth**

        '''
        for model in self.models:
            model.save(prefix)

    def get_model(self, gene):
        '''
        Gets model for gene

        Parameters
        ----------

        gene : str
            Fetch RP model for this gene

        '''
        try:
            return self.models[np.argwhere(self.genes == gene)[0,0]]
        except IndexError:
            raise IndexError('Model for gene {} does not exist'.format(gene))

    def load(self, prefix):
        '''
        Load RP models saved with *prefix*.

        Parameters
        ----------

        prefix : str
            Prefix under which RP models were saved.

        '''

        genes = self.genes
        self.models = []
        for gene in genes:
            try:
                model = GeneModel(gene)
                model.load(prefix)
                self.models.append(model)
            except FileNotFoundError:
                logger.warn('Cannot load {} model. File not found.'.format(gene))

        if len(self.models) == 0:
            raise ValueError('No models loaded.')

        return self

    def _subset_fit_models(self, models):

        self.models = []
        for model in models:
            if not model.was_fit:
                logger.warn('{} model failed to fit.'.format(model.gene))
            else:
                self.models.append(model)

        return self

    #@wraps_rp_func(lambda self, expr_adata, atac_data, output, **kwargs : self._subset_fit_models(output), bar_desc = 'Fitting models')
    def _fit(self,*, expr_adata, atac_adata):
        '''
        Optimize parameters of RP models to learn *cis*-regulatory relationships.

        Parameters
        ----------

        expr_adata : anndata.AnnData
            AnnData of expression features
        atac_adata : anndata.AnnData
            AnnData of accessibility features. Must be annotated with 
            mira.tl.get_distance_to_TSS.

        Returns
        -------

        rp_model : mira.rp.LITE_Model, mira.rp.NITE_Model
            RP model with optimized parameters
 
        '''

        return _parallel_apply(
            lambda model, features : model.fit(**features),
            n_jobs=self.n_jobs,
            expr_adata=expr_adata,
            atac_adata=atac_adata,
        )

        

    @wraps_rp_func(lambda self, expr_adata, atac_data, output, **kwargs: np.array(output).sum(), bar_desc = 'Scoring')
    def _score(self, model, features):
        return model.score(features)


    @wraps_rp_func(lambda self, expr_adata, atac_data, output, **kwargs: \
        add_predictions(expr_adata, (self.features, output), model_type = self.model_type, sparse = True),
        bar_desc = 'Predicting expression')
    def _predict(self, model, features):
        '''
        Predicts the expression of genes given their *cis*-accessibility state.
        Also evaluates the probability of that prediction for LITE/NITE evaluation.

        Parameters
        ----------

        expr_adata : anndata.AnnData
            AnnData of expression features
        atac_adata : anndata.AnnData
            AnnData of accessibility features. Must be annotated with 
            mira.tl.get_distance_to_TSS.

        Returns
        -------

        anndata.AnnData
            `.layers['LITE_prediction']` or `.layers['NITE_prediction']`: np.ndarray[float] of shape (n_cells, n_features)
                Predicted relative frequencies of features using LITE or NITE model, respectively
            `.layers['LTIE_logp']` or `.layers['NITE_logp']` : np.ndarray[float] of shape (n_cells, n_features)
                Probability of observed expression given posterior predictive estimate of LITE or
                NITE model, respectively.
        
        '''
        try:
            return True, model.predict(features)
        except ValueError:
            return False, None


    def fit_score_predict(self,*,expr_adata, atac_adata):
        
        all_data = dict(
                expr_adata = expr_adata, 
                atac_adata = atac_adata
            )
        
        def _train_test_split():
            pass
        
        train, test = _train_test_split(**all_data)

        self.models = self._fit(**train)

        per_gene_scores, per_cell_scores = self._score(**test)

        predictions = self._predict(**all_data)

        ## add adata stuff

        return self 


    @wraps_rp_func(lambda self, expr_adata, atac_adata, output, **kwargs : output, bar_desc = 'Formatting features')
    def get_features(self, model, features):
        return features


    @wraps_rp_func(lambda self, expr_adata, atac_adata, output, **kwargs : output, 
        bar_desc = 'Formatting features', include_factor_data = True)
    def get_isd_features(self, model, features,*,hits_matrix, metadata):
        return features


    @wraps_rp_func(add_isd_results, 
        bar_desc = 'Predicting TF influence', include_factor_data = True)
    def probabilistic_isd(self, model, features, n_samples = 1500, checkpoint = None,
        *,hits_matrix, metadata):
        '''
        For each gene, calcuate association scores with each transcription factor.
        Association scores detect when a TF binds within *cis*-regulatory
        elements (CREs) that are influential to expression predictions for that gene.
        CREs that influence the RP model expression prediction are nearby a 
        gene's TSS and have accessibility that correlates with expression. This
        model assumes these attributes indicate a factor is more likely to 
        regulate a gene. 

        Parameters
        ----------

        expr_adata : anndata.AnnData
            AnnData of expression features
        atac_adata : anndata.AnnData
            AnnData of accessibility features. Must be annotated with TSS and factor
            binding data using mira.tl.get_distance_to_TSS **and** 
            mira.tl.get_motif_hits_in_peaks/mira.tl.get_CHIP_hits_in_peaks.
        n_samples : int>0, default=1500
            Downsample cells to this amount for calculations. Speeds up computation
            time. Cells are sampled by stratifying over expression levels.
        checkpoint : str, default = None
            Path to checkpoint h5 file. pISD calculations can be slow, and saving
            a checkpoint ensures progress is not lost if calculations are 
            interrupted. To resume from a checkpoint, just pass the path to the h5.

        Returns
        -------

        anndata.AnnData
            `.varm['motifs-prob_deletion']` or `.varm['chip-prob_deletion']`: np.ndarray[float] of shape (n_genes, n_factors)
                Association scores for each gene-TF combination. Higher scores indicate
                greater predicted association/regulatory influence.

        '''

        already_calculated = False
        if not checkpoint is None:
            if not os.path.isfile(checkpoint):
                h5.File(checkpoint, 'w').close()

            with h5.File(checkpoint, 'r') as h:
                try:
                    h[model.gene]
                    already_calculated = True
                except KeyError:
                    pass

        if checkpoint is None or not already_calculated:
            result = model.probabilistic_isd(features, hits_matrix, n_samples = n_samples)

            if not checkpoint is None:
                with h5.File(checkpoint, 'a') as h:
                    g = h.create_group(model.gene)
                    g.create_dataset('samples_mask', data = result[1])
                    g.create_dataset('isd', data = result[0])

            return result
        else:
            with h5.File(checkpoint, 'r') as h:
                g = h[model.gene]
                result = g['isd'][...], g['samples_mask'][...]

            return result

    @property
    def parameters_(self):
        '''
        Returns parameters of all contained RP models.
        '''
        return {
            gene : self[gene]._params
            for gene in self.features
        }
    