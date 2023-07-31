import numpy as np
import logging
from mira.adata_interface.core import add_layer, add_obs_col
import h5py as h5
import os
from ._gene_model import GeneModel
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import pickle

logger = logging.getLogger(__name__)


def _pickle_load(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
    
def _pickle_save(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, file)


def _parallel_apply(func, gene_models, n_jobs = 1, bar_desc = '', **feature_kw):

    return Parallel(n_jobs=n_jobs, verbose=0, pre_dispatch='2*n_jobs', max_nbytes = None, return_as = 'generator')\
                ( delayed(func)(model, features) 
                  for model, features in tqdm(
                        map(lambda x : (x, _generate_features(x, **feature_kw)), gene_models), 
                        desc = bar_desc, total = len(gene_models)
                    )
                )


class RPModel:

    @classmethod
    def load(cls,counts_layer = None,*,expr_model, accessibility_model, filename):
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

            >>> rpmodel = mira.rp.LITE_Model.load_dir(
            ...     counts_layer = 'counts',
            ...     expr_model = rna_model, 
            ...     accessibility_model = atac_model,
            ...     prefix = 'path/to/rpmodels/'
            ... )

        '''

        model = cls(expr_model = expr_model, 
                    accessibility_model = accessibility_model,
                    counts_layer = counts_layer, 
                    genes = [])\
                ._load(filename)

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

    def _load(self, filename):

        save_data = _pickle_load(filename)

        for gene, data in save_data.items():

            self.models.append(
                GeneModel(gene = gene)._load_save_data(data)
            )

        return self
    
    def save(self, filename):
        '''
        Save RP models.

        Parameters
        ----------

        prefix : str
            Prefix under which to save RP models. May be filename prefix
            or directory. RP models will save with format:
            **{prefix}_{LITE/NITE}_{gene}.pth**

        '''

        save_data = {
            model.gene : model._get_save_data()
            for model in self.models
        }

        _pickle_save(save_data, filename)


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

        assert(isinstance(rp_model, RPModel))

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


    #@wraps_rp_func(lambda self, expr_adata, atac_data, output, **kwargs : self._subset_fit_models(output), bar_desc = 'Fitting models')
    def fit(self,*, expr_adata, atac_adata):
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

        def _apply(model, features):
            return model.fit(**features)

        return list(_parallel_apply(
            _apply, self.models,
            n_jobs=self.n_jobs,
            expr_adata=expr_adata,
            atac_adata=atac_adata,
            expr_model = self.expr_model,
            accessibility_model = self.accessibility_model,
            bar_desc='Fitting models'
        ))

        

    #wraps_rp_func(lambda self, expr_adata, atac_data, output, **kwargs: np.array(output).sum(), bar_desc = 'Scoring')
    def score(self,*,expr_adata, atac_adata):
        
        def _apply(model, features):
            return model.score(**features)
        
        cell_scores = np.zeros(len(expr_adata))
        gene_scores = np.empty(len(self.models))
        
        for i, per_cell_scores in enumerate(_parallel_apply(
            _apply, self.models,
            n_jobs=self.n_jobs,
            expr_adata=expr_adata,
            atac_adata=atac_adata,
            expr_model = self.expr_model,
            accessibility_model = self.accessibility_model,
            bar_desc= 'Calculating deviances'
        )):
            
            cell_scores = cell_scores + per_cell_scores
            gene_scores[i] = per_cell_scores.sum()

        return cell_scores, gene_scores


    def predict(self, expr_adata, atac_adata):
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
        def _apply(model, features):
            return model.predict(**features)

        nite_predictions, lite_predictions = list(zip(*_parallel_apply(
            _apply, self.models,
            n_jobs=self.n_jobs,
            expr_adata=expr_adata,
            atac_adata=atac_adata,
            expr_model = self.expr_model,
            accessibility_model = self.accessibility_model,
            bar_desc= 'Predicting expression'
        )))

        nite_predictions = np.hstack(nite_predictions)
        lite_predictions = np.hstack(lite_predictions)

        return nite_predictions, lite_predictions


    def fit_score_predict(self,*,expr_adata, atac_adata):
        
        all_data = dict(
                expr_adata = expr_adata, 
                atac_adata = atac_adata
            )
        
        np.random.seed(self.seed)
        train_set = np.random.rand(len(expr_adata)) < 0.7

        def subset_dictionary(data, mask):
            return {k : v[mask] for k, v  in data.items()}
        
        train, test = subset_dictionary(all_data, train_set), subset_dictionary(all_data, ~train_set)

        self.models = self.fit(**train)
        cell_scores, gene_scores = self.score(**test)
        predictions = self.predict(**all_data)

        return cell_scores, gene_scores, predictions 


    #@wraps_rp_func(add_isd_results, 
    #    bar_desc = 'Predicting TF influence', include_factor_data = True)
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
    