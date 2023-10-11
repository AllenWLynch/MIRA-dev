import numpy as np
import logging
from ._gene_model import fit_models, score, fit_and_score
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import pickle
from mira.adata_interface.rp_model import get_feature_generator


logger = logging.getLogger(__name__)


def _pickle_load(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
    
def _pickle_save(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, file)


def _parallel_apply(func, gene_models, n_jobs = 1, bar_desc = '', **feature_kw):

    feature_generator = get_feature_generator(**feature_kw)
    # iterates over (gene_name, model) pairs
    # *lazily*, so that all of the features for all of the models are not created at once.
    model_generator = map(lambda x : (x[1], feature_generator(x[0])), gene_models ) 

    return Parallel(n_jobs=n_jobs, verbose=0, pre_dispatch='2*n_jobs', 
                    max_nbytes = None, return_as = 'generator', backend='threading')\
                ( delayed(func)(**model, **features) 
                  for model, features in tqdm(model_generator, 
                                              desc = bar_desc, total = len(gene_models))
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
        genes = None,
        counts_layer = None,
        train_split = 0.7,
        models = None,
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
        num_nones = (not models is None) + (not genes is None)
        assert num_nones == 1, 'Must provide either "genes" or "models" to instantiate RPModel object'

        self.seed = seed
        self.n_jobs = n_jobs
        self.expr_model = expr_model
        self.accessibility_model = accessibility_model
        self.counts_layer = counts_layer
        self.train_split = train_split

        if models is None:
            self.models = {
                gene : {} for gene in genes
            }
        else:
            self.models = models

    
    @property
    def genes(self):
        return np.array(list(self.models.keys()))


    @property
    def features(self):
        return self.genes

    def _load(self, filename):
        self.models = _pickle_load(filename)
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
        _pickle_save(self.models, filename)


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
        modeled_genes = self.genes
        
        for gene in genes:
            if not gene in modeled_genes:
                raise ValueError('Gene {} is not in RP model'.format(str(gene)))        


        return self.__class__(
            expr_model = self.expr_model,
            accessibility_model = self.accessibility_model, 
            counts_layer=self.counts_layer, 
            train_split= self.train_split,
            models = {
                gene : model for gene, model in self.models.items() if gene in genes
            }
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
            self.models[add_gene] = rp_model.get_model(add_gene)
            
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
            return self.models[gene]
        except IndexError:
            raise IndexError('Model for gene {} does not exist'.format(gene))
        

    def _apply(self, fn, expr_adata, atac_adata, bar_desc = ''):

        return _parallel_apply(
            fn, self.models.items(),
            n_jobs=self.n_jobs,
            expr_adata=expr_adata,
            atac_adata=atac_adata,
            expr_model = self.expr_model,
            accessibility_model = self.accessibility_model,
            bar_desc = bar_desc,
        )

        
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
        return list(self._apply(fit_models, expr_adata, atac_adata, bar_desc='Fitting models'))
    
    
    def _score_predict(self,*, expr_adata, atac_adata, include_cell_mask = None):

        num_cells = include_cell_mask.sum() if not include_cell_mask is None else len(expr_adata)

        cell_scores = np.empty( (num_cells, 3) )
        gene_scores = np.empty( (3, len(self.models)) )

        lite_predictions = np.empty( (num_cells, len(self.genes)) )
        nite_predictions = lite_predictions.copy()

        for i, (per_cell_scores, logrates) in enumerate(
            self._apply(score, expr_adata, atac_adata, 
                        bar_desc= 'Calculating deviances')
        ):
            
            per_cell_scores = np.hstack(per_cell_scores)[include_cell_mask]
            
            cell_scores = cell_scores + per_cell_scores
            gene_scores[:,i] = per_cell_scores.sum(0)

            nite_predictions[:,i] = logrates[0]
            lite_predictions[:,i] = logrates[1]
    
        return (cell_scores, gene_scores), (nite_predictions, lite_predictions)
    


    def _fit_score_predict(self,*,expr_adata, atac_adata):
                
        train_set = np.random.RandomState(self.seed).rand(len(expr_adata)) < self.test_split
        test_set = ~train_set

        cell_scores = np.empty( (test_set.sum(), 3) )
        gene_scores = np.empty( (3, len(self.features)) )

        lite_predictions = np.empty( (len(expr_adata), len(self.features)) )
        nite_predictions = lite_predictions.copy()

        def _apply(**kw):
            return fit_and_score(train_mask=train_set, **kw)
        
        models = []

        for i, (model, per_cell_scores, logrates) in enumerate(
            self._apply(_apply, expr_adata, atac_adata, 
                        bar_desc= 'Progress')
        ):
            models.append(model)
            per_cell_scores = np.hstack(per_cell_scores)[test_set]
            
            cell_scores = cell_scores + per_cell_scores
            gene_scores[:,i] = per_cell_scores.sum(0)

            nite_predictions[:,i] = logrates[0]
            lite_predictions[:,i] = logrates[1]
    
        return {
            'models' : dict(zip(self.genes, models)), 
            'scores' : (cell_scores, gene_scores), 
            'predictions' : (nite_predictions, lite_predictions),
            'train_set' : train_set,
        }