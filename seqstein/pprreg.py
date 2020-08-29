import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin
from matplotlib import pyplot as plt
from matplotlib import gridspec

from .smspline import SMSplineRegressor
from .mysim import SIM

__all__ = ["PPRRegressor"]

class PPRRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, nterms=5, opt_level='low',
                 # params of smoothing spline
                 knot_num=10, knot_dist='quantile', reg_gamma="GCV",
                 # params of stage fitting
                 eps_stage=1e-4, stage_maxiter=10,
                 # params of backfitting
                 eps_backfit=1e-3, backfit_maxiter=10,
                 random_state=0, verbose=0):

        self.nterms = nterms
        self.opt_level = opt_level
        self.reg_gamma = reg_gamma # penalty of regression spline
        self.knot_num = knot_num
        self.knot_dist = knot_dist
        self.eps_stage = eps_stage
        self.stage_maxiter = stage_maxiter
        self.eps_backfit = eps_backfit
        self.backfit_maxiter = backfit_maxiter
        self.random_state = random_state
        self.verbose = verbose
    
    @property
    def importance_ratios_(self):
        """return the estimator importance ratios (the higher, the more important the feature)

        Returns
        -------
        dict of selected estimators
            the importance ratio of each fitted base learner.
        """
        importance_ratios_ = {}
        if (self.component_importance_ is not None) and (len(self.component_importance_) > 0):
            total_importance = np.sum([item["importance"] for key, item in self.component_importance_.items()])
            importance_ratios_ = {key: {"type": item["type"],
                               "indice": item["indice"],
                               "ir": item["importance"] / total_importance} for key, item in self.component_importance_.items()}
        return importance_ratios_


    @property
    def projection_indices_(self):
        """return the projection indices

        Returns
        -------
        ndarray of shape (n_features, n_estimators)
            the projection indices
        """
        projection_indices = np.array([])
        if len(self.best_estimators_) > 0:
            projection_indices = np.array([est.beta_.flatten() 
                                for est in self.best_estimators_]).T
        return projection_indices
        
    @property
    def orthogonality_measure_(self):
        """return the orthogonality measure (the lower, the better)
        
        Returns
        -------
        float
            the orthogonality measure
        """
        ortho_measure = np.nan
        if len(self.best_estimators_) > 0:
            ortho_measure = np.linalg.norm(np.dot(self.projection_indices_.T,
                                      self.projection_indices_) - np.eye(self.projection_indices_.shape[1]))
            if self.projection_indices_.shape[1] > 1:
                ortho_measure /= self.projection_indices_.shape[1]# ((betas.shape[1] ** 2 - betas.shape[1]))
        return ortho_measure


    def fit_stage(self, x, y, beta_init=None): 
        ## initialization
        if beta_init is None:
            prev_beta = np.random.randn(x.shape[1],1)
            #prev_beta = np.linalg.lstsq(x,y,rcond=None)[0]
            #prev_beta = np.dot(x.T,y)
            prev_beta = prev_beta/np.linalg.norm(prev_beta)
        else:
            prev_beta = beta_init
        xb = np.dot(x,prev_beta)
        
        # fitting for each stage
        prev_loss = np.inf
        current_loss = -np.inf
        itr = 0
        
        while ((prev_loss<=current_loss) or (prev_loss>=(current_loss+self.eps_stage))) and (itr<self.stage_maxiter):

            ## update ridge function
            ridge_fun = SMSplineRegressor(knot_num=self.knot_num,
                                 knot_dist=self.knot_dist,
                                 reg_gamma=self.reg_gamma,
                                 xmin=xb.min(),
                                 xmax=xb.max())
            ridge_fun.fit(xb,y.flatten())

            ## update beta
            residuals = y.flatten() - ridge_fun.predict(xb)

            # first and second order derivatives of f
            df = ridge_fun.diff(xb,order=1)
            ddf = ridge_fun.diff(xb,order=2)

            # calculate newton update step (delta)
            gd = np.mean((-2*residuals*df).reshape(-1,1)*x,axis=0)
            Hess = np.dot(x.T,(2*(df**2-residuals*ddf)).reshape(-1,1)*x)/x.shape[0]
            delta = -np.dot(np.linalg.pinv(Hess),gd).reshape(-1,1)
            
            # newton update without halving delta
            beta = prev_beta+delta
            beta = beta/np.linalg.norm(beta)
            xb_tmp = np.dot(x,beta)
            tmp_loss = np.mean((y.flatten() - ridge_fun.predict(xb_tmp).flatten())**2)
            
            prev_loss = current_loss
            current_loss = tmp_loss
            prev_beta = beta
            xb = xb_tmp
            itr += 1
            
            if self.verbose:
                print('prev_loss:',prev_loss,'|current_loss:',current_loss,'|iter:',itr,
                      '|prev_loss > current_loss:',prev_loss > current_loss)
            
        return ridge_fun, beta
    
    def fit(self, x, y):
        
        r = y.copy()
        self.best_estimators_ = []
        
        if self.opt_level not in ['low','high']:
            print('opt_level should be "high" or "low".')
            raise TypeError
        
        np.random.seed(self.random_state)
        for i in range(self.nterms):
            if self.verbose:
                print('------------ nterm:', i + 1,'------------')
           
            ridge_fun, beta = self.fit_stage(x,r)
            est = SIM(degree=3, knot_num=self.knot_num, random_state=self.random_state)
            est.beta_ = beta
            est.shape_fit_ = ridge_fun
            
            self.best_estimators_.append(est)
            if len(self.best_estimators_)>1 and self.opt_level == 'high':
                self.back_fit_(x,y)
        
            xb = np.dot(x,beta)
            r = r - ridge_fun.predict(xb).reshape(-1, 1)
            
            if self.verbose:
                print('------------ MSE:', np.mean(r**2), '------------')
        
        component_importance = {}
        for indice, est in enumerate(self.best_estimators_):
            component_importance.update({"sim " + str(indice + 1): {"type": "sim", "indice": indice,
                                "importance": np.std(est.predict(x))}})
        self.component_importance_ = dict(sorted(component_importance.items(), key=lambda item: item[1]["importance"])[::-1])

        return self

    def back_fit_(self, x, y):
        
        ## backfitting
        prev_loss = np.inf
        current_loss = -np.inf
        itr = 0
        
        while ((prev_loss<=current_loss) or (prev_loss>=(current_loss+self.eps_backfit))) and (itr<self.backfit_maxiter):
            if self.verbose:
                print('----------------------------- backfitting:', 
                      itr+1,
                      '-----------------------------')
            
            for i in range(len(self.best_estimators_)):
                # residual calculation
                y_hat = self.predict(x).flatten()
                xb = np.dot(x,self.best_estimators_[i].beta_)
                y_hat_no_i = y_hat - self.best_estimators_[i].shape_fit_.predict(xb).flatten()
                r = y.flatten() - y_hat_no_i 
                # backfitting
                ridge_fun, beta = self.fit_stage(x, r, beta_init=self.best_estimators_[i].beta_.reshape(-1,1))
                # update
                self.best_estimators_[i].beta_ = beta.flatten()
                self.best_estimators_[i].shape_fit_ = ridge_fun

            prev_loss = current_loss 
            current_loss = np.mean((y.flatten()-self.predict(x).flatten())**2)
            itr += 1
            
            if self.verbose:
                print('backfitting prev_loss:',prev_loss, 
                      'backfitting current_loss:',current_loss)
        
        return self
    
    
    def predict(self, x):
        check_is_fitted(self, "best_estimators_")

        y_pred = 0
        for i, est in enumerate(self.best_estimators_):
            y_pred += est.predict(x).reshape([-1, 1])
        return y_pred
    
    def visualize(self, cols_per_row=3, 
                  show_top=None, 
                  show_indices = 20,
                  folder="./results/", name="global_plot", save_png=False, save_eps=False):

        """draw the global interpretation of the fitted model
        
        Parameters
        ---------
        cols_per_row : int, optional, default=3,
            the number of sim models visualized on each row
        show_top: None or int, default=None,
            optional, show top ridge components
        show_indices: int, default=20,
            only show first indices in high-dim cases
        folder : str, optional, defalut="./results/"
            the folder of the file to be saved
        name : str, optional, default="global_plot"
            the name of the file to be saved
        save_png : bool, optional, default=False
            whether to save the figure in png form
        save_eps : bool, optional, default=False
            whether to save the figure in eps form
        """
        check_is_fitted(self, "best_estimators_")
        
        subfig_idx = 0
        max_ids = len(self.best_estimators_)
        fig = plt.figure(figsize=(8 * cols_per_row, 4.6 * int(np.ceil(max_ids / cols_per_row))))
        outer = gridspec.GridSpec(int(np.ceil(max_ids / cols_per_row)), cols_per_row, wspace=0.15, hspace=0.25)
        
        if self.projection_indices_.shape[1] > 0:
            xlim_min = - max(np.abs(self.projection_indices_.min() - 0.1), np.abs(self.projection_indices_.max() + 0.1))
            xlim_max = max(np.abs(self.projection_indices_.min() - 0.1), np.abs(self.projection_indices_.max() + 0.1))
        
        if show_top is None:
            estimators = [self.best_estimators_[item['indice']]  for _, item in self.component_importance_.items()]
        else:
            estimators = [self.best_estimators_[item['indice']]  for _, item in self.component_importance_.items()][:show_top]
            
        for indice, sim in enumerate(estimators):

            estimator_key = list(self.importance_ratios_)[indice]
            inner = outer[subfig_idx].subgridspec(2, 2, wspace=0.2, height_ratios=[6, 1], width_ratios=[3, 1])
            ax1_main = fig.add_subplot(inner[0, 0])
            xgrid = np.linspace(sim.shape_fit_.xmin, sim.shape_fit_.xmax, 100).reshape([-1, 1])
            ygrid = sim.shape_fit_.decision_function(xgrid)
            ax1_main.plot(xgrid, ygrid, color="red")
            ax1_main.set_xticklabels([])
            ax1_main.set_title("SIM " + str(self.importance_ratios_[estimator_key]["indice"] + 1) +
                         " (IR: " + str(np.round(100 * self.importance_ratios_[estimator_key]["ir"], 2)) + "%)",
                         fontsize=16)
            fig.add_subplot(ax1_main)

            ax1_density = fig.add_subplot(inner[1, 0])  
            xint = ((np.array(sim.shape_fit_.bins_[1:]) + np.array(sim.shape_fit_.bins_[:-1])) / 2).reshape([-1, 1]).reshape([-1])
            ax1_density.bar(xint, sim.shape_fit_.density_, width=xint[1] - xint[0])
            ax1_main.get_shared_x_axes().join(ax1_main, ax1_density)
            ax1_density.set_yticklabels([])
            fig.add_subplot(ax1_density)
            
            ax2 = fig.add_subplot(inner[:, 1])
            if len(sim.beta_) <= 10:
                rects = ax2.barh(np.arange(len(sim.beta_)), [beta for beta in sim.beta_.ravel()][::-1])
                ax2.set_yticks(np.arange(len(sim.beta_)))
                ax2.set_yticklabels(["X" + str(idx + 1) for idx in range(len(sim.beta_.ravel()))][::-1])
                ax2.set_xlim(xlim_min, xlim_max)
                ax2.set_ylim(-1, len(sim.beta_))
                ax2.axvline(0, linestyle="dotted", color="black")
            else:
                input_ticks = np.arange(show_indices)[::-1]
                rects = plt.barh(np.arange(show_indices), [beta for beta in sim.beta_.ravel()][:show_indices][::-1])
                ax2.set_yticks(input_ticks)
                ax2.set_yticklabels(["X" + str(idx + 1) for idx in input_ticks][::-1])
                ax2.set_xlim(xlim_min, xlim_max)
                ax2.set_ylim(-1, min(len(sim.beta_),show_indices))
                ax2.axvline(0, linestyle="dotted", color="black")
            fig.add_subplot(ax2)
            subfig_idx += 1
            
        if max_ids > 0:
            if save_png:
                if not os.path.exists(folder):
                    os.makedirs(folder)
                save_path = folder + name
                f.savefig("%s.png" % save_path, bbox_inches="tight", dpi=100)
            if save_eps:
                if not os.path.exists(folder):
                    os.makedirs(folder)
                save_path = folder + name
                f.savefig("%s.eps" % save_path, bbox_inches="tight", dpi=100)