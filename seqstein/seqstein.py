import numpy as np
import time

from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import make_scorer, mean_squared_error

from matplotlib import pyplot as plt
from matplotlib import gridspec

from .mysim import SIM

__all__ = ["SeqStein"]

class SeqStein(BaseEstimator, RegressorMixin):
    
    def __init__(self, nterms=5, reg_lambda=0.2,
                 reg_gamma=0, knot_num=10,
                 early_stop_thres=1, ortho_enhance=True,
                 val_ratio=0.2,
                 random_state=0):
        self.nterms = nterms
        self.reg_lambda = reg_lambda
        self.reg_gamma = reg_gamma
        self.knot_num = knot_num
        self.early_stop_thres = early_stop_thres
        self.ortho_enhance = ortho_enhance
        self.val_ratio = val_ratio
        self.random_state = random_state
        
        if not isinstance(self.reg_lambda,list):
            self.reg_lambda = [self.reg_lambda]
        
        if not isinstance(self.reg_gamma,list):
            self.reg_gamma = [self.reg_gamma]
            
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

    
    def fit(self, x, y):
        
        datanum = x.shape[0]
        indices = np.arange(datanum)
        valnum = int(round(datanum * self.val_ratio))
        
        self.tr_idx, self.val_idx = train_test_split(indices, test_size=valnum, random_state=self.random_state)
        self.val_fold = np.ones((len(indices)))
        self.val_fold[self.tr_idx] = -1
        
        pred_val = 0
        temp = y.copy()
        mse_opt = np.inf
        early_stop_count = 0
        self.best_estimators_ = [] 
        for idx in range(self.nterms):

            # projection matrix
            if (idx == 0) or (idx >= x.shape[1]) or (self.ortho_enhance == False):
                proj_mat = np.eye(x.shape[1])
            else:
                betas = np.array([model.beta_.flatten() for model in self.best_estimators_]).T
                betas = betas / np.linalg.norm(betas, axis=0)
                u, _, _ = np.linalg.svd(betas, full_matrices=False)
                proj_mat = np.eye(u.shape[0]) - np.dot(u, u.T)

            # initialization
            param_grid = {"method": ["second",'first'],
                          "reg_lambda": self.reg_lambda,
                          "reg_gamma": self.reg_gamma}
            grid = GridSearchCV(SIM(degree=3, knot_num=self.knot_num,
                                    random_state=self.random_state), 
                              scoring={"mse": make_scorer(mean_squared_error, 
                                                          greater_is_better=False)}, refit=False,
                              cv=PredefinedSplit(self.val_fold), param_grid=param_grid, verbose=0, error_score=np.nan)
            # time
            grid.fit(x, temp, proj_mat=proj_mat)
            model = grid.estimator.set_params(**grid.cv_results_['params'][np.where((grid.cv_results_['rank_test_mse'] == 1))[0][0]])
            model.fit(x[self.tr_idx, :], temp[self.tr_idx,:].ravel(), proj_mat=proj_mat)

            # early stop
            pred_val_temp = pred_val + model.predict(x[self.val_idx, :]).reshape([-1, 1])
            mse_new = np.mean((pred_val_temp - y[self.val_idx]) ** 2)
            if mse_opt > mse_new:           
                mse_opt = mse_new
                early_stop_count = 0
            else:
                early_stop_count += 1

            if early_stop_count >= self.early_stop_thres:
                break

            # update 
            pred_val += model.predict(x[self.val_idx, :]).reshape([-1, 1])
            temp = temp - model.predict(x).reshape([-1, 1])
            self.best_estimators_.append(model)
        
        component_importance = {}
        for indice, est in enumerate(self.best_estimators_):
            component_importance.update({"sim " + str(indice + 1): {"type": "sim", "indice": indice,
                                "importance": np.std(est.predict(x[self.tr_idx, :]))}})

        self.component_importance_ = dict(sorted(component_importance.items(), key=lambda item: item[1]["importance"])[::-1])
        return self
    
    def predict(self, x):
        check_is_fitted(self, "best_estimators_")

        y_pred = 0
        for i, est in enumerate(self.best_estimators_):
            y_pred += est.predict(x).reshape([-1, 1])
        return y_pred
    
    def visualize(self, cols_per_row=3, 
                  show_top=None, 
                  max_ids=None,
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
        if max_ids is None:
            max_ids = len(self.best_estimators_)+1
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
                if show_indices > 50:
                    ax2.set_yticklabels([])
                else:
                    ax2.set_yticklabels(["X" + str(idx + 1) for idx in input_ticks][::-1])
                ax2.set_xlim(xlim_min, xlim_max)
                ax2.set_ylim(-1, min(len(sim.beta_),show_indices))
                ax2.axvline(0, linestyle="dotted", color="black")
            fig.add_subplot(ax2)
            subfig_idx += 1
            
        #plt.show()
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