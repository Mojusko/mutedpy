
class QuadraticModelLearner(ProteinKernelLearner):

    def __str__(self):
        return "quadratic_model"

    def fit(self):
        Embedding = AminoAcidEmbedding(data = "data/amino-acid-features.csv")
        Embedding.load_projection("data/projection-dim5-norm.pt")
        ChebEmbedding = PolynomialEmbedding(d=Embedding.projected_components ,p=2)

        kernel_function = lambda x ,y ,kappa ,group: \
                    (ChebEmbedding.embed(Embedding.embed(x)) @ ChebEmbedding.embed(Embedding.embed(y)).T).T
        k = KernelFunction(kernel_function=kernel_function, d=5)

        self.estimate_noise_std()
        self.GP = GaussianProcess(kernel=k, s=self.s)
        self.GP.fit_gp(self.x_train, self.y_train)
        self.fitted = True


class LassoModelLearner(ProteinKernelLearner):

    def __str__(self):
        prefix = self.loss
        if self.features_V:
            prefix += 'volume_'
        if self.features_G:
            prefix += 'geometric_'
        if self.features_A:
            prefix += "AA_"
        if self.pca:
            prefix += "pca_"
        if self.feature_split:
            prefix += "splits_"
        return prefix + "lasso_linear"

    def fit(self):
        self.EmbeddingV = LookUpMap(self.data_folder + "new_features_volume.csv", truncation=False)
        if self.pca:
            self.EmbeddingV.pca(std=0.1, relative_var=True)
            self.EmbeddingV.normalize(xtest=True)
        else:
            self.EmbeddingV.normalize(xtest=True)
            self.EmbeddingV.restrict_to_varaint(std=0.001)

        if self.feature_split:
            self.EmbeddingV.split_features(xtest=True)

        self.EmbeddingA = AminoAcidEmbedding(data=self.data_folder + "amino-acid-features.csv")
        if self.pca:
            self.EmbeddingA.load_projection(self.data_folder + "projection-dim5-demean-norm.pt")
        else:
            self.EmbeddingA.load_projection(self.data_folder + "embedding-dim5-demean-norm.pt")

        if self.feature_split:
            self.EmbeddingA.split_features(xtest=True, n_sites=5)

        self.EmbeddingG = LookUpMap(self.data_folder + "new_features_rosetta.csv", truncation=False)
        if self.pca:
            self.EmbeddingG.pca(std=0.01, relative_var=True)
            self.EmbeddingG.normalize(xtest=True)
        else:
            self.EmbeddingG.normalize(xtest=True)
            self.EmbeddingG.restrict_to_varaint(std=0.001)

        if self.feature_split:
            self.EmbeddingG.split_features(xtest=True)

        self.Embedding = AdditiveEmbeddings([self.EmbeddingG, self.EmbeddingV, self.EmbeddingA],
                                            [self.EmbeddingG.m, self.EmbeddingV.m,
                                             self.EmbeddingA.projected_components])
        self.embed = lambda x: torch.hstack(
            [self.EmbeddingG.embed(x), self.EmbeddingV.embed(x), self.EmbeddingA.embed(x)])

        self.estimate_noise_std()
        phi_train = self.embed(self.x_train)
        if self.loss == "squared":
            self.regr = LassoCV(cv=10, n_alphas=200, random_state=0, max_iter=8000).fit(phi_train.numpy(),
                                                                                        self.y_train.numpy().ravel())
        elif self.loss == "huber":
            alphas = _alpha_grid(phi_train.numpy(), self.y_train.numpy().ravel(), n_alphas=200)
            scores = []
            for alpha in alphas:
                self.regr = sklearn.linear_model.SGDRegressor(loss="huber", tol=1e-5, penalty='l1', alpha=alpha,
                                                              l1_ratio=0.)
                scores.append(
                    np.mean(cross_val_score(self.regr, phi_train.numpy(), self.y_train.numpy().ravel(), cv=10)))
            alpha = alphas[np.argmax(scores)]
            self.regr = sklearn.linear_model.SGDRegressor(loss="huber", penalty='l1', alpha=alpha, l1_ratio=0.)
            self.regr.fit(phi_train.numpy(), self.y_train.numpy().ravel())
        dts = pd.DataFrame([self.regr.coef_, np.concatenate(
            (self.EmbeddingG.feature_names, self.EmbeddingV.feature_names, self.EmbeddingA.feature_names))]).T
        dts.to_csv(self.results_folder + str(self) + "/lasso-features" + str(self.split) + ".csv")
        self.fitted = True

    def predict(self):
        mu = self.regr.predict(self.embed(self.x_test).numpy())
        mu_train = self.regr.predict(self.embed(self.x_train).numpy())

        std = mu * 0
        std_train = std * 0

        self.mu = torch.from_numpy(mu)
        self.std = torch.from_numpy(std)
        self.mu_train = torch.from_numpy(mu_train)
        self.std_train = torch.from_numpy(std_train)
        return self.mu, self.std


class LinearContactMap(ProteinKernelLearner):

    def __str__(self):
        return "linear+contact_map"

    def load(self):
        self.Embedding = AminoAcidEmbedding(data="data/amino-acid-features.csv")
        self.Embedding.load_projection("data/projection-dim5-norm.pt")
        self.ContactMapEmbedding = LookUpMap("data/contact_map_1.csv")
        self.ContactMapEmbedding.restrict_to_varaint()

    def fit(self):
        self.load()

        def kernel_function(x, y, kappa=1., group=None, kappa2=1., cut_off=0.):
            return kappa ** 2 * (self.Embedding.embed(x) @ self.Embedding.embed(y).T).T + 1 \
                   + kappa2 ** 2 * (
                               self.ContactMapEmbedding.embed(x, cut=cut_off ** 2) @ self.ContactMapEmbedding.embed(y,
                                                                                                                    cut=cut_off ** 2).T).T

        k = KernelFunction(kernel_function=kernel_function, d=5, params={'kappa': 1., 'kappa2': 1., 'cut_off': 10.})

        self.estimate_noise_std()
        self.GP = GaussianProcess(kernel=k, s=self.s)
        self.GP.fit_gp(self.x_train, self.y_train)
        self.GP.optimize_params_general(
            params={'0': {"cut_off": (1., Euclidean(1), None), 'kappa2': (1., Euclidean(1), None)}},
            restarts=self.restarts, verbose=True, maxiter=self.maxiter, mingradnorm=10e-6, optimizer='pytorch-minimize',
            scale=10)

        self.fitted = True


class LinearOnlyContactMap(LinearContactMap):

    def __str__(self):
        return "contact_map"

    def fit(self):
        self.load()

        def kernel_function(x, y, kappa=1., group=None, cut_off=0.):
            return kappa ** 2 * (self.ContactMapEmbedding.embed(x, cut=cut_off ** 2) @ self.ContactMapEmbedding.embed(y,
                                                                                                                      cut=cut_off ** 2).T).T

        k = KernelFunction(kernel_function=kernel_function, d=5, params={'kappa': 1., 'cut_off': 10.})

        self.estimate_noise_std()
        self.GP = GaussianProcess(kernel=k, s=self.s)
        self.GP.fit_gp(self.x_train, self.y_train)
        self.GP.optimize_params_general(params={'0': {"cut_off": (1., np.arange(0, 10, 10), None)}},
                                        restarts=self.restarts, verbose=True, maxiter=self.maxiter, mingradnorm=10e-4,
                                        optimizer='discrete', scale=1)
        self.fitted = True


class ARDModelLearnerContactMap(ProteinKernelLearner):

    def __str__(self):
        return "ard_model+contact_map"

    def fit(self):
        self.EmbeddingAA = AminoAcidEmbedding(data="data/amino-acid-features.csv")
        self.EmbeddingAA.load_projection("data/projection-dim5-norm.pt")
        self.ContactMapEmbedding = LookUpMap("data/contact_map_1.csv", truncation=False)
        self.ContactMapEmbedding.restrict_to_varaint()

        self.embed = lambda x: torch.hstack([self.EmbeddingAA.embed(x), self.ContactMapEmbedding.embed(x)])
        d = self.EmbeddingAA.projected_components + self.ContactMapEmbedding.m
        phi_train = self.embed(self.x_train)

        k = KernelFunction(kernel_name="ard", kappa=3., ard_gamma=torch.ones(d).double() * 0.01, d=d)

        self.estimate_noise_std()
        self.GP = GaussianProcess(kernel=k, s=self.s)
        self.GP.fit_gp(phi_train, self.y_train)
        self.GP.optimize_params(type="bandwidth", restarts=self.restarts, verbose=True,
                                maxiter=self.maxiter, mingradnorm=10e-5, optimizer='pytorch-minimize', scale=100)
        self.GP.fit_gp(phi_train, self.y_train)
        self.fitted = True

    def predict(self):
        self.mu, self.std = self.GP.mean_std(self.embed(self.x_test))
        return self.mu, self.std


class ARDModelLearnerOnlyContactMap(ARDModelLearnerContactMap):

    def __str__(self):
        return "ard_contact_map"

    def fit(self):
        self.EmbeddingAA = AminoAcidEmbedding(data="data/amino-acid-features.csv")
        self.EmbeddingAA.load_projection("data/projection-dim5-norm.pt")
        self.ContactMapEmbedding = LookUpMap("data/contact_map_1.csv", truncation=False)
        self.ContactMapEmbedding.restrict_to_varaint()

        self.embed = lambda x: self.ContactMapEmbedding.embed(x)
        d = self.ContactMapEmbedding.m
        phi_train = self.embed(self.x_train)

        k = KernelFunction(kernel_name="ard", kappa=3., ard_gamma=torch.ones(d).double() * 0.01, d=d)

        self.estimate_noise_std()
        self.GP = GaussianProcess(kernel=k, s=self.s)
        self.GP.fit_gp(phi_train, self.y_train)
        self.GP.optimize_params(type="bandwidth", restarts=self.restarts, verbose=True, maxiter=self.maxiter,
                                mingradnorm=10e-5, optimizer='pytorch-minimize', scale=100)
        self.GP.fit_gp(phi_train, self.y_train)
        self.fitted = True


class GeometricLinearFeatures(ProteinKernelLearner):

    def __str__(self):
        return "linear_geometric"

    def load(self):
        self.Embedding = LookUpMap(self.features, truncation=False)
        if self.pca:
            self.Embedding.pca(std=self.err)
        else:
            self.Embedding.restrict_to_varaint(std=self.err)

        self.Embedding.normalize(xtest=True)

    def fit(self, optimize=True):
        self.load()
        phi_train = self.Embedding.embed(self.x_train)
        d = self.Embedding.m
        kernel_function = lambda x, y, kappa, group: (self.Embedding.embed(x) @ self.Embedding.embed(y).T).T + 1.
        k = KernelFunction(kernel_function=kernel_function, d=d)

        self.estimate_noise_std()
        self.GP = GaussianProcess(kernel=k, s=self.s)
        self.GP.fit_gp(self.x_train, self.y_train)
        self.fitted = True

    def embed(self, x):
        phi_test = self.Embedding.embed(x)
        return phi_test


if __name__ == "__main__":

    # Load data

    filename = "../../../data/streptavidin/5sites.xls"
    loader = BaselLoader(filename)
    dts = loader.load()

    filename = "../../../data/streptavidin/2sites.xls"
    loader = BaselLoader(filename)
    total_dts = loader.load(parent='SK', positions=[112, 121])
    total_dts = loader.add_mutations('T111T+N118N+A119A', total_dts)

    total_dts = pd.concat([dts, total_dts], ignore_index=True, sort=False)
    total_dts = drop_neural_mutations(total_dts)
    total_dts['LogFitness'] = np.log10(total_dts['Fitness'])

    Op = ProteinOperator()
    x = torch.from_numpy(Op.translate_mutation_series(total_dts['variant']))
    y = torch.from_numpy(total_dts['LogFitness'].values).view(-1, 1)

    # initialize the model
    models = []
    restarts = 1
    maxiter = 20
    splits = 5

    models = [LassoModelLearner]
    default_params = {'restarts': restarts, 'data_folder': "../data/",
                      'results_folder': "../results_strep/", "maxiter": maxiter, 'feature_split': False, 'pca': False,
                      'loss': 'huber'}

    for model in models:
        model = model(**default_params)
        model.add_data(x, y)
        # create folder for results_strep
        try:
            os.mkdir(default_params['results_folder'] + str(model))
        except:
            print("Folder already exists.")

        # try loading it
        model.evaluate_metrics_on_splits(no_splits=splits, split_location="../splits/random_splits.pt")