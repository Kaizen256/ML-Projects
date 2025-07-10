# ML-Models-on-Datasets
Collection of machine learning projects created using scikit-learn, PyTorch, or implemented from scratch in NumPy. All projects are original, built from the ground up based on my own understanding. No tutorials were used, these are genuine implementations to deepen my grasp of ML concepts.

# Top Projects
These are projects I am especially proud of. This list will evolve as I continue to learn and build more.

| Project                                  | Description                                                                                                                                                                                  | Location                                      |
|------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------|
| **LeNet-5 on MNIST (NumPy & PyTorch)**   | Implemented LeNet-5 entirely from scratch in NumPy, with manual forward & backward for convolutions, pooling, and dense layers. Later replicated in PyTorch to achieve ~98% on MNIST.        | [`LeNet-5_Numpy/`](./LeNet-5_Numpy)           |
| **GoogLeNet from Scratch (Tiny ImageNet)** | Built Inception v1 from scratch in PyTorch, carefully adapted for 64×64 Tiny ImageNet by modifying pooling & convolutions. Included hand-drawn architecture diagrams and tensor checks.    | [`GoogLeNet_Scratch/`](./GoogLeNet_Scratch) |
| **Stellar Classification (SDSS)**        | Classified stars, galaxies, and quasars using Sloan Digital Sky Survey data. Engineered rich photometric features, handled imbalance with SMOTE, tuned CatBoost/LightGBM/XGBoost to ~98.5%.  | [`Stellar_Classification/`](./Stellar_Classification) |
| **Tiny ImageNet ResNet-34 (PyTorch)**    | Modified ResNet-34 for small 64×64 Tiny ImageNet images by removing early pooling and adjusting initial convolutions. Trained on Kaggle GPUs to ~61% top-1 accuracy.                         | [`ResNet-34_Scratch/`](./ResNet-34_Scratch)   |
| **Invariant Mass from Dielectron Events**| Predicted invariant mass of electron pairs from CMS proton-proton collisions (CERN). Used physics-driven features & XGBoost tuned to ~0.997 R² on test data.                                 | [`Invariant_Mass_CERN/`](./Invariant_Mass_CERN) |




| **Exoplanet Detection (Kepler)**           | Classified NASA Kepler objects as true exoplanets or false positives using LightGBM, XGBoost, Random Forest, and SGD. Conducted careful leakage removal and preprocessing. Tuned models to achieve \~95% test accuracy. Used 5-fold CV and plotted correlation heatmaps.                                                                         |
| **Mental Health Depression Detection**     | Predicted depression from a large-scale survey dataset using manual feature engineering and a ColumnTransformer pipeline. Applied XGBoost with Optuna tuning, achieving \~94% test accuracy. Included features from external data (crime rates) and handled diverse categorical variables.                                                       |
| **MLP on MNIST (NumPy)**                   | Built a fully connected neural network from scratch in NumPy with ReLU, softmax, cross-entropy, and explicit backprop. Used mini-batch gradient descent and He initialization, achieving \~98.3% accuracy on MNIST. A pure linear algebra implementation demonstrating how MLPs learn.                                                           |
| **Abalone Age Prediction**                 | Predicted the age of abalone based on physical measurements (length, diameter, height, whole weight, etc.), replacing the tedious ring-counting lab process. Used XGBoost, Random Forest, and SGDRegressor, achieving a test RMSE as low as \~2.01. Applied feature engineering, outlier removal, scaling, and Optuna for hyperparameter tuning. |
| **Softmax Classifier on Iris (NumPy)**     | Implemented a multi-class logistic regression (softmax) classifier from scratch using only NumPy, trained on the Iris dataset. Used SGD and cross-entropy loss, achieving \~97.4% test accuracy, illustrating fundamental multiclass classification without ML libraries.                                                                        |
