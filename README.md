# logs-predictor
Predicts aqueous solubility (LogS) at STP of organic compounds using SMILES

Aqueous solubility is a defined as a compounds ability to dissociate in water in mol/dm^3. This physical property is obtained experimentally. When experimentation cannot be done literature and databased are used. However there are limitations of this. Novel/Rare molecules may not exist on a database. 

To address this large databases such as AqSol and ESOL use prediction algorithms and equations. There exist machine learning models built of these datasets to make predicitions that often have a large MAE ranging from 0.5 - 1. These deviations are huge and as a result causes these prediction algorithms to be inaccurate.

Using MACCS keys (166 features), Morgan Fingerprints (512), Dominant 2D descriptors (11) and (2) 3D descriptors allowed every molecule in SMILE form to be expressed as a 691 bit key. The model was trained on the experimental values in the ESOL database which after cleaning and processing cave 1128 SMILES to train the model with. 

Using XGBoost which is an optmized random forest algorithm that learns after every iteration. Hyperoptimization using Baysian Optimization was implimented to create and train the trees. The decision trees are made for every feature in order to evaluate how much that given feature affects the LogS.

The result is a model with a MAE of 0.085




