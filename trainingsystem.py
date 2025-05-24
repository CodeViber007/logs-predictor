# ------------------------- LIBRARIES ------------------------------------------------
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import MACCSkeys
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import joblib

# ------------------------- LOAD DATA ------------------------------------------------
url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
esol_data = pd.read_csv(url)
esol_data = esol_data[esol_data['smiles'].apply(lambda x: Chem.MolFromSmiles(x) is not None)]
esol_data.dropna(inplace=True)
esol_data.drop_duplicates(subset='smiles', inplace=True)

# ------------------------- FEATURE GENERATION ---------------------------------------
fp_generator = GetMorganGenerator(radius=2, fpSize=512)

def get_all_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    radius_of_gyration = 0
    pbf = 0
    mol_3d = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol_3d)
        if mol_3d.GetNumConformers() > 0:
            radius_of_gyration = rdMolDescriptors.CalcRadiusOfGyration(mol_3d)
            pbf = rdMolDescriptors.CalcPBF(mol_3d)
    except:
        pass

    morgan_fp = np.array(fp_generator.GetFingerprint(mol))
    maccs_fp = np.array(MACCSkeys.GenMACCSKeys(mol))
    descriptors_2d = np.array([
        Descriptors.MolWt(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.RingCount(mol),
        Descriptors.HeavyAtomCount(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        Descriptors.FractionCSP3(mol),
        rdMolDescriptors.CalcChi0n(mol),
        rdMolDescriptors.CalcHallKierAlpha(mol)
    ])
    descriptors_3d = np.array([radius_of_gyration, pbf])

    return np.concatenate([morgan_fp, maccs_fp, descriptors_2d, descriptors_3d])

# ------------------------- FEATURE MATRIX -------------------------------------------
X_raw = [get_all_descriptors(smi) for smi in esol_data['smiles']]
valid_indices = [i for i, x in enumerate(X_raw) if x is not None]
X = np.array([X_raw[i] for i in valid_indices])
y = esol_data['measured log solubility in mols per litre'].values[valid_indices]

# ------------------------- PIPELINE -------------------------------------------------
pipeline = Pipeline([
    ('selector', SelectKBest(mutual_info_regression, k=300)),
    ('scaler', StandardScaler()),
    ('model', XGBRegressor(random_state=42, n_jobs=-1))
])

# ------------------------- HYPERPARAMETER SPACE -------------------------------------
search_space = {
    'model__n_estimators': Integer(100, 500),
    'model__max_depth': Integer(3, 10),
    'model__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'model__subsample': Real(0.5, 1.0),
    'model__colsample_bytree': Real(0.5, 1.0),
    'model__gamma': Real(0, 5),
    'model__reg_alpha': Real(0, 10),
    'model__reg_lambda': Real(0, 10)
}

# ------------------------- BAYESIAN OPTIMIZATION ------------------------------------
opt = BayesSearchCV(
    pipeline,
    search_space,
    n_iter=50,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    random_state=42
)
opt.fit(X, y)
best_pipeline = opt.best_estimator_

# ------------------------- EVALUATION -----------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
test_preds = best_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, test_preds)
print(f"\nFinal logS MAE: {mae:.3f}")

# ------------------------- SAVE MODEL -----------------------------------------------
joblib.dump(best_pipeline, 'logS_pipeline.pkl')

# ------------------------- PREDICTION FUNCTION --------------------------------------
def predict_logS(smiles):
    features = get_all_descriptors(smiles)
    if features is None:
        return None
    return best_pipeline.predict([features])[0]

# ------------------------- EXAMPLE --------------------------------------------------
example_smiles = "CCO"
predicted_logS = predict_logS(example_smiles)
print(f"Predicted logS for {example_smiles}: {predicted_logS:.2f}")

