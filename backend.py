# --------------------- LIBRARIES ------------------------------------------------------
import joblib
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import MACCSkeys

# --------------------- DESCRIPTOR FUNCTION --------------------------------------------
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

# --------------------- PREDICTION FUNCTION --------------------------------------------
def predictlogs(smiles):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_pipeline = joblib.load(os.path.join(BASE_DIR, "logS_pipeline.pkl"))

    features = get_all_descriptors(smiles)
    if features is None:
        return "Invalid SMILES"
    
    prediction = model_pipeline.predict([features])[0]
    return prediction

# --------------------- COMMAND LINE ENTRY POINT ---------------------------------------
if __name__ == "__main__":
    smiles = input("Enter SMILES: ")
    prediction = predictlogs(smiles)
    print(f"Predicted logS: {prediction}")
