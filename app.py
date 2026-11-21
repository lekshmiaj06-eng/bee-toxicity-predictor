import os
import base64
import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Draw, rdFingerprintGenerator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

st.set_page_config(page_title="Bee Toxicity Predictor", page_icon="🐝")

FP_BITS = 256
FP_COLS = [f"fp{i}" for i in range(FP_BITS)]
FLAGS = ["herbicide", "fungicide", "insecticide", "other_agrochemical"]
ROUTES = ["Contact", "Oral", "Other"]
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)


# ------------------------- Picture helpers -------------------------
@st.cache_data
def img_data_uri(path):
    """Read a picture file and return it as a data URI (None if the file is missing)."""
    if not os.path.exists(path):
        return None
    ext = path.rsplit(".", 1)[-1].lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    with open(path, "rb") as f:
        return f"data:image/{mime};base64,{base64.b64encode(f.read()).decode()}"


def inline_bee(path, width, center=False):
    uri = img_data_uri(path)
    if uri:
        align = "text-align:center;" if center else ""
        st.markdown(f'<div style="{align}"><img class="bee-inline" src="{uri}" '
                    f'style="width:{width}px;"></div>', unsafe_allow_html=True)


def floating_bee(path, style):
    """Bee placed at a fixed spot on the PAGE (scrolls together with the content)."""
    uri = img_data_uri(path)
    if uri:
        st.markdown(f'<img class="bee-float" src="{uri}" style="{style}">',
                    unsafe_allow_html=True)


st.markdown("""
<style>
.bee-inline { mix-blend-mode: multiply; }

/* The page content area is the reference point for the side bees,
   so they scroll away with the page like the other two bees. */
.block-container,
[data-testid="stMainBlockContainer"] {
    position: relative;
    overflow: visible;
}

.bee-float {
    position: absolute;      /* was: fixed (that is why they did not scroll) */
    z-index: 1;
    pointer-events: none;
    mix-blend-mode: multiply;
}
@media (max-width: 1100px) { .bee-float { display: none; } }
</style>
""", unsafe_allow_html=True)


# ------------------------- Model code -------------------------
def build_row(mol, flag_values, route):
    """Turn one molecule + user choices into the same features the model was trained on."""
    row = {
        "MolWt": Descriptors.MolWt(mol),
        "HeavyAtomCount": mol.GetNumHeavyAtoms(),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "MolLogP": Descriptors.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "RotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
    }
    for f in FLAGS:
        row[f] = int(flag_values[f])
    for r in ROUTES:
        row[f"route_{r}"] = int(r == route)
    fp = gen.GetFingerprintAsNumPy(mol).astype(int)
    for i, v in enumerate(fp):
        row[f"fp{i}"] = int(v)
    return row


@st.cache_resource
def train_model():
    """Train the Stage 2 tuned AdaBoost on the cleaned ApisTox data."""
    df = pd.read_csv("dataset_final.csv")
    df = df.drop_duplicates(subset="SMILES").drop_duplicates(subset="CID").reset_index(drop=True)

    rows = []
    for _, r in df.iterrows():
        mol = Chem.MolFromSmiles(r["SMILES"])
        rows.append(build_row(mol, {f: r[f] for f in FLAGS}, r["toxicity_type"]))
    X = pd.DataFrame(rows)
    y = df["label"]

    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3, min_samples_leaf=5,
                                         class_weight="balanced", random_state=42),
        n_estimators=100, learning_rate=0.1011, random_state=42)
    model = Pipeline([
        ("prep", ColumnTransformer([("fp", VarianceThreshold(0.02), FP_COLS)],
                                   remainder="passthrough")),
        ("model", ada)])
    model.fit(X, y)
    return model, list(X.columns), len(df)


model, columns, n_train = train_model()

# ------------------------- Decorative bees -------------------------
# Positions are in pixels, measured from the top-left of the page content area.
# Negative left/right pushes a bee out into the empty side margin.
# Tweak the numbers to move them.

# bee_trail: left margin, lower down
floating_bee("bee_trail.png", "left: -130px; top: 230px; width: 90px;")
# bee_flower: right margin, below the title
floating_bee("bee_flower.png", "right: -240px; top: 470px; width: 150px;")

# ------------------------- Header -------------------------
head1, head2 = st.columns([5, 1])
with head1:
    st.title("🐝 Honey Bee Pesticide Toxicity Predictor")
    st.caption("AdaBoost (tuned) trained on the ApisTox dataset using RDKit descriptors "
               "and Morgan fingerprints. Course project demo, not a regulatory tool.")
with head2:
    # bee_trail: next to the title
    inline_bee("bee_trail.png", 90)

# ------------------------- Sidebar -------------------------
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)
st.sidebar.write("0.50 is the default. About 0.23 catches more toxic compounds "
                 "but raises more false alarms.")
st.sidebar.write(f"Model trained on {n_train} compounds.")

with st.sidebar.expander("New to chemistry? Quick guide", expanded=True):
    st.markdown("""
| Term | Simple meaning |
|---|---|
| **Pesticide** | Any chemical used to kill or control pests on crops. |
| **Insecticide** | A pesticide that kills insects. It often harms bees too. |
| **Herbicide** | A pesticide that kills weeds. |
| **Fungicide** | A pesticide that kills mould and fungus. |
""")

# ------------------------- Inputs -------------------------
EXAMPLES = {
    "Custom (type below)": "",
    "Imidacloprid (neonicotinoid insecticide)": "C1CN(C(=N1)N[N+](=O)[O-])CC2=CN=C(C=C2)Cl",
    "Dimethoate (organophosphate insecticide)": "CNC(=O)CSP(=S)(OC)OC",
    "Benzoic acid": "O=C(O)c1ccccc1",
}
choice = st.selectbox("Pick an example or type your own SMILES", list(EXAMPLES.keys()))
smiles = st.text_input("SMILES string", value=EXAMPLES[choice])

col1, col2 = st.columns(2)
with col1:
    types = st.multiselect("Agrochemical type (leave empty if unknown)",
                           ["Insecticide", "Herbicide", "Fungicide", "Other agrochemical"])
with col2:
    route = st.radio("Exposure route", ROUTES, horizontal=True)

# ------------------------- Prediction -------------------------
if st.button("Predict toxicity", type="primary"):
    mol = Chem.MolFromSmiles(smiles) if smiles.strip() else None
    if mol is None:
        st.error("Please enter a valid SMILES string.")
    else:
        flag_values = {
            "insecticide": "Insecticide" in types,
            "herbicide": "Herbicide" in types,
            "fungicide": "Fungicide" in types,
            "other_agrochemical": "Other agrochemical" in types,
        }
        row = build_row(mol, flag_values, route)
        X_new = pd.DataFrame([row])[columns]
        prob = float(model.predict_proba(X_new)[0, 1])

        left, right = st.columns([1, 1])
        with left:
            st.image(Draw.MolToImage(mol, size=(350, 300)), caption="Molecule")
        with right:
            st.metric("Probability toxic to bees", f"{prob:.1%}")
            st.progress(min(max(prob, 0.0), 1.0))
            if prob >= threshold:
                st.error(f"Predicted TOXIC (threshold {threshold:.2f})")
            else:
                st.success(f"Predicted NON-TOXIC (threshold {threshold:.2f})")

        st.subheader("Descriptors used")
        desc_names = ["MolWt", "HeavyAtomCount", "RingCount", "NumAromaticRings",
                      "MolLogP", "TPSA", "HBD", "HBA", "RotatableBonds", "FractionCSP3"]
        st.dataframe(pd.DataFrame({k: [round(row[k], 2)] for k in desc_names}))

        st.info("The agrochemical type is the strongest signal for this model, so "
                "changing the type boxes can change the prediction a lot.")

# ------------------------- Results panel -------------------------
with st.expander("Model performance (held-out test set, 206 compounds)"):
    st.table(pd.DataFrame({
        "Stage 1 (descriptors)": ["0.58", "0.56", "0.78"],
        "Stage 2 (+ fingerprints)": ["0.67", "0.66", "0.86"],
    }, index=["F1", "Recall", "ROC-AUC"]))
    st.caption("These come from the notebook. The app itself is trained on all 1,030 "
               "compounds, so predictions for known pesticides are not a test of accuracy.")

# ------------------------- Bottom picture -------------------------
inline_bee("bee_path.png", 320, center=True)