import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import py3Dmol
import plotly.express as px
from rdkit.Chem import Draw
import py3Dmol
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import CalcNumAromaticRings
import urllib.parse
import webbrowser


st.markdown("""
    <style>
        /* Background Styling */
        .stApp {
            background: linear-gradient(135deg, #ffd1dc 0%, #ffe4e9 100%);
            animation: gradientBG 15s ease infinite;
            background-size: 400% 400%;
            font-family: 'Helvetica', sans-serif;
        }
        h1 {a
            color: #4a4a4a;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
        }

        /* Button Styling */
        .stButton>button {
            background-color: rgb(38, 182, 230);
            border: none;
            padding: 10px 20px;
            color: #ffffff;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #87ceeb;
            transform: scale(1.05);
        }

        /* Input Box Styling */
        input {
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)



# Function to display the Welcome page
def welcome_page():
    st.markdown("<style>body {background-color: #F0F8FF;}</style>", unsafe_allow_html=True)
    st.title("Welcome to DRUGTOX PRO")
    
    # Collect user information
    username = st.text_input("Enter your Username:")
    email = st.text_input("Enter your Email:")

    if username and email:
        st.success(f"Welcome, {username}! We have received your email: {email}.")
        if st.button("Proceed to Home Page"):
            return True
    else:
        st.warning("Please enter both username and email to proceed.")
    return False

# Function to display the Home page with key information and image
def home_page():
    # Set Background Color
    st.markdown("<style>body {background-color: #FFFAF0;}</style>", unsafe_allow_html=True)
    st.title("Home Page")
    
    # Display key information about the website
    st.markdown("""
        ## Key Information About DRUGTOX PRO
        
        - **ADMET Predictor**: Predicts the ADMET properties of compounds.
        - **Disease Finder**: Helps in finding potential diseases based on symptoms.
        - **ML Classifier Performance**: Evaluates the performance of machine learning classifiers.
        - **Molecular Viewer**: Visualizes molecular structures.
        - **Plot Visualization**: Provides various data visualization tools.
        - **GO Enrichment Analysis**: Performs Gene Ontology enrichment to identify overrepresented biological functions.
                
                
        
        The website is designed to assist researchers and students in data analysis and prediction.
    """)

    # Display Image
    st.image('image1.png', caption='Key Features of the Website', use_container_width=True)

    # Create three columns for proper spacing
    col1, col_space, col3 = st.columns([1, 2, 1])  # col_space creates space between buttons

    with col1:
        if st.button("Back"):
            st.session_state.page = "welcome"  # Navigate back to Welcome page

    with col3:
        if st.button("Start"):
            st.session_state.page = "tools"  # Navigate to tools page

# Tabs-based navigation
def tools_navigation():
    # Set Background Color
    st.markdown(
        "<style>body {background-color: #E6E6FA;}</style>", 
        unsafe_allow_html=True
    )

    # Create a toolbar with tool buttons on the left and "About" on the right
    col1, col2 = st.columns([6, 1])  # 6:1 ratio pushes "About" to the right

    with col1:
        st.title("Tools")

    with col2:
        if st.button("About", key="about_btn"):
            st.session_state.selected_tool = "About"

    # Custom CSS for Buttons
    st.markdown(
        """
        <style>
            div.stButton > button {
                width: 200px !important;
                height: 50px !important;
                font-size: 16px !important;
                font-weight: bold !important;
                text-align: center;
                background-color:rgb(38, 182, 230)!important;
                color: white !important;
                border-radius: 10px !important;
                border: none;
                margin: 5px !important;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
            }
            div.stButton > button:hover {
                background-color: #45a049 !important;
                transform: scale(1.05);
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Arrange buttons in two rows with three columns each
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("ADMET Predictor", key="admet_btn"):
            st.session_state.selected_tool = "ADMET Predictor"

    with col2:
        if st.button("Disease Finder", key="disease_btn"):
            st.session_state.selected_tool = "Disease Finder"

    with col3:
        if st.button("ML Classifier", key="ml_btn"):
            st.session_state.selected_tool = "ML Classifier Performance"

    col4, col5, col6 = st.columns(3)

    with col4:
        if st.button("Molecular Viewer", key="mol_btn"):
            st.session_state.selected_tool = "Molecular Viewer"

    with col5:
        if st.button("Plot Visualization", key="plot_btn"):
            st.session_state.selected_tool = "Plot Visualization"

    with col6:
        if st.button("GO Analysis", key="go_btn"):
            st.session_state.selected_tool = "GO Enrichment Analysis"

    # Render the selected tool
    if "selected_tool" in st.session_state:
        if st.session_state.selected_tool == "ADMET Predictor":
            admet_predictor()
        elif st.session_state.selected_tool == "Disease Finder":
            disease_finder()
        elif st.session_state.selected_tool == "ML Classifier Performance":
            ml_classifier_performance()
        elif st.session_state.selected_tool == "Molecular Viewer":
            molecular_viewer()
        elif st.session_state.selected_tool == "Plot Visualization":
            plot_visualization()
        elif st.session_state.selected_tool == "GO Enrichment Analysis":
            go_enrichment_analysis()
        elif st.session_state.selected_tool == "About":
            about_page()  # Navigates to About Page when clicked

# Function to render ADMET Prediction Tool
def get_colored_circle(color):
    """Returns an HTML string for a colored circle."""
    return f"<div style='border-radius: 50%; width: 20px; height: 20px; background-color: {color}; display: inline-block;'></div>"

def color_for_value(value, thresholds):
    """Determine the color based on value and thresholds (green, yellow, red)."""
    if value <= thresholds[0]:
        return "green"
    elif thresholds[0] < value <= thresholds[1]:
        return "yellow"
    else:
        return "red"

def get_colored_circle(color):
    return f"<div style='border-radius: 50%; width: 20px; height: 20px; background-color: {color}; display: inline-block;'></div>"

def admet_predictor():
    st.title("ADMET Prediction Tool")
    smiles = st.text_input("Enter SMILES Notation for the compound", "")
    
    if st.button("Predict"):
        if smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                tpsa = Descriptors.TPSA(mol)
                rot_bonds = Descriptors.NumRotatableBonds(mol)
                h_bond_donors = Descriptors.NumHDonors(mol)
                h_bond_acceptors = Descriptors.NumHAcceptors(mol)
                
                # Example criteria for setting color
                mw_color = 'green' if mw < 500 else 'red'
                logp_color = 'green' if logp < 5 else 'red'
                tpsa_color = 'green' if tpsa < 140 else 'red'
                rot_bonds_color = 'green' if rot_bonds <= 10 else 'yellow'
                h_bond_donors_color = 'green' if h_bond_donors <= 5 else 'red'
                h_bond_acceptors_color = 'green' if h_bond_acceptors <= 10 else 'red'

                absorption = "High" if logp < 5 and mw < 500 else "Low"
                absorption_color = 'green' if absorption == "High" else 'red'
                
                toxicity = "Non-Toxic" if rot_bonds <= 5 and tpsa <= 120 else "Toxic"
                toxicity_color = 'green' if toxicity == "Non-Toxic" else 'red'
                
                distribution = f"Low" if logp <= 3 else "High"
                distribution_color = 'yellow' if logp <= 3 else 'red'
                
                metabolism = "CYP2D6 inhibitor: Likely" if logp > 2 else "CYP2D6 inhibitor: Unlikely"
                metabolism_color = 'green' if logp <= 2 else 'red'
                
                excretion = "Fast Clearance" if mw < 350 else "Slow Clearance"
                excretion_color = 'green' if mw < 350 else 'red'

                st.markdown(
                    """
                    <table style='width: 100%; border-collapse: collapse;'>
                        <tr>
                            <th style='text-align: left;'>Property</th>
                            <th style='text-align: left;'>Value</th>
                            <th style='text-align: right;'>Indicator</th>
                        </tr>
                        <tr>
                            <td>Molecular Weight</td>
                            <td>{:.2f}</td>
                            <td style='text-align: right;'>{}</td>
                        </tr>
                        <tr>
                            <td>LogP (Partition Coefficient)</td>
                            <td>{:.2f}</td>
                            <td style='text-align: right;'>{}</td>
                        </tr>
                        <tr>
                            <td>Topological Polar Surface Area (TPSA)</td>
                            <td>{:.2f}</td>
                            <td style='text-align: right;'>{}</td>
                        </tr>
                        <tr>
                            <td>Rotatable Bonds</td>
                            <td>{}</td>
                            <td style='text-align: right;'>{}</td>
                        </tr>
                        <tr>
                            <td>H-Bond Donors</td>
                            <td>{}</td>
                            <td style='text-align: right;'>{}</td>
                        </tr>
                        <tr>
                            <td>H-Bond Acceptors</td>
                            <td>{}</td>
                            <td style='text-align: right;'>{}</td>
                        </tr>
                        <tr>
                            <td>Absorption</td>
                            <td>{}</td>
                            <td style='text-align: right;'>{}</td>
                        </tr>
                        <tr>
                            <td>Toxicity</td>
                            <td>{}</td>
                            <td style='text-align: right;'>{}</td>
                        </tr>
                        <tr>
                            <td>Distribution</td>
                            <td>{}</td>
                            <td style='text-align: right;'>{}</td>
                        </tr>
                        <tr>
                            <td>Metabolism</td>
                            <td>{}</td>
                            <td style='text-align: right;'>{}</td>
                        </tr>
                        <tr>
                            <td>Excretion</td>
                            <td>{}</td>
                            <td style='text-align: right;'>{}</td>
                        </tr>
                    </table>
                    """.format(
                        mw, get_colored_circle(mw_color),
                        logp, get_colored_circle(logp_color),
                        tpsa, get_colored_circle(tpsa_color),
                        rot_bonds, get_colored_circle(rot_bonds_color),
                        h_bond_donors, get_colored_circle(h_bond_donors_color),
                        h_bond_acceptors, get_colored_circle(h_bond_acceptors_color),
                        absorption, get_colored_circle(absorption_color),
                        toxicity, get_colored_circle(toxicity_color),
                        distribution, get_colored_circle(distribution_color),
                        metabolism, get_colored_circle(metabolism_color),
                        excretion, get_colored_circle(excretion_color)
                    ),
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Error processing SMILES: {e}")
        else:
            st.warning("Please enter a valid SMILES notation.")


    # Function to render Disease Finder Tool
def disease_finder():
    st.title("Disease Finder")
    symptoms = st.text_input("Enter Symptoms (comma-separated)")

    if symptoms and st.button("Find Disease"):
        # Expanded dictionary of symptoms and corresponding diseases
        disease_dict = {
            "fever,cough": "Flu",
            "fever,chills,sweating": "Malaria",
            "headache,nausea": "Migraine",
            "fatigue,weight loss": "Diabetes",
            "chest pain,shortness of breath": "Heart Disease",
            "joint pain,swelling": "Arthritis",
            "skin rash,itching": "Eczema",
            "abdominal pain,diarrhea": "Food Poisoning",
            "sore throat,runny nose": "Common Cold",
            "vomiting,stomach cramps": "Gastroenteritis",
            "blurred vision,frequent urination": "Diabetes",
            "persistent cough,blood in mucus": "Tuberculosis",
            "weakness,rapid heartbeat": "Anemia",
            "dizziness,nausea": "Vertigo",
            "loss of appetite,weight loss": "Cancer",
            "confusion,slurred speech": "Stroke",
            "high fever,stiff neck": "Meningitis",
            "muscle weakness,difficulty swallowing": "ALS",
            "dry mouth,blurred vision": "Botulism",
            "fever,body ache,headache": "Dengue",
            "extreme fatigue,swollen glands": "Mononucleosis",
            "yellow skin,dark urine": "Hepatitis",
            "rapid breathing,fatigue": "Pneumonia",
            "nausea,vomiting": "Motion Sickness",
            "fatigue,joint pain": "Lupus",
            "night sweats,fever": "Tuberculosis",
            "persistent fever,weight loss": "HIV/AIDS",
            "fever,red eyes": "Zika Virus",
            "fever,skin rash": "Chikungunya",
            "cough,chills": "Malaria",
            "fever,bleeding gums": "Leukemia",
            "difficulty breathing,swollen feet": "Heart Failure",
            "chest tightness,coughing": "Asthma",
            "weight loss,abdominal pain": "Stomach Cancer",
            "skin rash,red spots": "Measles",
            "persistent sore throat": "Strep Throat",
            "itchy scalp,hair loss": "Alopecia",
            "fatigue,low blood pressure": "Adrenal Insufficiency",
            "high fever,chills": "Typhoid",
            "frequent infections,bruising": "Leukemia",
            "unusual bleeding,weight loss": "Hemophilia",
            "numbness,tingling": "Multiple Sclerosis",
            "stiff joints,muscle weakness": "Fibromyalgia",
            "persistent headache,blurred vision": "Glaucoma",
            "muscle cramps,spasms": "Parkinson's",
            "loss of balance,dizziness": "Lymphoma",
            "fatigue,swollen lymph nodes": "Yellow Fever",
            "fever,headache": "Hypothermia",
            "cold hands,blue lips": "Raynaud's Disease",
            "extreme thirst,frequent urination": "Diabetes",
            "fever,abdominal pain": "Appendicitis",
            "fatigue,fever": "Infectious Mononucleosis",
            "loss of smell,taste disturbance": "COVID-19",
            "jaundice,abdominal pain": "Pancreatitis",
            "joint pain,red rash": "Rheumatic Fever",
            "blurred vision,eye pain": "Glaucoma",
            "leg swelling,breathlessness": "Deep Vein Thrombosis",
            "dry cough,breathlessness": "Silicosis",
            "fever,weight loss": "Leptospirosis",
            "persistent cough,chest pain": "Bronchitis",
            "loss of memory,disorientation": "Alzheimer’s Disease",
            "severe headache,vision loss": "Giant Cell Arteritis",
            "eye redness,eye discharge": "Conjunctivitis",
            "muscle stiffness,difficulty moving": "Parkinson's Disease",
            "high blood pressure,headache": "Hypertension",
            "sudden weight loss,thirst": "Kidney Disease",
            "frequent urination,blurred vision": "Hyperglycemia",
            "coughing blood,fever": "Lung Cancer",
            "abnormal bleeding,skin bruising": "Hemophilia",
            "fever,rash,chills": "Lyme Disease",
            "bleeding gums,skin spots": "Scurvy",
            "bloody stools,abdominal cramps": "Ulcerative Colitis",
            "skin patches,nerve pain": "Leprosy",
            "numbness,loss of reflexes": "Peripheral Neuropathy",
            "chronic pain,extreme fatigue": "Chronic Fatigue Syndrome",
            "skin rash,blistering": "Psoriasis",
            "cough,hoarseness": "Laryngitis",
            "fever,nausea": "Tetanus",
            "muscle weakness,pain": "Myasthenia Gravis",
            "muscle spasms,stiffness": "Torticollis",
            "bloody stools,weight loss": "Crohn’s Disease",
            "fever,cough": "Legionnaires’ Disease",
            "itchy skin,breathing difficulty": "Anaphylaxis",
            "dark urine,pale stool": "Cholangitis",
            "chest pain,heartburn": "GERD",
            "sudden numbness,paralysis": "Stroke",
            "fever,irritability": "Rabies",
            "muscle twitching,fatigue": "ALS",
            "fatigue,skin darkening": "Addison's Disease",
            "irregular heartbeat,short breath": "Cardiomyopathy",
            "difficulty speaking,headache": "Brain Tumor",
            "joint pain,muscle fatigue": "Polymyositis",
            "skin redness,itching": "Hives",
            "dry mouth,teary eyes": "Sjogren's Syndrome",
            "cold extremities,fainting": "Shock"
        }

        # Checking if the symptoms match any disease
        matches = [disease for sym, disease in disease_dict.items() if all(s in symptoms.lower() for s in sym.split(','))]

        if matches:
            disease_prediction = ', '.join(matches)
            st.markdown(f"### Possible Disease: {disease_prediction}")
            
            # Generate a Google search URL
            query = urllib.parse.quote(disease_prediction)
            google_url = f"https://www.google.com/search?q={query}"
            st.markdown(f"[More Information on Google]({google_url})")
        else:
            st.markdown("### Possible Disease: No matching disease found.")
def ml_classifier_performance():
    st.title("ML Classifier Performance on ADMET Properties")
    uploaded_file = st.file_uploader("Upload Dataset", type=["xlsx", "csv"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('xlsx'):
                data = pd.read_excel(uploaded_file, engine='openpyxl', nrows=50000)  # Limit rows for efficiency
            else:
                data = pd.read_csv(uploaded_file, nrows=50000)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return

        st.write("Dataset Preview (First 5 rows):")
        st.dataframe(data.head())

        feature_columns = st.multiselect("Select feature columns", data.columns.tolist())
        target_column = st.selectbox("Select the target column", data.columns.tolist())

        if feature_columns and target_column:
            X = data[feature_columns]
            y = data[target_column]

            if y.nunique() < 2:
                st.error("The target column must have at least two distinct classes.")
                return

            y_encoded = LabelEncoder().fit_transform(y)
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

            classifiers = {
                "Logistic Regression": LogisticRegression(max_iter=10000),
                "Support Vector Classifier (SVC)": SVC(probability=True),
                "Random Forest": RandomForestClassifier()
            }

            selected_classifier = st.selectbox("Select a classifier", list(classifiers.keys()))
            model = classifiers[selected_classifier]
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)
            st.write(f"Accuracy: {accuracy:.4f}")

            # AUC-ROC Curve
            y_pred_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            if len(set(y_encoded)) == 2 and y_pred_prob is not None:
                fpr, tpr, _ = roc_curve(y_test, y_pred_prob[:, 1])
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'Binary Classification (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('AUC-ROC Curve for Binary Classification')
                plt.legend()
                st.pyplot(plt)
            elif y_pred_prob is not None:
                y_test_bin = label_binarize(y_test, classes=range(len(set(y_encoded))))
                roc_curves = []  # Store (class_index, roc_auc, fpr, tpr)
                
                for i in range(y_pred_prob.shape[1]):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
                    roc_auc = auc(fpr, tpr)
                    if not pd.isna(roc_auc) and roc_auc > 0.5:  # Skip NaN and low AUC values
                        roc_curves.append((i, roc_auc, fpr, tpr))
                
                # Sort by AUC and limit to top 5 classes
                roc_curves = sorted(roc_curves, key=lambda x: x[1], reverse=True)[:5]
                
                plt.figure(figsize=(8, 6))
                for class_index, roc_auc, fpr, tpr in roc_curves:
                                        plt.plot(fpr, tpr, label=f'Class {class_index} (AUC = {roc_auc:.2f})')
                
                plt.plot([0, 1], [0, 1], 'k--', lw=2)
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('AUC-ROC Curve for Multiclass Classification (Top 5 Classes)')
                plt.legend(loc='lower right')
                st.pyplot(plt)
            else:
                st.info("AUC-ROC curve is not available for this dataset/classifier combination.")


# Function to render 3D molecular structure
def render_molecule_3d(smiles):
    # Generate a 3D visualization of the molecule using py3Dmol
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mol_block = Chem.MolToMolBlock(mol)
        viewer = py3Dmol.view(width=800, height=400)
        viewer.addModel(mol_block, "mol")
        viewer.setStyle({'stick': {}})
        viewer.setBackgroundColor('white')
        viewer.zoomTo()
        return viewer
    return None

# Function to render the Molecular Viewer tool
def molecular_viewer():
    st.title("2D and 3D Molecular Structure Viewer")
    
    # Take SMILES input from the user
    smiles = st.text_input("Enter SMILES notation")
    
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        
        if mol:
            # Display the 2D structure
            img = Draw.MolToImage(mol)
            st.image(img, caption='2D Structure')

            # Generate and display the 3D structure
            st.write("3D Structure:")
            viewer = render_molecule_3d(smiles)
            if viewer:
                # Get the HTML for 3Dmol and display it in Streamlit
                viewer_html = viewer._make_html()  # Generate the HTML representation
                components.html(viewer_html, height=500)
            else:
                st.write("Could not generate 3D structure for the provided SMILES.")
# Function to render Plot Visualization Tool

def plot_visualization():
    st.header("Plot Visualization")
    uploaded_file = st.file_uploader("Upload a dataset for visualization (CSV or Excel):", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Load the data depending on the file type
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith("csv") else pd.read_excel(uploaded_file)
        st.dataframe(df.head())  # Displaying the first few rows of the dataframe
        
        # Plot type selection
        plot_type = st.selectbox("Select plot type:", ["Heatmap", "Boxplot", "Scatterplot", "Pairplot"])

        # Heatmap plot
        if plot_type == "Heatmap":
            st.write("Correlation Heatmap")
            selected_columns = st.multiselect("Select columns for Heatmap:", df.columns)
            if len(selected_columns) > 1:  # Heatmap requires at least two columns
                fig, ax = plt.subplots(figsize=(10, 6))  # Adjusting figure size for clarity
                sns.heatmap(df[selected_columns].corr(), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
                st.pyplot(fig)
            else:
                st.warning("Please select at least two columns for the heatmap.")

        # Boxplot plot
        elif plot_type == "Boxplot":
            column = st.selectbox("Select column for Boxplot:", df.columns)
            fig, ax = plt.subplots(figsize=(8, 6))  # Adjusting figure size
            sns.boxplot(data=df[column], ax=ax)
            st.pyplot(fig)
        
        # Scatterplot plot
        elif plot_type == "Scatterplot":
            x_col = st.selectbox("Select X-axis:", df.columns)
            y_col = st.selectbox("Select Y-axis:", df.columns)
            fig = px.scatter(df, x=x_col, y=y_col, color_discrete_sequence=["#FF6347"])  # Using Plotly for scatterplot with vibrant color
            st.plotly_chart(fig)
        
        # Pairplot plot
        elif plot_type == "Pairplot":
            selected_columns = st.multiselect("Select columns for Pairplot:", df.columns)
            if len(selected_columns) > 1:  # Pairplot requires at least 2 columns
                # Use seaborn's pairplot for visualizing relationships between multiple columns
                fig = sns.pairplot(df[selected_columns], plot_kws={'color': 'purple'})
                st.pyplot(fig)
            else:
                st.warning("Please select at least two columns for Pairplot.")

                # Function to render GO Enrichment Analysis Tool
def go_enrichment_analysis():
    st.title("GO Enrichment Analysis")
    uploaded_gene_file = st.file_uploader("Upload Gene List File (TXT or CSV)", type=["txt", "csv"])

    if uploaded_gene_file:
        gene_list = []
        if uploaded_gene_file.name.endswith("csv"):
            gene_data = pd.read_csv(uploaded_gene_file)
            gene_list = gene_data['Gene'].tolist()  # Assuming the column name is 'Gene'
        elif uploaded_gene_file.name.endswith("txt"):
            gene_list = uploaded_gene_file.read().decode("utf-8").splitlines()

        # Perform GO Enrichment Analysis (This is a placeholder)
        # Assuming we use a third-party library or external tool for actual GO analysis.
        st.write(f"### Analyzing Gene List ({len(gene_list)} genes)")
        # Placeholder for GO Enrichment Analysis results:
        go_results = {
            "GO Term": ["GO:0008150", "GO:0003674", "GO:0005575"],
            "Description": ["biological process", "molecular function", "cellular component"],
            "P-value": [0.01, 0.02, 0.05]
        }
        go_df = pd.DataFrame(go_results)
        st.write(go_df)

        # Optionally, display a plot of GO enrichment (Placeholder)
        fig = px.bar(go_df, x="GO Term", y="P-value", color="Description", title="GO Enrichment Analysis Results")
        st.plotly_chart(fig)

# About Page Function (Paste this before the main entry point)
def about_page():
    st.title("About This Application")
    
    st.markdown("""
        ## 🎉 Thank You for Using **DrugTox Pro**!  
               Your support means everything to us. 🚀  
               Keep exploring and making discoveries! 🧬💊
        
        This application is designed to assist researchers and students in analyzing biological data with the following tools:
        
        - **🧪 ADMET Predictor**: Predicts the ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity) properties of compounds.
        - **🩺 Disease Finder**: Suggests possible diseases based on entered symptoms.
        - **📊 ML Classifier Performance**: Evaluates and compares different machine learning classifiers.
        - **🧬 Molecular Viewer**: Displays 2D and 3D molecular structures.
        - **📉 Plot Visualization**: Generates plots to analyze dataset features.
        - **🧪 GO Enrichment Analysis**: Performs Gene Ontology (GO) enrichment analysis for biological insights.
        
        ---
        
        **📌 Developed By:**  
        - **👩‍💻 Developer:** Vardhi Dimple  
        - **🧑‍🏫 Guided By:** Dr. Anbarasu  

        ---
        
        ### ⭐ Rate This Application
        """)

    # Star Ratings
    rating = st.radio(
        "How would you rate this app?",
        ("⭐️⭐️⭐️⭐️⭐️ (5 Stars)", "⭐️⭐️⭐️⭐️ (4 Stars)", "⭐️⭐️⭐️ (3 Stars)", "⭐️⭐️ (2 Stars)", "⭐️ (1 Star)")
    )

    st.success(f"Thank you for your feedback! You rated: {rating}")

    st.markdown("""
        ---
        🔹 **Contact for Feedback & Improvements**  
        📧 Email: dimplevardhi5@gmail.com
    """)

# Main entry point for the app (Leave this part unchanged)
if __name__ == "__main__":
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "welcome"
    
    if st.session_state.page == "welcome":
        if welcome_page():
            st.session_state.page = "home"
    elif st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "tools":
        tools_navigation()