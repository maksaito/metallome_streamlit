# Metalloproteomic Viewer 3D plot
# Mak Saito 2025 
# Pseudomonas aeruginosa, oxic and anoxic treatments
# https://github.com/maksaito/metallome_streamlit
# https://www.biorxiv.org/content/10.1101/2025.01.15.633287v2

### import packages
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Load Metal Data, oxic and anoxic treatments from csv files, available on Zenodo
m_oxic = pd.read_csv('2025_0214_pao_metals_oxic.csv')
m_anoxic = pd.read_csv('2025_0214_pao_metals_anoxic.csv')

# Setting the first column (Sample_ID) as the index
m_oxic.set_index('Sample_ID', inplace=True)
m_anoxic.set_index('Sample_ID', inplace=True)

# Extract metal columns (all columns except 'SE Fraction' and 'AE Fraction')
metal_columns = m_oxic.columns.difference(['SE Fraction', 'AE Fraction'])

# Import dataset
@st.cache_data
def get_data():
    return m_oxic, m_anoxic

# Create Dropdown Menu for metals
metal_dropdown = st.sidebar.selectbox(
    'Metal:',
    sorted(metal_columns),  # Sort the metal columns
    index=sorted(metal_columns).index('Fe 56')  # default to Fe 56
)

# Create Dropdown Menu for dataset type (oxic or anoxic)
dataset_type_dropdown = st.sidebar.selectbox(
    'Dataset Type:',
    ['Oxic', 'Anoxic'],
    index=0  # default to Oxic
)

# Define Plotting Function
def plot_metal(metal, dataset_type):
    if dataset_type == 'Oxic':
        data = m_oxic
    else:
        data = m_anoxic

    # Creating a pivot table to reshape the data
    pivot_table = data.pivot(index='SE Fraction', columns='AE Fraction', values=metal)

    # Extracting the x, y, and z data from the pivot table
    X, Y = np.meshgrid(pivot_table.columns, pivot_table.index)
    Z = pivot_table.values
    
    # Define the height for the contour plot based on the selected factor
    z_min, z_max = np.min(Z), np.max(Z)
    contour_height = z_min + 1 * (z_max - z_min)

    # Create a single 3D plot 
    fig = go.Figure()

    # Add surface plot 
    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Hot', showscale=False))

    # Add contour plot at a constant height above the surface
    fig.add_trace(go.Surface(
        z=np.ones_like(Z) * contour_height,
        x=X,
        y=Y,
        surfacecolor=Z,
        colorscale='Hot',
        showscale=False,
        opacity=0.6,  # Set transparency
        contours=dict(
            x=dict(show=False, highlight=False),  # Remove grid lines
            y=dict(show=False, highlight=False),  # Remove grid lines
            z=dict(show=True, project=dict(z=True))
        )
    ))

    # Set figure size, titles, labels, and shades of grey for backgrounds
    fig.update_layout(
        #title=f'{metal} ({dataset_type})',
        width=800,  # Set the width of the figure
        height=600,  # Set the height of the figure
        scene=dict(
            xaxis_title='AE Fraction',
            yaxis_title='SE Fraction',
            zaxis_title=f'{metal}',
            xaxis=dict(showgrid=False, backgroundcolor='lightgrey', range=[100, 600]),  # remove 600-1000 data
            yaxis=dict(showgrid=False, backgroundcolor='darkgrey'),
            zaxis=dict(showgrid=False, backgroundcolor='whitesmoke'),
            camera_eye=dict(x=1.5, y=1.5, z=1)
        )
    )

    # Display the plot
    st.plotly_chart(fig)

# Display the selected metal 
st.title(f'Metalloproteome of *Pseudomonas aeruginosa*: {dataset_type_dropdown}')

# Plot the selected metal with the selected dataset type
plot_metal(metal_dropdown, dataset_type_dropdown)

# Button to display the dataframe
if st.button('Show Dataframe'):
    st.dataframe(m_oxic if dataset_type_dropdown == 'Oxic' else m_anoxic)


# Protein Viewer 3D plot
st.write("Display selected protein (currently oxic treatment only)")

# Load protein data
@st.cache_data
def load_protein_data():
    pr_full = pd.read_csv('2025_0214_pao_proteins_oxic_pivot_processed.csv')
    pr_full.set_index('PA_ID', inplace=True)
    return pr_full

pr_full = load_protein_data()

# Set 'PA_ID' as the index and drop unnecessary columns, remove extra columns
# Transpose protein file to machine readable format 
# version with annotations in second line
pr = pr_full.drop(columns=['Molecular Weight', 'Annotation']).transpose().reset_index()
# clean version
pr = pr_full.drop(columns=['Molecular Weight', 'Annotation', 'Processed Annotation']).transpose().reset_index()

# organize 2D matrix information
pr.rename(columns={'index': 'AE-SE'}, inplace=True)
pr[['AE Fraction', 'SE Fraction']] = pr['AE-SE'].str.split('-', expand=True)

# Move the new columns to the far left
cols = ['AE Fraction', 'SE Fraction'] + pr.columns[:-2].tolist()
pr = pr[cols]

# extract gene IDs (also the index)
pa_columns = pr_full.index.tolist()

# extract annotations
processed_annotations = pr_full['Processed Annotation'].tolist()

# change name of df 
pr_o = pr

# Convert relevant columns to numeric types if necessary 
pr_o = pr_o.apply(pd.to_numeric, errors='ignore')

# Create a dictionary to map protein columns to their combined text
protein_annotation_dict = dict(zip(pa_columns, processed_annotations))

# Sidebar menu for PA_ID selection
protein_combined = st.sidebar.selectbox(
    "Select Protein:",
    options=sorted([f"{pa_id} - {annotation}" for pa_id, annotation in zip(pa_columns, processed_annotations)])
)

# Define Plotting Function
def plot_protein(protein_combined):
    # Extract the original protein column name from the combined text
    protein = next(key for key, value in protein_annotation_dict.items() if f"{key} - {value}" == protein_combined)
    
    # Creating a pivot table to reshape the protein data
    protein_pivot_table = pr_o.pivot(index='SE Fraction', columns='AE Fraction', values=protein)
    X_protein, Y_protein = np.meshgrid(protein_pivot_table.columns, protein_pivot_table.index)
    Z_protein = protein_pivot_table.values

    z_min_protein, z_max_protein = np.min(Z_protein), np.max(Z_protein)
    contour_height_protein = z_min_protein + 1 * (z_max_protein - z_min_protein)

    # Get the combined text for the selected protein
    combined_text = protein_annotation_dict.get(protein, "")

    # Creating the plot with Plotly
    fig = go.Figure()

    # Add protein surface plot with a hotter color palette
    fig.add_trace(go.Surface(z=Z_protein, x=X_protein, y=Y_protein, colorscale='Hot', showscale=False))

    # Add protein contour plot at a constant height above the surface
    fig.add_trace(go.Surface(
        z=np.ones_like(Z_protein) * contour_height_protein,
        x=X_protein,
        y=Y_protein,
        surfacecolor=Z_protein,
        colorscale='Hot',
        showscale=False,
        opacity=0.6,
        contours=dict(
            x=dict(show=False, highlight=False),
            y=dict(show=False, highlight=False),
            z=dict(show=True, project=dict(z=True))
        )
    ))

    # Update layout for the plot and set figure size
    fig.update_layout(
        title=f"Protein: {protein_combined}",
        title_font_size=20,  # Adjust the font size to match the metal plot title
        width=1150,  # Set the width of the figure
        height=600,  # Set the height of the figure
        scene=dict(
            xaxis_title='AE Fraction',
            yaxis_title='SE Fraction',
            zaxis_title=f'{protein}',
            xaxis=dict(showgrid=False, backgroundcolor='lightgrey'),
            yaxis=dict(showgrid=False, backgroundcolor='darkgrey'),
            zaxis=dict(showgrid=False, backgroundcolor='whitesmoke'),
            camera_eye=dict(x=1.5, y=1.5, z=1)
        )
    )
    
    # Display the plot
    st.plotly_chart(fig)

# Plot the data based on user selection
plot_protein(protein_combined)

# Protein Table with revised index and remove the wordy "annotation" column"
# Combine the index and 'Processed Annotation' column to create a new index
pr_full['New_Index'] = pr_full.index + '-' + pr_full['Processed Annotation']
# Set the new index
pr_full.set_index('New_Index', inplace=True)
# Remove the 'Annotation' column
pr_full.drop(columns=['Annotation'], inplace=True)

st.write('Protein Table')
st.dataframe(pr_full)

st.write('From Saito and McIlvin 2025 https://www.biorxiv.org/content/10.1101/2025.01.15.633287v2')

