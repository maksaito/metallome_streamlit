# Metalloproteomic Viewer 3D plot
# Mak Saito 2025 
# Pseudomonas aeruginosa, oxic and anoxic treatments
# https://github.com/maksaito/metallome_streamlit
# https://www.biorxiv.org/content/10.1101/2025.01.15.633287v2

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

# Check dataset
print(m_oxic.head())

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

# Create Dropdown Menu for contour height
contour_height_dropdown = st.sidebar.selectbox(
    'Contour Height:',
    [0.8, 1, 1.2],
    index=1  # default to 1
)

# Create Dropdown Menu for dataset type (oxic or anoxic)
dataset_type_dropdown = st.sidebar.selectbox(
    'Dataset Type:',
    ['Oxic', 'Anoxic'],
    index=0  # default to Oxic
)

# Define Plotting Function
def plot_metal(metal, contour_height_factor, dataset_type):
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
    contour_height = z_min + contour_height_factor * (z_max - z_min)

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
        title=f'{metal} ({dataset_type})',
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
st.write(f'Metalloproteome of *Pseudomonas aeruginosa*: {dataset_type_dropdown}')

# Plot the selected metal with the selected contour height factor and dataset type
plot_metal(metal_dropdown, contour_height_dropdown, dataset_type_dropdown)

# Button to display the dataframe
if st.button('Show Dataframe'):
    st.dataframe(m_oxic if dataset_type_dropdown == 'Oxic' else m_anoxic)

st.write('From Saito and McIlvin 2025 https://www.biorxiv.org/content/10.1101/2025.01.15.633287v2')


