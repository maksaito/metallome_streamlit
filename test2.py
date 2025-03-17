### import packages
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# Define Plotting Function
def plot_metal(metal, contour_height_factor):
    # Creating a pivot table to reshape the data
    pivot_table = m_oxic.pivot(index='SE Fraction', columns='AE Fraction', values=metal)
    pivot_table_ao = m_anoxic.pivot(index='SE Fraction', columns='AE Fraction', values=metal)

    # Extracting the x, y, and z data from the pivot table
    X, Y = np.meshgrid(pivot_table.columns, pivot_table.index)
    Z = pivot_table.values
    X_ao, Y_ao = np.meshgrid(pivot_table_ao.columns, pivot_table_ao.index)
    Z_ao = pivot_table_ao.values
    
    # Define the height for the contour plot based on the selected factor
    z_min, z_max = np.min(Z), np.max(Z)
    contour_height = z_min + contour_height_factor * (z_max - z_min)
    z_min_ao, z_max_ao = np.min(Z_ao), np.max(Z_ao)
    contour_height_ao = z_min_ao + contour_height_factor * (z_max_ao - z_min_ao)

    # Create subplots 
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'surface'}, {'type': 'surface'}]], subplot_titles=(f"Metal (oxic): {metal}", f"Metal (anoxic): {metal}"))
  
    # Add surface plot 
    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Hot', showscale=False), row=1, col=1)
    fig.add_trace(go.Surface(z=Z_ao, x=X_ao, y=Y_ao, colorscale='Hot', showscale=False), row=1, col=2)

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
    ), row=1, col=1)

    fig.add_trace(go.Surface(
        z=np.ones_like(Z_ao) * contour_height_ao,
        x=X_ao,
        y=Y_ao,
        surfacecolor=Z_ao,
        colorscale='Hot',
        showscale=False,
        opacity=0.6,  # Adjusted transparency
        contours=dict(
            x=dict(show=False, highlight=False),  # Remove grid lines
            y=dict(show=False, highlight=False),  # Remove grid lines
            z=dict(show=True, project=dict(z=True))
        )
    ), row=1, col=2)
     
    # Set figure size, titles, labels, and shades of grey for backgrounds
    fig.update_layout(
        title=metal,
        width=1150,  # Set the width of the figure
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
    fig.update_layout(
        title=metal,
        scene2=dict(
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

# Display the selected metal and contour height
st.write(f'Selected metal: {metal_dropdown}')
st.write(f'Selected contour height factor: {contour_height_dropdown}')

# Plot the selected metal with the selected contour height factor
plot_metal(metal_dropdown, contour_height_dropdown)

# Display the dataframe
st.dataframe(m_oxic)