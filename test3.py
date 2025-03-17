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

# Define Plotting Function
def plot_metal(metal):
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
    z_min_ao, z_max_ao = np.min(Z_ao), np.max(Z_ao)
    
    # Add subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Metal (oxic): {metal}", f"Metal (anoxic): {metal}"))

    # Add contour plot for oxic metals without colorbar
    fig.add_trace(go.Contour(
        z=Z,
        x=X[0],
        y=Y[:, 0],
        colorscale='Hot',
        contours=dict(
            start=z_min,
            end=z_max,
            size=(z_max - z_min) / 10,
            coloring='heatmap'
        ),
        showscale=False  # Remove colorbar
    ), row=1, col=1)

    # Add contour plot for anoxic metals without colorbar
    fig.add_trace(go.Contour(
        z=Z_ao,
        x=X_ao[0],
        y=Y_ao[:, 0],
        colorscale='Hot',
        contours=dict(
            start=z_min_ao,
            end=z_max_ao,
            size=(z_max_ao - z_min_ao) / 10,
            coloring='heatmap'
        ),
        showscale=False  # Remove colorbar
    ), row=1, col=2)

    # Update layout for better visualization and separate legends
    fig.update_layout(
        width=1150,
        height=600,
        xaxis_title='AE Fraction',
        yaxis_title='SE Fraction'
    )

    # Display the plot
    st.plotly_chart(fig)

# Display the selected metal
st.write(f'Selected metal: {metal_dropdown}')

# Plot the selected metal
plot_metal(metal_dropdown)

# Display the dataframe
st.dataframe(m_oxic)