import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Page title
st.markdown("<h2 style='text-align: center; font-size: 36px;'>Metalloproteome of <em>Pelagibacter strain SAR11</em></h1>", unsafe_allow_html=True)
st.write('Data produced by the Saito Laboratory in collaboration with the Giovannoni lab (in prep)')

# Reading the data
m_oxic = pd.read_csv('sar11_metals.csv')

# Setting the first column (Sample_ID) as the index
m_oxic.set_index('Sample_ID', inplace=True)

# Extract metal columns (all columns except 'SE Fraction' and 'AE Fraction')
metal_columns = m_oxic.columns.difference(['SE Fraction', 'AE Fraction'])

 
# Create Dropdown Menu for metals
metal = st.selectbox('Select Metal:', sorted(metal_columns), index=sorted(metal_columns).index('56Fe'))

# Define Plotting Function
def plot_metal(metal):
    # Creating a pivot table to reshape the data
    pivot_table = m_oxic.pivot(index='SE Fraction', columns='AE Fraction', values=metal)

    # Extracting the x, y, and z data from the pivot table
    X, Y = np.meshgrid(pivot_table.columns, pivot_table.index)
    Z = pivot_table.values

    # Define the height for the contour plot based on the selected factor
    contour_height_factor = 1  # default contour height factor
    z_min, z_max = np.min(Z), np.max(Z)
    contour_height = z_min + contour_height_factor * (z_max - z_min)

    # Creating the surface plot with Plotly
    fig = go.Figure()

    # Add surface plot with a hotter color palette
    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Hot', showscale=False))

    # Add contour plot at a constant height above the surface
    fig.add_trace(go.Surface(
        z=np.ones_like(Z) * contour_height,
        x=X,
        y=Y,
        surfacecolor=Z,
        colorscale='Hot',
        showscale=False,
        opacity=0.6,  # Adjusted transparency
        contours=dict(
            x=dict(show=False, highlight=False),  # Remove grid lines
            y=dict(show=False, highlight=False),  # Remove grid lines
            z=dict(show=True, project=dict(z=True))
        )
    ))

    # Adding titles, labels, and setting different shades of grey for backgrounds
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

    # Display the plot
    st.plotly_chart(fig)

# Plot the selected metal
plot_metal(metal)
