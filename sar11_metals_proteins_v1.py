import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import streamlit as st


# Page title
st.markdown("<h2 style='text-align: center; font-size: 36px;'>Metalloproteome of <em>Pelagibacter strain SAR11</em></h2>", unsafe_allow_html=True)
st.write('Data produced by the Saito Laboratory in collaboration with the Giovannoni lab (in prep)')

# Reading the metals data
m_oxic = pd.read_csv('sar11_metals.csv')

# Setting the first column (Sample_ID) as the index
m_oxic.set_index('Sample_ID', inplace=True)

# Extract metal columns (all columns except 'SE Fraction' and 'AE Fraction')
metal_columns = m_oxic.columns.difference(['SE Fraction', 'AE Fraction'])

# Create Dropdown Menu for metals
metal = st.selectbox('Select Metal:', sorted(metal_columns), index=sorted(metal_columns).index('56Fe'))

# Define Plotting Function for metals
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

# Load the proteins data
pr_o = pd.read_csv('sar11_proteins_v2.csv')

# Function to parse the Annotation column
def parse_annotation(annotation):
    parameters = annotation.split("] [")
    parameters[0] = parameters[0][1:]
    parameters[-1] = parameters[-1][:-1]
    
    parsed_data = {}
    for param in parameters:
        key, value = param.split("=")
        parsed_data[key] = value
    
    return parsed_data

# Apply the parsing function to the Annotation column
parsed_annotations = pr_o['Annotation'].apply(parse_annotation)

# Convert the parsed annotations into a DataFrame
parsed_df = pd.DataFrame(parsed_annotations.tolist())

# Concatenate the parsed annotations DataFrame with the original DataFrame
result_df = pd.concat([parsed_df, pr_o], axis=1)

# Set the 'locus_tag' column as the index
result_df.set_index('locus_tag', inplace=True)

# Remove specified columns
columns_to_remove = ['gene', 'db_xref', 'protein_id', 'location', 'gbkey', 'pseudo', 'partial', 'Annotation', 'Accession Number', 'Molecular Weight', 'protein']
modified_df = result_df.drop(columns=columns_to_remove)

# Transpose the DataFrame and reset the index
flip_df = modified_df.transpose().reset_index()

# Rename the column 'index' to 'AE-SE' and split it into 'AE Fraction' and 'SE Fraction'
flip_df.rename(columns={'index': 'AE-SE'}, inplace=True)
flip_df[['AE Fraction', 'SE Fraction']] = flip_df['AE-SE'].str.split('-', expand=True)

# Move the new columns to the far left
cols = ['AE Fraction', 'SE Fraction'] + flip_df.columns[:-2].tolist()
flip_df = flip_df[cols]

# Extract gene IDs (also the index)
pa_columns = result_df.index.tolist()

# Extract annotations
processed_annotations = result_df['protein'].tolist()

# Change name of DataFrame
pr_o = flip_df

# Convert relevant columns to numeric types if necessary
pr_o = pr_o.apply(pd.to_numeric, errors='ignore')

# Create a dictionary to map protein columns to their combined text
protein_annotation_dict = dict(zip(pa_columns, processed_annotations))

# Create Dropdown Menu with combined text
protein_combined = st.selectbox(
    'Select Protein:',
    sorted([f"{pa_id} - {annotation}" for pa_id, annotation in zip(pa_columns, processed_annotations)])
)

# Define Plotting Function for proteins
def plot_data(protein_combined):
    # Extract the original protein column name from the combined text
    protein = next(key for key, value in protein_annotation_dict.items() if f"{key} - {value}" == protein_combined)
    
    # Creating a pivot table to reshape the protein data
    protein_pivot_table = pr_o.pivot(index='SE Fraction', columns='AE Fraction', values=protein)
    X_protein, Y_protein = np.meshgrid(protein_pivot_table.columns, protein_pivot_table.index)
    Z_protein = protein_pivot_table.values

    z_min_protein, z_max_protein = np.min(Z_protein), np.max(Z_protein)
    contour_height_protein = z_min_protein + 1 * (z_max_protein - z_min_protein)

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
        width=1150,
        height=600,
        scene=dict(
            xaxis_title='AE Fraction',
            yaxis_title='SE Fraction',
            zaxis_title=f'{protein}',
            xaxis=dict(showgrid=False, backgroundcolor='lightgrey'),
            yaxis=dict(showgrid=False, backgroundcolor='darkgrey'),
            zaxis=dict(showgrid=False, backgroundcolor='whitesmoke'),
            camera_eye=dict(x=1.5, y=1.5, z=1)
        ),
        title=f"Protein: {protein_combined}"
    )
    
    # Display the plot
    st.plotly_chart(fig)

# Plot the selected protein
plot_data(protein_combined)


# Table of protein data

# Remove extra columns except protein annotation (protein)
columns_to_remove2 = ['gene', 'db_xref', 'protein_id', 'location', 'gbkey', 'pseudo', 'partial', 'Annotation', 'Accession Number', 'Molecular Weight']
modified_df_annot = result_df.drop(columns=columns_to_remove2)

# Exclude string columns and columns with NaNs
numeric_df = modified_df_annot.select_dtypes(include=[np.number]).dropna(axis=1, how='any')

# Find the maximum value along each row and return the column header of that location
modified_df_annot['Max_Column'] = numeric_df.idxmax(axis=1)

# Insert the new column as the second column in the DataFrame
cols = list(modified_df_annot.columns)
cols.insert(1, cols.pop(cols.index('Max_Column')))
modified_df_annot = modified_df_annot[cols]

# Sort the DataFrame on the Max_Column in ascending order
modified_df_annot = modified_df_annot.sort_values(by='Max_Column')

# Streamlit app
st.write("Protein Dataset with Maxima locations")

# Search functionality
search_term = st.text_input("Search for a protein annotation:")
if search_term:
    filtered_df = modified_df_annot[modified_df_annot['protein'].str.contains(search_term, case=False, na=False)]
    st.dataframe(filtered_df)
else:
    st.dataframe(modified_df_annot)

# Option to download the data as a CSV file
st.download_button(
    label="Download data as CSV",
    data=modified_df_annot.to_csv(index=False).encode('utf-8'),
    file_name='protein_data.csv',
    mime='text/csv',
)

# 2D plot
# Create Dropdown Menu for metals

# Define Plotting Function
def plot_metal(metal):
    # Creating a pivot table to reshape the data
    pivot_table = m_oxic.pivot(index='SE Fraction', columns='AE Fraction', values=metal)

    # Extracting the x, y, and z data from the pivot table
    X, Y = np.meshgrid(pivot_table.columns, pivot_table.index)
    Z = pivot_table.values

    # Define the height for the contour plot based on the selected factor
    z_min, z_max = np.min(Z), np.max(Z)
    
    # Add subplot
    fig = make_subplots(rows=1, cols=1, subplot_titles=(f"Metal (oxic): {metal}",))

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

    # Update layout for better visualization and separate legends
    fig.update_layout(
        width=1150,
        height=600,
        xaxis_title='AE Fraction',
        yaxis_title='SE Fraction'
    )

    # Display the plot
    st.plotly_chart(fig)

# Plot the selected metal
plot_metal(metal)
