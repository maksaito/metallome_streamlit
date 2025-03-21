import streamlit as st
import streamlit.components.v1 as components

# Title and description
st.title('Metalloproteomics of Microbes')
st.markdown("""
Metals are essential micronutrients for all life. Metalloproteomics combines inorganic and organic mass spectrometry methods to localize metals within protein. 
Explore the metalloproteomes of various organisms studied by the Saito Lab.

""")

st.image("pao_zn_example_image.jpg", caption="Metalloproteomics Research")

# Define the options for the dropdown menu
options = {
    "Select an organism: ": "",
    "Pseudomonas aeruginosa": "https://metallomeapp-pao-3d-metals.streamlit.app/",
    "Pelagibacter (SAR11)": "https://the-sar11-metalloproteome.streamlit.app/",
    "Pseudoalteromonas - coming soon": "",
    "E. coli - coming soon": "",
    "Trichodesmium - coming soon": "",
}

# Create the dropdown menu
selection = st.selectbox("Choose an organism's metalloproteome:", list(options.keys()))

# Redirect to the selected webpage
if selection != "":
    st.write(f"Link for {options[selection]}")
    
st.markdown("""
---
Visit the [Saito Lab](https://www.whoi.edu/saitolab) for more information on our research and datasets.
Future site of [Metallome.org](https://www.metallome.org)
""")
