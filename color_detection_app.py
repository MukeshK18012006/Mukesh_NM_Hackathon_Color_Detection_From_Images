import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image

@st.cache
def load_colors():
    colors = pd.read_csv("DATA/colors.csv", header=None, names=["color_name", "color_name_alt", "hex", "R", "G", "B"])
    return colors

def closest_color_name(r, g, b, colors):
    distances = np.sqrt((colors["R"] - r)**2 + (colors["G"] - g)**2 + (colors["B"] - b)**2)
    closest_index = distances.idxmin()
    return colors.loc[closest_index, "color_name"], (colors.loc[closest_index, "R"], colors.loc[closest_index, "G"], colors.loc[closest_index, "B"])

st.title("Color Detection from Images")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Click anywhere on the image to detect the color.")

    colors = load_colors()

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            b, g, r = image_np[y, x]
            color_name, (r, g, b) = closest_color_name(r, g, b, colors)
            st.session_state["detected_color"] = {
                "name": color_name,
                "rgb": (r, g, b),
                "hex": "#{:02x}{:02x}{:02x}".format(r, g, b)
            }

    cv2.imshow("Image", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    cv2.setMouseCallback("Image", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if "detected_color" in st.session_state:
        detected_color = st.session_state["detected_color"]
        st.write(f"**Detected Color:** {detected_color['name']}")
        st.write(f"**RGB Values:** {detected_color['rgb']}")
        st.write(f"**Hex Code:** {detected_color['hex']}")
        st.markdown(
            f"<div style='width:100px; height:50px; background-color:{detected_color['hex']};'></div>",
            unsafe_allow_html=True
        )
