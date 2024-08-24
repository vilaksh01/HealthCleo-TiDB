import streamlit as st
import os
from fit import FoodIntoleranceAnalysisService

# Initialize the service
service = FoodIntoleranceAnalysisService(
    upstage_api_key="up_0Xq9",
    tavily_api_key="tvly-Kjrbd",
    tidb_connection_string="mysql+....p0/test?ssl_ca=/etc/ssl/cert.pem&ssl_verify_cert=true&ssl_verify_identity=true"
)

st.title("Food Intolerance Analysis App")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Upload Report", "Analyze Product", "View Report"])

if page == "Upload Report":
    st.header("Upload Food Intolerance Report")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        with st.spinner("Processing report..."):
            # Save the uploaded file temporarily
            with open("temp_report.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            # Process the PDF
            report = service.process_pdf("temp_report.pdf")
            # Save the report to session state
            st.session_state.report = report
            st.success("Report processed successfully!")
        # Clean up the temporary file
        os.remove("temp_report.pdf")

elif page == "Analyze Product":
    st.header("Analyze Product")
    analysis_type = st.radio("Choose analysis type", ["Text", "Image"])
    
    if analysis_type == "Text":
        product_name = st.text_input("Enter product name")
        if st.button("Analyze"):
            with st.spinner("Analyzing product..."):
                analysis = service.analyze_product_from_text(product_name)
                st.subheader(f"Analysis for {analysis.product_name}")
                st.write(f"Ingredients: {', '.join(analysis.ingredients)}")
                st.write("Suitability:")
                for ingredient, suitability in analysis.suitability.items():
                    st.write(f"  {ingredient}: {suitability}")
                st.write(f"Overall Rating: {analysis.overall_rating}")
                st.write(f"Explanation: {analysis.explanation}")
    
    else:  # Image analysis
        uploaded_image = st.file_uploader("Upload product image", type=["jpg", "png", "jpeg"])
        if uploaded_image is not None:
            if st.button("Analyze"):
                with st.spinner("Analyzing product image..."):
                    # Save the uploaded image temporarily
                    with open("temp_image.jpg", "wb") as f:
                        f.write(uploaded_image.getbuffer())
                    # Analyze the image
                    analysis = service.analyze_product_from_image("temp_image.jpg")
                    st.subheader(f"Analysis for {analysis.product_name}")
                    st.write(f"Ingredients: {', '.join(analysis.ingredients)}")
                    st.write("Suitability:")
                    for ingredient, suitability in analysis.suitability.items():
                        st.write(f"  {ingredient}: {suitability}")
                    st.write(f"Overall Rating: {analysis.overall_rating}")
                    st.write(f"Explanation: {analysis.explanation}")
                # Clean up the temporary file
                os.remove("temp_image.jpg")

elif page == "View Report":
    st.header("Food Intolerance Report")
    if 'report' in st.session_state:
        report = st.session_state.report
        st.subheader("Reference Range")
        st.write(f"Elevated: {report.reference_range.elevated}")
        st.write(f"Borderline: {report.reference_range.borderline}")
        st.write(f"Normal: {report.reference_range.normal}")
        
        st.subheader("Food Items")
        for item in report.food_items:
            st.write(f"{item.name}: {item.value} U/mL - {item.category}")
    else:
        st.write("No report available. Please upload a report first.")

# Add a footer with information about TiDB usage
st.sidebar.markdown("---")
st.sidebar.info("This app uses TiDB Serverless for efficient data storage and Vector Search for advanced analysis capabilities.")