import streamlit as st 
from components.financial_doc_analysis import FinancialDocumentAnalyzer
from components.prediction import StockPredictor
from components.sentiment_analysis import TickerSentimentAnalyzer
st.set_page_config(
    page_title="Main",
    page_icon="💸",
)

st.markdown("<h1 style='color: black; font-style: italic; font-family: Comic Sans MS; font-size:5rem' >AIInvestmentAdvisor 💸</h1> <h3 style='color: black; font-style: italic; font-family: Comic Sans MS; font-size:2rem'>Provides cloud architecture tailored to your needs and visualizes it.</h3>", unsafe_allow_html=True)


st.markdown("<p style='color: #4FC978; font-style: italic; font-family: Comic Sans MS; ' >AICloudArc is an AI powered platform that provides investment advice to the user with features like Share price prediction, sentiment analysis, financial doc analysis and RAG-enabled bot. </p>", unsafe_allow_html=True)

tick = st.input_text("Enter the tick")
if st.button("Predict"):
    predictor = StockPredictor(tick)
    predictions, model, x_train, y_train = predictor.predict()
    prediction_text = ', '.join([str(pred) for pred in predictions])
    st.write(f"Predicted prices for {tick}: {prediction_text}")
    predictor.plot_predictions(model, x_train, y_train)
    url = "https://www.tickertape.in/stocks"
    analyzer = TickerSentimentAnalyzer(url, tick)
    result_df = analyzer.analyze()
    st.write(result_df)


uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    save_path = os.path.join("uploads", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File saved successfully: {save_path}")

analyzer = FinancialDocumentAnalyzer(
        pdf_path="./TSLA-Q3-2023-Update-3.pdf",
        org_id="<YOUR-ACTIVELOOP-ORG-ID>",
        dataset_name="tsla_q3"
    )
analyzer.partition_pdf()
analyzer.create_vector_store_index()
analyzer.convert_pdf_to_images()
prompt = st.input_text("Ask any question related to the doc")
response = analyzer.query(prompt)
st.write(response)
print(response)