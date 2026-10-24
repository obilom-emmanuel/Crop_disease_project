import streamlit as st
import os
import base64
from pathlib import Path
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
from PIL import Image
from database import CropDiseaseDB
from model import predict_leaf_disease


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CropCare AI | Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --green: #176B45;
        --dark-green: #104C33;
        --light-green: #EAF5EE;
        --text: #20352A;
        --muted: #53685B;
        --border: #D8E5DB;
    }

    /* Main page */
    .stApp {
        background: #F5F8F5;
        font-family: 'Inter', sans-serif;
        color: var(--text);
        font-size: 18px;
    }

    .block-container {
        max-width: 1600px;
        padding: 2.5rem 3rem 3rem;
    }

    /* Larger, more readable headings */
    h1 {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        line-height: 1.25 !important;
    }

    h2 {
        font-size: 2rem !important;
        font-weight: 750 !important;
    }

    h3 {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }

    p, li, label, .stMarkdown {
        font-size: 1.05rem;
        line-height: 1.7;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #104C33 0%, #082F21 100%);
        min-width: 260px;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        font-size: 1rem;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #20352A !important;
    }

    /* Dashboard metric cards */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid var(--border);
        padding: 24px;
        border-radius: 18px;
        box-shadow: 0 5px 18px rgba(25, 75, 48, 0.07);
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-size: 1rem !important;
        font-weight: 700;
    }

    [data-testid="stMetricValue"] {
        color: var(--green);
        font-size: 2.2rem !important;
        font-weight: 800;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.95rem !important;
    }

    /* Welcome banner */
    /* Welcome banner with maize background */
.hero {
    position: relative;
    min-height: 300px;
    padding: 48px;
    border-radius: 24px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    overflow: hidden;
    margin-bottom: 30px;
    box-shadow: 0 10px 30px rgba(16, 76, 51, 0.18);
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 3.2rem !important;
    font-weight: 800 !important;
    line-height: 1.2 !important;
    margin: 0 0 18px 0;
    max-width: 850px;
}

.hero p {
    color: #F1F8F2 !important;
    font-size: 1.25rem !important;
    line-height: 1.8 !important;
    margin: 0;
    max-width: 750px;
}


    /* Feature cards */
    .section-card {
        background: #FFFFFF;
        padding: 28px;
        border: 1px solid var(--border);
        border-radius: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(25, 75, 48, 0.05);
    }

    /* Dashboard feature cards */
    .section-card {
        background: #FFFFFF;
        padding: 28px;
        border: 1px solid #D8E5DB;
        border-radius: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(25, 75, 48, 0.05);
        min-height: 230px;
    }

    /* Card icon */
    .card-icon {
        font-size: 48px;
        margin-bottom: 12px;
    }

    /* Card heading */
    .card-title {
        font-size: 28px;
        font-weight: 800;
        color: #176B45;
        margin: 0 0 12px 0;
        line-height: 1.3;
    }

    /* Card description */
    .card-description {
        font-size: 19px;
        line-height: 1.7;
        color: #53685B;
        margin: 0;
    }

    /* Buttons */
    .stButton > button {
        background: var(--green);
        color: #FFFFFF;
        border: none;
        border-radius: 12px;
        min-height: 52px;
        padding: 10px 22px;
        font-size: 1.05rem;
        font-weight: 700;
        transition: 0.2s ease;
    }
    .stButton > button:hover {
            background: var(--dark-green);
            color: #FFFFFF;
            border: none;
            transform: translateY(-1px);
        }

        /* Upload panel */
        [data-testid="stFileUploader"] {
            background: #FFFFFF;
            border: 2px dashed #91BDA0;
            border-radius: 16px;
            padding: 20px;
        }

        [data-testid="stFileUploader"] * {
            font-size: 1rem;
        }

        /* Tabs */
        [data-testid="stTabs"] button {
            font-size: 1.05rem;
            font-weight: 700;
            padding: 12px 18px;
        }

        /* Text inputs and select boxes */
        [data-testid="stTextInput"] input,
        [data-testid="stSelectbox"] div,
        [data-testid="stDateInput"] input {
            font-size: 1rem;
        }

        /* Alerts */
        [data-testid="stAlert"] {
            border-radius: 14px;
            font-size: 1rem;
            padding: 18px;
        }

        hr {
            border-color: var(--border);
            margin: 28px 0;
        }


/* Maize disease categories section */
.disease-section {
    margin-top: 40px;
    margin-bottom: 25px;
}

.disease-section-title {
    font-size: 30px;
    font-weight: 800;
    color: #104C33;
    margin-bottom: 8px;
}

.disease-section-subtitle {
    font-size: 17px;
    color: #53685B;
    margin-bottom: 24px;
}

/* Individual disease card */
.disease-card {
    background: #FFFFFF;
    border: 1px solid #DDE9DF;
    border-radius: 18px;
    overflow: hidden;
    height: 100%;
    box-shadow: 0 5px 18px rgba(16, 76, 51, 0.08);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.disease-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(16, 76, 51, 0.15);
}

/* Maize leaf image */
.disease-card-image {
    width: 100%;
    height: 175px;
    object-fit: cover;
    display: block;
}

/* Card content */
.disease-card-content {
    padding: 20px;
}

.disease-card-title {
    font-size: 22px;
    font-weight: 800;
    color: #104C33;
    margin: 0 0 12px 0;
    line-height: 1.4;
}

.disease-card-description {
    font-size: 16px;
    color: #53685B;
    line-height: 1.7;
    margin: 0;
}

.disease-card-label {
    display: inline-block;
    font-size: 13px;
    font-weight: 700;
    padding: 6px 10px;
    border-radius: 20px;
    margin-bottom: 12px;
}

.label-disease {
    background: #FFF0E8;
    color: #C44B1A;
}

.label-healthy {
    background: #E5F5E9;
    color: #176B45;
}


    /* Mobile-friendly sizing */
    @media (max-width: 768px) {
        .block-container {
            padding: 1.2rem 1rem 2rem;
        }

        h1 {
            font-size: 2rem !important;
        }

        h2 {
            font-size: 1.65rem !important;
        }

        h3 {
            font-size: 1.3rem !important;
        }

        .hero {
        min-height: 250px;
        padding: 26px;
        background-position: center;
    }

    .hero h1 {
        font-size: 2.1rem !important;
    }

    .hero p {
        font-size: 1.05rem !important;
    }
}

        [data-testid="stMetric"] {
            padding: 15px;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.7rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DISEASE INFORMATION AND MANAGEMENT ADVICE
# ============================================================

DISEASE_ADVICE = {
    "cercospora leaf spot": {
        "title": "Cercospora Leaf Spot",
        "description": (
            "A fungal disease that causes leaf lesions and can reduce "
            "the healthy leaf area available for photosynthesis."
        ),
        "symptoms": [
            "Rectangular or elongated lesions on leaves.",
            "Lesions may appear between leaf veins.",
            "Affected leaves may lose healthy green tissue."
        ],
        "solutions": [
            "Use maize varieties with resistance or tolerance where available.",
            "Rotate maize with suitable non-host crops where practical.",
            "Manage infected crop residues according to local agricultural guidance.",
            "Monitor the field regularly for new or spreading lesions.",
            "Ask an agricultural extension officer whether a locally registered "
            "fungicide is appropriate for the situation."
        ],
        "advice": (
            "Inspect other maize plants for similar symptoms. Avoid selecting "
            "a chemical treatment based only on an AI prediction."
        )
    },

    "common rust": {
        "title": "Common Rust",
        "description": (
            "A fungal disease of maize that commonly produces rust-coloured "
            "pustules on leaf surfaces."
        ),
        "symptoms": [
            "Small reddish-brown or rust-coloured pustules.",
            "Pustules may become more numerous as infection develops.",
            "Severe infection can reduce healthy leaf area."
        ],
        "solutions": [
            "Choose resistant maize varieties where suitable varieties are available.",
            "Inspect nearby plants for similar symptoms.",
            "Maintain appropriate crop spacing and field management.",
            "Monitor crops when weather conditions favour disease development.",
            "Consult an agricultural extension officer before considering "
            "a registered fungicide."
        ],
        "advice": (
            "Check whether the visible symptoms resemble common rust. If the "
            "symptoms are unclear, obtain expert confirmation."
        )
    },

    "healthy": {
        "title": "Healthy Maize Leaf",
        "description": (
            "The model classified the submitted image as a healthy maize leaf. "
            "This does not guarantee that the entire plant or field is disease-free."
        ),
        "symptoms": [
            "The model did not identify one of its supported disease classes "
            "as the most likely result."
        ],
        "solutions": [
            "Continue inspecting maize plants regularly.",
            "Use suitable maize varieties and quality seed.",
            "Maintain appropriate soil fertility and irrigation practices.",
            "Manage weeds and crop residues appropriately.",
            "Monitor leaves for new spots, pustules, or unusual discoloration."
        ],
        "advice": (
            "Continue routine field monitoring. Seek advice if the plant shows "
            "unusual symptoms despite receiving a healthy prediction."
        )
    },

    "northern leaf blight": {
        "title": "Northern Leaf Blight",
        "description": (
            "A fungal disease of maize that can cause elongated leaf lesions "
            "and damage healthy leaf tissue."
        ),
        "symptoms": [
            "Long, elongated greyish-green or tan lesions.",
            "Lesions may enlarge and merge as the disease progresses.",
            "Severe infection can damage substantial leaf area."
        ],
        "solutions": [
            "Use maize varieties with resistance where available.",
            "Rotate maize with suitable non-host crops where practical.",
            "Manage infected crop residues according to local guidance.",
            "Inspect the field regularly and monitor disease progression.",
            "Ask an agricultural extension officer about appropriate registered "
            "fungicides when local recommendations justify their use."
        ],
        "advice": (
            "Check other plants for similar lesions. Seek professional advice "
            "if symptoms spread rapidly or the diagnosis is uncertain."
        )
    }
}


def get_disease_advice(disease_label):
    """Match the model's predicted label to its advice."""

    label = str(disease_label).strip().lower()
    label = label.replace("_", " ").replace("-", " ")
    label = " ".join(label.split())

    # Handle common differences in model class naming.
    if "cercospora" in label:
        key = "cercospora leaf spot"
    elif "common rust" in label or label.endswith("rust"):
        key = "common rust"
    elif "northern leaf blight" in label:
        key = "northern leaf blight"
    elif "healthy" in label:
        key = "healthy"
    else:
        key = None

    return DISEASE_ADVICE.get(key)


def display_disease_advice(disease_label):
    """Display management recommendations for the predicted class."""

    advice = get_disease_advice(disease_label)

    st.markdown("---")
    st.subheader("🌿 Disease Information & Recommended Solutions")

    if advice is None:
        st.warning(
            f"No specific advice is configured for '{disease_label}'. "
            "Please consult an agricultural extension officer."
        )
        return

    st.markdown(f"### {advice['title']}")
    st.write(advice["description"])

    with st.expander("🔎 Common Symptoms", expanded=True):
        for symptom in advice["symptoms"]:
            st.markdown(f"- {symptom}")

    with st.expander("💡 Recommended Management Measures", expanded=True):
        for solution in advice["solutions"]:
            st.markdown(f"- {solution}")

    st.info(f"🌱 **Advice for the user:** {advice['advice']}")

    st.caption(
        "This guidance is for general information. The model's prediction "
        "is not a confirmed diagnosis. Confirm uncertain cases with an "
        "agricultural extension officer and follow local crop-protection guidance."
    )



# ============================================================
# INITIALIZE DATABASE AND MODEL
# ============================================================

db = CropDiseaseDB()

MODEL_FILE_PATH = "crop_disease_model.h5"


# ============================================================
# SESSION MANAGEMENT
# ============================================================

if "username" not in st.session_state:
    st.session_state["username"] = "Default_Farmer"


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================


st.sidebar.markdown("""
<div style="padding: 10px 0 22px 0;">
    <div style="font-size: 27px; font-weight: 800; color: white;">
        🌿 CropCare AI
    </div>
    <div style="font-size: 15px; color: #D5EBDD;">
        Maize Disease Detector
    </div>
</div>
""", unsafe_allow_html=True)

main_menu = st.sidebar.radio(
    "MAIN MENU",
    [
        "🏠  Home Dashboard",
        "🔬  Disease Diagnosis Panel",
        "📋  Historical Records Log"
    ],
    label_visibility="visible"
)



# ============================================================
# VIEW 1: HOME DASHBOARD
# ============================================================

if main_menu == "🏠  Home Dashboard":

    
    # ==========================================
    # WELCOME BANNER WITH LOCAL MAIZE IMAGE
    # ==========================================

    BASE_DIR = Path(__file__).resolve().parent
    banner_path = BASE_DIR / "images" / "maize_farm1.jpg"

    if banner_path.exists():
        with open(banner_path, "rb") as image_file:
            banner_base64 = base64.b64encode(
                image_file.read()
            ).decode("utf-8")

        banner_html = f"""
        <div class="hero" style="
            background-image:
                linear-gradient(
                    90deg,
                    rgba(7, 45, 29, 0.92),
                    rgba(12, 72, 43, 0.45)
                ),
                url('data:image/jpeg;base64,{banner_base64}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            min-height: 300px;
            padding: 48px;
            border-radius: 24px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-sizing: border-box;
        ">
            <h1 style="
                color: white;
                font-size: 48px;
                font-weight: 800;
                margin-bottom: 16px;
            ">
                🌿 Welcome to CropCare AI
            </h1>

            <p style="
                color: #F1F8F2;
                font-size: 21px;
                line-height: 1.7;
                max-width: 750px;
            ">
                Smarter farming starts with early detection.
                Identify maize leaf diseases and explore
                practical management advice.
            </p>
        </div>
        """

        st.html(banner_html)

    else:
        st.error(
            f"Banner image not found: {banner_path}. "
            "Check the filename and extension in your images folder."
        )



    st.subheader("Your crop health workspace")

    st.write(
        "Upload a clear maize leaf image to receive an AI-generated "
        "classification, a confidence score, and recommended next steps."
    )

    col1, col2, col3 = st.columns(3)

    
    with col1:
        st.markdown("""
        <div class="section-card">
            <div class="card-icon">🔬</div>
            <h3 class="card-title">AI Diagnosis</h3>
            <p class="card-description">
                Analyze maize leaf images using your trained CNN model.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="section-card">
            <div class="card-icon">🌱</div>
            <h3 class="card-title">Treatment Guidance</h3>
            <p class="card-description">
                Explore practical advice for managing identified maize diseases.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="section-card">
            <div class="card-icon">📋</div>
            <h3 class="card-title">Diagnosis History</h3>
            <p class="card-description">
                Review previous disease predictions saved in your database.
            </p>
        </div>
        """, unsafe_allow_html=True)


    st.info(
        "To begin, select **Disease Diagnosis Panel** from the sidebar."
    )


    # ==========================================
    # MAIZE DISEASE CATEGORIES
    # ==========================================

    st.markdown("""
    <div class="disease-section">
        <div class="disease-section-title">
            🌿 Maize Disease Categories
        </div>
        <div class="disease-section-subtitle">
            Explore the common maize leaf conditions recognized by the system.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Replace these paths with your own maize leaf images.
    disease_cards = [
        {
            "title": "Cercospora Leaf Spot",
            "image": "images/cercospora.jpg",
            "label": "Disease",
            "label_class": "label-disease",
            "description": (
                "Look for small brown or grey leaf spots, "
                "often with yellow areas around the lesions."
            )
        },
        {
            "title": "Common Rust",
            "image": "images/common_rust.jpg",
            "label": "Disease",
            "label_class": "label-disease",
            "description": (
                "Often appears as raised orange-brown or "
                "rust-coloured pustules on the leaf surface."
            )
        },
        {
            "title": "Healthy Maize Leaf",
            "image": "images/healthy.jpg",
            "label": "Healthy",
            "label_class": "label-healthy",
            "description": (
                "A generally green leaf without obvious "
                "disease symptoms. Continue regular monitoring."
            )
        }
    ]

    card_columns = st.columns(3, gap="large")

    for column, disease in zip(card_columns, disease_cards):
        with column:
            import html

            image_path = BASE_DIR / disease["image"]

            if os.path.exists(image_path):
                from PIL import Image as PILImage
                import base64

                with open(image_path, "rb") as image_file:
                    image_data = base64.b64encode(
                        image_file.read()
                    ).decode("utf-8")

                extension = os.path.splitext(image_path)[1].lower()
                mime_type = (
                    "image/png" if extension == ".png"
                    else "image/jpeg"
                )

                image_source = (
                    f"data:{mime_type};base64,{image_data}"
                )
            else:
                image_source = ""

            safe_title = html.escape(disease["title"])
            safe_description = html.escape(disease["description"])
            safe_label = html.escape(disease["label"])
            safe_label_class = html.escape(disease["label_class"])

            if image_source:
                image_html = f"""
                    <img class="disease-card-image"
                         src="{image_source}"
                         alt="{safe_title}">
                """
            else:
                image_html = """
                    <div style="
                        height:175px;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        background:#EAF5EE;
                        color:#176B45;
                        font-size:18px;
                        font-weight:700;">
                        🌿 Add maize leaf image
                    </div>
                """

            
            card_html = f"""
            <div class="disease-card">
                {image_html}
                <div class="disease-card-content">
                    <span class="disease-card-label {safe_label_class}">
                        {safe_label}
                    </span>
                    <h3 class="disease-card-title">
                        {safe_title}
                    </h3>
                    <p class="disease-card-description">
                        {safe_description}
                    </p>
                </div>
            </div>
            """

            st.html(card_html)



# ============================================================
# VIEW 2: DISEASE DIAGNOSIS PANEL
# ============================================================

elif main_menu == "🔬  Disease Diagnosis Panel":

    st.subheader("🔬 Automated Diagnostics Interface")

    st.write(
        "Upload a leaf image or capture one using your camera "
        "to begin the disease diagnosis."
    )

    # --------------------------------------------------------
    # TWO INPUT TABS
    # --------------------------------------------------------

    submenu_tab1, submenu_tab2 = st.tabs(
        [
            "📁 Upload Sample Leaf",
            "📷 Live Camera Capture"
        ]
    )

    # Keep the two sources separate
    uploaded_file = None
    camera_file = None

    # --------------------------------------------------------
    # TAB 1: FILE UPLOAD
    # --------------------------------------------------------

    with submenu_tab1:

        uploaded_file = st.file_uploader(
            "Select a JPEG or PNG image of a plant leaf",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image showing the affected plant leaf."
        )

    # --------------------------------------------------------
    # TAB 2: CAMERA
    # --------------------------------------------------------

    with submenu_tab2:

        camera_file = st.camera_input(
            "Position the affected leaf clearly within the camera view"
        )

    # --------------------------------------------------------
    # SELECT IMAGE SOURCE
    # --------------------------------------------------------

    if uploaded_file is not None:

        selected_file = uploaded_file

    elif camera_file is not None:

        selected_file = camera_file

    else:

        selected_file = None


    # ========================================================
    # PROCESS SELECTED IMAGE
    # ========================================================

    if selected_file is not None:

        st.write("---")

        st.subheader("Selected Leaf Image")

        try:

            # Open image
            image = Image.open(selected_file)

            # Display image
            st.image(
                image,
                caption="Sample Leaf",
                use_container_width=True
            )

        except Exception as error:

            st.error("Unable to read the selected image.")
            st.exception(error)

            st.stop()


        # ----------------------------------------------------
        # PREDICTION BUTTON
        # ----------------------------------------------------

        predict_button = st.button(
            "🔍 Predict Disease",
            type="primary",
            use_container_width=True
        )


        # ====================================================
        # RUN PREDICTION
        # ====================================================

        if predict_button:

            st.subheader("📊 Analysis Results & Metrics")

            # ------------------------------------------------
            # CHECK MODEL FILE
            # ------------------------------------------------

            if not os.path.exists(MODEL_FILE_PATH):

                st.error(
                    f"Model file '{MODEL_FILE_PATH}' was not found."
                )

                st.info(
                    "Make sure crop_disease_model.h5 is located "
                    "inside your crop_disease_project folder."
                )

                st.stop()


            # ------------------------------------------------
            # CREATE TEMPORARY UPLOAD DIRECTORY
            # ------------------------------------------------

            temp_dir = "temp_uploads"

            os.makedirs(
                temp_dir,
                exist_ok=True
            )


            # ------------------------------------------------
            # CREATE SAFE FILE NAME
            # ------------------------------------------------

            if uploaded_file is not None:

                file_name = uploaded_file.name

            else:

                file_name = "camera_leaf.jpg"


            saved_img_path = os.path.join(
                temp_dir,
                file_name
            )


            # ------------------------------------------------
            # SAVE IMAGE
            # ------------------------------------------------

            try:

                image.save(
                    saved_img_path
                )

            except Exception as error:

                st.error(
                    "The selected image could not be saved."
                )

                st.exception(error)

                st.stop()


            # ------------------------------------------------
            # RUN AI MODEL
            # ------------------------------------------------

            with st.spinner(
                "🤖 Analyzing leaf image with the AI model..."
            ):

                try:

                    disease_label, confidence_metric = (
                        predict_leaf_disease(
                            MODEL_FILE_PATH,
                            saved_img_path
                        )
                    )


                    # ========================================
                    # DISPLAY PREDICTION
                    # ========================================

                    st.success(
                        "Disease analysis completed successfully!"
                    )


                    col1, col2 = st.columns(2)


                    with col1:

                        st.metric(
                            label="Predicted Disease",
                            value=disease_label
                        )


                    with col2:

                        st.metric(
                            label="Confidence",
                            value=f"{confidence_metric:.2f}%"
                        )


                    # ========================================
                    # DISPLAY DISEASE-SPECIFIC ADVICE
                    # ========================================

                    display_disease_advice(disease_label)



                    # ========================================
                    # SAVE RESULT TO DATABASE
                    # ========================================

                    try:

                        db.log_diagnosis(
                            username=st.session_state["username"],
                            image_path=saved_img_path,
                            disease=disease_label,
                            confidence=float(confidence_metric)
                        )

                        st.success(
                            "Diagnosis result has been saved "
                            "to the historical records."
                        )

                    except Exception as database_error:

                        st.warning(
                            "The prediction was successful, "
                            "but the result could not be saved "
                            "to the database."
                        )

                        st.exception(database_error)


                except Exception as error:

                    st.error(
                        "❌ Classification failed."
                    )

                    st.write(
                        "Please make sure that:"
                    )

                    st.write(
                        "• The AI model file is available."
                    )

                    st.write(
                        "• The image is a valid plant leaf image."
                    )

                    st.write(
                        "• All required TensorFlow dependencies "
                        "are installed."
                    )

                    st.exception(error)


# ============================================================
# VIEW 3: HISTORICAL RECORDS
# ============================================================

elif main_menu == "📋  Historical Records Log":

    st.subheader(
        "📜 Database Diagnosis Audit Trail"
    )

    st.write(
        f"Displaying diagnosis records for operator ID: "
        f"**{st.session_state['username']}**"
    )


    # --------------------------------------------------------
    # GET HISTORY FROM DATABASE
    # --------------------------------------------------------

    try:

        history_records = db.get_history(
            st.session_state["username"]
        )


        # ----------------------------------------------------
        # DISPLAY RECORDS
        # ----------------------------------------------------

        if history_records:

            for record in history_records:

                with st.container():

                    st.write(
                        f"**Timestamp:** "
                        f"{record['diagnosis_time']}"
                    )

                    st.write(
                        f"**Identified Condition:** "
                        f"{record['predicted_disease']}"
                    )

                    st.write(
                        f"**Confidence Level:** "
                        f"{record['confidence_score']:.2f}%"
                    )

                    st.write(
                        f"**Storage Reference Path:** "
                        f"`{record['image_path']}`"
                    )

                    st.write("---")


        else:

            st.warning(
                "No historical diagnosis records were found "
                "for this user."
            )


    except Exception as error:

        st.error(
            "Unable to retrieve historical diagnosis records."
        )

        st.exception(error)

