import streamlit as st
import pandas as pd
import joblib
import json
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Optional imports with error handling
try:
    import requests
except ImportError:
    requests = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import geocoder
except ImportError:
    geocoder = None
    st.warning("Geocoder not available. Location detection will be disabled.")

# Configure page
st.set_page_config(
    page_title="Maternal Health App",
    page_icon="🤱",
    layout="wide"
)

# Error handling for model loading
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('maternal_risk_model.joblib')
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'maternal_risk_model.joblib' and 'scaler.joblib' are in the app directory.")
        return None, None

# Load model and scaler
model, scaler = load_model_and_scaler()

# Load dataset with error handling
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv('maternal_risk_with_disease_diagnosis.csv')
    except FileNotFoundError:
        st.warning("Dataset file not found. Some features may not work properly.")
        # Create a dummy dataset structure
        return pd.DataFrame({
            'Age': [25], 'HeartRate': [70], 'BodyTemp': [98.6], 'BS': [7.0],
            'SystolicBP': [120], 'DiastolicBP': [80], 'StressLevel': [3],
            'SleepQuality': [3], 'Swelling': [0], 'Nausea': [0],
            'AbdominalPain': [0], 'VaginalBleeding': [0],
            'DepressionSymptoms': [0], 'Diabetes': [0],
            'Anemia': [0], 'PriorMiscarriage': [0],
            'PossibleDiseases': ['No specific diagnosis']
        })

df = load_dataset()

# Load book recommendations with error handling
@st.cache_data
def load_books():
    try:
        with open('books.txt', 'r') as f:
            return [line.strip() for line in f]
    except FileNotFoundError:
        return [
            "What to Expect When You're Expecting",
            "The First Forty Days",
            "Ina May's Guide to Childbirth",
            "The Birth Partner",
            "Pregnancy Day by Day"
        ]

books = load_books()

# Sample doctor data
doctors = {
    "Delhi": [{"name": "Dr. Anjali Sharma", "phone": "+91-9876543210"}],
    "Mumbai": [{"name": "Dr. Priya Patel", "phone": "+91-9876543211"}],
    "Pune": [{"name": "Dr. Meera Joshi", "phone": "+91-9876543213"}],
    "Default": [{"name": "Dr. General", "phone": "+91-9876543212"}]
}

# Create doctors.json if it doesn't exist
if not os.path.exists('doctors.json'):
    with open('doctors.json', 'w') as f:
        json.dump(doctors, f)

# Translation function with error handling
def translate_text(text, lang='hi'):
    try:
        from googletrans import Translator
        translator = Translator()
        return translator.translate(text, dest=lang).text
    except Exception as e:
        st.warning(f"Translation service unavailable: {str(e)}")
        return text  # Return original text if translation fails

# Initialize session state
if 'language' not in st.session_state:
    st.session_state.language = 'en'
if 'points' not in st.session_state:
    st.session_state.points = 0
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}

# Language selection
st.sidebar.selectbox(
    "Select Language / भाषा चुनें",
    options=['English', 'Hindi'],
    key='language_select',
    on_change=lambda: st.session_state.update(language='hi' if st.session_state.language_select == 'Hindi' else 'en')
)

lang = st.session_state.language
is_hindi = lang == 'hi'

# Function to get translated text
def t(text):
    return translate_text(text, 'hi') if is_hindi else text

# Streamlit app layout
st.title(t("Maternal Health Prediction App"))

# Load logo with error handling
try:
    if os.path.exists('logo.png'):
        st.image('logo.png', caption=t("App Logo"), width=200)
    else:
        st.info("Logo image not found. Add 'logo.png' to display the app logo.")
except Exception as e:
    st.warning(f"Could not load logo: {str(e)}")

# Navigation menu with error handling
try:
    from streamlit_option_menu import option_menu
    menu = option_menu(
        t("Menu"), 
        [t("Home"), t("Mom's Risk Prediction"), t("Dad's Support"), t("Doctor Assistance"), t("Books"), t("SOS"), t("Predictive Timeline")],
        icons=['house', 'heart', 'person', 'telephone', 'book', 'exclamation-triangle', 'calendar'],
        menu_icon="cast", 
        default_index=0,
        orientation="horizontal"
    )
except ImportError:
    st.warning("streamlit-option-menu not available. Using sidebar navigation instead.")
    menu = st.sidebar.selectbox(
        t("Navigation"),
        [t("Home"), t("Mom's Risk Prediction"), t("Dad's Support"), t("Doctor Assistance"), t("Books"), t("SOS"), t("Predictive Timeline")]
    )

# Home Page
if menu == t("Home"):
    st.header(t("Welcome to Maternal Health App"))
    st.write(t("This app helps expectant mothers and families monitor maternal health risks, access support, and connect with resources."))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(t("Features"))
        st.write(t("🔮 Predict maternal health risks offline"))
        st.write(t("👨‍👩‍👧‍👦 Tips and meal plans for dads"))
        st.write(t("📍 Location-based doctor contacts"))
        st.write(t("📚 Pregnancy book recommendations"))
    
    with col2:
        st.subheader(t("Additional Services"))
        st.write(t("🆘 SOS emergency contacts"))
        st.write(t("📈 Predictive risk timeline"))
        st.write(t("🎮 Gamification and health points"))
        st.write(t("💡 Personalized health tips"))

# Mom's Risk Prediction
elif menu == t("Mom's Risk Prediction"):
    st.header(t("Mom's Health Risk Prediction"))
    
    if model is None or scaler is None:
        st.error("Model not available. Please ensure model files are properly loaded.")
    else:
        # Input form
        st.subheader(t("Enter Maternal Health Details"))
        with st.form("health_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input(t("Age"), min_value=10, max_value=70, value=25)
                heart_rate = st.number_input(t("Heart Rate (bpm)"), min_value=60, max_value=120, value=70)
                body_temp = st.number_input(t("Body Temperature (F)"), min_value=97.0, max_value=103.0, value=98.6)
                bs = st.number_input(t("Blood Sugar (mmol/L)"), min_value=4.0, max_value=20.0, value=7.0)
                systolic_bp = st.number_input(t("Systolic BP (mmHg)"), min_value=70, max_value=200, value=120)
                diastolic_bp = st.number_input(t("Diastolic BP (mmHg)"), min_value=40, max_value=120, value=80)
                stress_level = st.slider(t("Stress Level"), 1, 5, 3)
                sleep_quality = st.slider(t("Sleep Quality"), 1, 5, 3)
            
            with col2:
                swelling = st.selectbox(t("Swelling"), [0, 1], format_func=lambda x: t("Yes") if x == 1 else t("No"))
                nausea = st.selectbox(t("Nausea"), [0, 1], format_func=lambda x: t("Yes") if x == 1 else t("No"))
                abdominal_pain = st.selectbox(t("Abdominal Pain"), [0, 1], format_func=lambda x: t("Yes") if x == 1 else t("No"))
                vaginal_bleeding = st.selectbox(t("Vaginal Bleeding"), [0, 1], format_func=lambda x: t("Yes") if x == 1 else t("No"))
                depression_symptoms = st.selectbox(t("Depression Symptoms"), [0, 1], format_func=lambda x: t("Yes") if x == 1 else t("No"))
                diabetes = st.selectbox(t("Diabetes"), [0, 1], format_func=lambda x: t("Yes") if x == 1 else t("No"))
                anemia = st.selectbox(t("Anemia"), [0, 1], format_func=lambda x: t("Yes") if x == 1 else t("No"))
                prior_miscarriage = st.selectbox(t("Prior Miscarriage"), [0, 1], format_func=lambda x: t("Yes") if x == 1 else t("No"))
            
            submitted = st.form_submit_button(t("Predict Risk"))
            
        if submitted:
            try:
                # Prepare input data
                input_data = pd.DataFrame({
                    'Age': [age], 'HeartRate': [heart_rate], 'BodyTemp': [body_temp], 'BS': [bs],
                    'SystolicBP': [systolic_bp], 'DiastolicBP': [diastolic_bp], 'StressLevel': [stress_level],
                    'SleepQuality': [sleep_quality], 'Swelling': [swelling], 'Nausea': [nausea],
                    'AbdominalPain': [abdominal_pain], 'VaginalBleeding': [vaginal_bleeding],
                    'DepressionSymptoms': [depression_symptoms], 'Diabetes': [diabetes],
                    'Anemia': [anemia], 'PriorMiscarriage': [prior_miscarriage]
                })

                # Scale numerical features
                numerical_cols = ['Age', 'HeartRate', 'BodyTemp', 'BS', 'SystolicBP', 'DiastolicBP', 'StressLevel', 'SleepQuality']
                input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

                # Predict risk
                prediction = model.predict(input_data)[0]
                risk = "High" if prediction == 1 else "Low"
                
                if risk == "High":
                    st.error(t(f"⚠️ Predicted Risk Level: {risk}"))
                    st.write(t("Please consult with a healthcare provider immediately."))
                else:
                    st.success(t(f"✅ Predicted Risk Level: {risk}"))
                    st.write(t("Continue monitoring your health regularly."))

                # Save to session state for offline use
                st.session_state.user_data['mom'] = input_data.to_dict()
                st.session_state.points += 10  # Gamification: Earn points for completing prediction

                # Disease lookup (approximate matching)
                if len(df) > 1:
                    try:
                        similar_row = df.iloc[(df[numerical_cols] - input_data[numerical_cols].values).abs().sum(axis=1).idxmin()]
                        diseases = similar_row['PossibleDiseases']
                        st.info(t(f"Possible conditions to monitor: {diseases}"))
                    except:
                        st.info(t("General pregnancy monitoring recommended."))

                # Online information link
                if st.button(t("Learn More Online")):
                    st.write(t("Helpful resources:"))
                    st.markdown("- [WebMD Pregnancy Health](https://www.webmd.com/baby/default.htm)")
                    st.markdown("- [Mayo Clinic Pregnancy Guide](https://www.mayoclinic.org/healthy-lifestyle/pregnancy-week-by-week)")
                    
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")

# Dad's Support
elif menu == t("Dad's Support"):
    st.header(t("Support for Dads"))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(t("How Dads Can Help"))
        st.write(t("💝 **Emotional Support**: Listen to her concerns and provide reassurance."))
        st.write(t("📊 **Health Monitoring**: Help track symptoms like swelling or nausea."))
        st.write(t("🏥 **Appointments**: Accompany her to doctor visits."))
        st.write(t("🏠 **Healthy Environment**: Ensure a stress-free home."))
        st.write(t("🧘 **Relaxation**: Help with prenatal yoga and meditation."))
    
    with col2:
        st.subheader(t("Daily Support Tips"))
        st.write(t("🌅 **Morning**: Prepare healthy breakfast"))
        st.write(t("🌞 **Afternoon**: Check in on comfort level"))
        st.write(t("🌙 **Evening**: Plan relaxing activities"))
        st.write(t("🛏️ **Night**: Ensure comfortable sleep setup"))
    
    st.session_state.points += 5  # Gamification: Points for reading support tips

    st.subheader(t("Meal Suggestions for a Healthy Baby"))
    
    # Check if user has made a prediction
    user_conditions = []
    if 'mom' in st.session_state.user_data:
        # Simulate condition checking based on user input
        user_data = st.session_state.user_data['mom']
        if user_data.get('Diabetes', [0])[0] == 1:
            user_conditions.append('Gestational Diabetes')
        if user_data.get('Anemia', [0])[0] == 1:
            user_conditions.append('Anemia')
    
    if 'Gestational Diabetes' in user_conditions:
        st.write(t("🍗 **Low-Sugar Meals**: Grilled chicken with quinoa and steamed vegetables."))
        st.write(t("🥜 **High-Fiber Snacks**: Apple slices with almond butter."))
    if 'Anemia' in user_conditions:
        st.write(t("🥬 **Iron-Rich Foods**: Spinach salad with lentils and citrus dressing."))
        st.write(t("🍊 **Vitamin C Boost**: Orange juice to enhance iron absorption."))
    
    # General meal suggestions
    st.write(t("🍝 **Balanced Diet**: Whole grain pasta with lean protein and greens."))
    st.write(t("🥤 **Hydration**: Fresh fruit smoothies with yogurt."))
    st.write(t("🥗 **Snacks**: Greek yogurt with berries and nuts."))
    
    st.session_state.points += 5  # Gamification: Points for meal planning

# Doctor Assistance
elif menu == t("Doctor Assistance"):
    st.header(t("Doctor Assistance"))
    
    # Location detection with error handling
    city = "Default"
    if geocoder:
        try:
            g = geocoder.ip('me')
            city = g.city if g.city else "Default"
            st.write(t(f"📍 Detected Location: {city}"))
        except Exception as e:
            st.write(t("📍 Location: Unable to detect automatically"))
            city = st.selectbox(t("Select your city"), ["Delhi", "Mumbai", "Pune", "Bangalore", "Chennai", "Kolkata", "Other"])
    else:
        st.write(t("📍 Location detection unavailable"))
        city = st.selectbox(t("Select your city"), ["Delhi", "Mumbai", "Pune", "Bangalore", "Chennai", "Kolkata", "Other"])
    
    try:
        with open('doctors.json', 'r') as f:
            doctors = json.load(f)
    except:
        # Fallback to default doctors
        pass
    
    doctor_list = doctors.get(city, doctors["Default"])
    
    st.subheader(t("Available Doctors"))
    for i, doc in enumerate(doctor_list):
        with st.expander(f"Dr. {doc['name']}"):
            st.write(t(f"📞 Phone: {doc['phone']}"))
            st.write(t("📧 Email: Available on request"))
            st.write(t("🏥 Clinic: Contact for address"))
    
    st.subheader(t("Emergency Contacts"))
    st.write(t("🚨 **Ambulance**: 108 (India)"))
    st.write(t("🏥 **Emergency Services**: 102"))
    
    st.session_state.points += 5  # Gamification: Points for accessing doctor info

# Books Section
elif menu == t("Books"):
    st.header(t("📚 Recommended Pregnancy Books"))
    
    for i, book in enumerate(books, 1):
        st.write(f"{i}. {t(book)}")
    
    st.subheader(t("Where to Find These Books"))
    st.write(t("📖 Available as PDFs online"))
    st.write(t("🏪 Local bookstores"))
    st.write(t("📱 E-book platforms"))
    st.write(t("📚 Public libraries"))
    
    st.session_state.points += 5  # Gamification: Points for exploring books

# SOS Feature
elif menu == t("SOS"):
    st.header(t("🆘 Emergency SOS"))
    
    st.error(t("In case of emergency (e.g., labor pain, bleeding), contact:"))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(t("Personal Contacts"))
        husband_phone = st.text_input(t("Husband's Phone Number"), value="+91-1234567890")
        family_phone = st.text_input(t("Family Contact"), value="+91-9876543210")
        
    with col2:
        st.subheader(t("Emergency Services"))
        st.write(t("🚨 **Ambulance**: 108 (India)"))
        st.write(t("🏥 **Medical Emergency**: 102"))
        st.write(t("👮 **Police**: 100"))
    
    # Location-based doctor
    city = "Default"
    if geocoder:
        try:
            detected_city = geocoder.ip('me').city if geocoder.ip('me') else None
            city = detected_city if detected_city else "Default"
        except:
            city = "Default"
    
    if city == "Default":
        city = st.selectbox(t("Select your city for doctor recommendations"), 
                          ["Delhi", "Mumbai", "Pune", "Bangalore", "Chennai", "Kolkata", "Other"], 
                          key="sos_city")
        
    doctor_list = doctors.get(city, doctors["Default"])
    st.subheader(t("Nearby Doctors"))
    for doc in doctor_list:
        st.write(t(f"👨‍⚕️ **{doc['name']}**: {doc['phone']}"))
    
    if st.button(t("🚨 EMERGENCY ALERT"), type="primary"):
        st.balloons()
        st.success(t("Emergency contacts have been notified! (Simulation)"))
    
    st.session_state.points += 10  # Gamification: Points for setting up SOS

# Predictive Timeline
elif menu == t("Predictive Timeline"):
    st.header(t("📈 Predictive Risk Timeline"))
    
    weeks = list(range(1, 41))
    # Simulate risk progression (simplified for demo)
    base_risk = 0.2
    risk_scores = []
    
    for week in weeks:
        if week < 12:  # First trimester
            risk = base_risk + random.uniform(0, 0.1)
        elif week < 28:  # Second trimester  
            risk = base_risk + random.uniform(-0.1, 0.2)
        else:  # Third trimester
            risk = base_risk + random.uniform(0.1, 0.4)
        risk_scores.append(max(0, min(1, risk)))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(weeks, risk_scores, linewidth=2, color='#ff6b6b')
    ax.fill_between(weeks, risk_scores, alpha=0.3, color='#ff6b6b')
    ax.set_xlabel(t("Pregnancy Week"))
    ax.set_ylabel(t("Risk Score"))
    ax.set_title(t("Risk Progression Over Pregnancy"))
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Add trimester markers
    ax.axvline(x=12, color='green', linestyle='--', alpha=0.7, label=t('End of 1st Trimester'))
    ax.axvline(x=28, color='orange', linestyle='--', alpha=0.7, label=t('End of 2nd Trimester'))
    ax.legend()
    
    st.pyplot(fig)
    
    # Risk interpretation
    current_week = st.slider(t("Current Pregnancy Week"), 1, 40, 20)
    current_risk = risk_scores[current_week - 1]
    
    if current_risk < 0.3:
        st.success(t(f"Week {current_week}: Low risk ({current_risk:.2f})"))
    elif current_risk < 0.6:
        st.warning(t(f"Week {current_week}: Moderate risk ({current_risk:.2f})"))
    else:
        st.error(t(f"Week {current_week}: High risk ({current_risk:.2f})"))
    
    st.session_state.points += 5  # Gamification: Points for viewing timeline

# Smart Personalization Sidebar
st.sidebar.header(t("🔧 Smart Personalization"))

region = st.sidebar.selectbox(t("Select Region"), ["North India", "South India", "West India", "East India", "Other"])

if region == "North India":
    st.sidebar.write(t("🥘 **Dietary Recommendation**: Include whole grains like roti and protein-rich dals."))
elif region == "South India":
    st.sidebar.write(t("🍛 **Dietary Recommendation**: Incorporate rice, sambar, and coconut-based dishes."))
elif region == "West India":
    st.sidebar.write(t("🌶️ **Dietary Recommendation**: Balanced spices with dhokla, and fresh vegetables."))
elif region == "East India":
    st.sidebar.write(t("🐟 **Dietary Recommendation**: Fish curry with rice and leafy greens."))
else:
    st.sidebar.write(t("🥗 **Dietary Recommendation**: Balanced diet with local vegetables and lean proteins."))

# Herb safety checker
herb = st.sidebar.text_input(t("Check Herb Safety"), placeholder=t("Enter herb name"))
if herb:
    # Common safe/unsafe herbs during pregnancy
    safe_herbs = ['ginger', 'mint', 'chamomile', 'lemon balm']
    unsafe_herbs = ['sage', 'rosemary', 'thyme', 'oregano']
    
    herb_lower = herb.lower()
    if any(safe in herb_lower for safe in safe_herbs):
        st.sidebar.success(t(f"✅ {herb} is generally considered safe during pregnancy."))
    elif any(unsafe in herb_lower for unsafe in unsafe_herbs):
        st.sidebar.error(t(f"⚠️ {herb} should be avoided during pregnancy."))
    else:
        st.sidebar.warning(t(f"❓ {herb} safety: Please consult a doctor before use."))
    st.session_state.points += 5  # Gamification: Points for herb check

# Weather-based alerts (simulated)
weather_conditions = ["Heat Wave", "Normal", "Rainy", "Cold", "Humid"]
weather = random.choice(weather_conditions)

if weather == "Heat Wave":
    st.sidebar.error(t("🌡️ Heat Wave Alert: Stay hydrated and avoid outdoor activities."))
elif weather == "Rainy":
    st.sidebar.info(t("🌧️ Rainy Weather: Be careful on wet surfaces, stay dry."))
elif weather == "Cold":
    st.sidebar.info(t("❄️ Cold Weather: Keep warm, avoid sudden temperature changes."))
elif weather == "Humid":
    st.sidebar.warning(t("💧 Humid Weather: Stay cool, drink plenty of water."))
else:
    st.sidebar.success(t("☀️ Normal Weather: Perfect for light outdoor activities."))

# Festival/fasting guidance
festival = st.sidebar.checkbox(t("Fasting/Festival Period"))
if festival:
    st.sidebar.write(t("🎉 **Fasting Guidance**: Ensure light, nutritious meals like fruits and milk."))
    st.sidebar.write(t("💡 **Tip**: Break fast gradually with easily digestible foods."))
    st.session_state.points += 5  # Gamification: Points for festival guidance

# SMS Alerts (simulated for demonstration)
if st.sidebar.button(t("📱 Enable SMS Alerts")):
    st.sidebar.success(t("✅ SMS Alerts enabled for basic phone users."))
    st.sidebar.info(t("You will receive daily health tips and reminders."))
    st.session_state.points += 5  # Gamification: Points for enabling SMS

# Display Gamification Points
st.sidebar.header(t("🎮 Your Health Points"))
st.sidebar.write(t(f"🏆 Points Earned: {st.session_state.points}"))

# Achievement system
if st.session_state.points >= 100:
    st.sidebar.success(t("🏅 Master Health Guardian!"))
elif st.session_state.points >= 75:
    st.sidebar.success(t("⭐ Health Champion!"))
elif st.session_state.points >= 50:
    st.sidebar.success(t("🌟 Great job! You've earned a Healthy Mom Badge!"))
elif st.session_state.points >= 25:
    st.sidebar.info(t("📈 Good progress! Keep going!"))
else:
    st.sidebar.write(t("💪 Start your health journey!"))

# Progress bar
progress = min(st.session_state.points / 100, 1.0)
st.sidebar.progress(progress)

# Footer with additional information
st.sidebar.markdown("---")
st.sidebar.subheader(t("📞 Support"))
st.sidebar.write(t("For technical support: help@maternalhealth.com"))
st.sidebar.write(t("Emergency: Always call your local emergency number"))

# Version info
st.sidebar.caption(t("Version 1.0 | Made with ❤️ for maternal health"))