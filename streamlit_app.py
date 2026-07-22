"""
streamlit_app.py

This script implements a production-quality Streamlit web application
for the AI-Based Skin Disease Detection System. It allows users to upload
an image, preprocesses it, and uses a trained MobileNetV2 model to predict
the disease category.
"""

import os
import textwrap
# 1. Import required dependencies
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
# Import load_model and preprocess_input precisely as required
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 1. Import class_names from preprocessing.py safely
try:
    from preprocessing import class_names
except ImportError:
    class_names = []

# Configuration Constants
MODEL_PATH = "models/skin_disease_model.keras"
ALT_MODEL_PATH = "models/best_skin_disease_model.keras"

FALLBACK_CLASSES = [
    "Acne and Rosacea Photos",
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "Atopic Dermatitis Photos",
    "Bullous Disease Photos",
    "Cellulitis Impetigo and other Bacterial Infections",
    "Eczema Photos",
    "Exanthems and Drug Eruptions",
    "Hair Loss Photos Alopecia and other Hair Diseases",
    "Herpes HPV and other STDs Photos",
    "Light Diseases and Disorders of Pigmentation",
    "Lupus and other Connective Tissue diseases",
    "Melanoma Skin Cancer Nevi and Moles",
    "Nail Fungus and other Nail Disease",
    "Poison Ivy Photos and other Contact Dermatitis",
    "Psoriasis pictures Lichen Planus and related diseases",
    "Scabies Lyme Disease and other Infestations and Bites",
    "Seborrheic Keratoses and other Benign Tumors",
    "Systemic Disease",
    "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Urticaria Hives",
    "Vascular Tumors",
    "Vasculitis Photos",
    "Warts Molluscum and other Viral Infections"
]

if not class_names:
    class_names = FALLBACK_CLASSES

# Clinical disease database for descriptions, risk profile, and guidelines
DISEASE_INFO = {
    "Acne and Rosacea Photos": {
        "description": "Acne vulgaris is a common inflammatory disease of the pilosebaceous unit, whereas Rosacea is a chronic inflammatory dermatosis primarily affecting the central face.",
        "symptoms": "Inflammatory papules, pustules, nodules, cysts, facial redness (erythema), flushing, and visible telangiectasia (broken blood vessels).",
        "causes": "Excess sebum production, follicular hyperkeratinization, microbial colonization, and vascular hyper-reactivity.",
        "consult": "If symptoms cause emotional distress, fail to respond to over-the-counter treatments, or if severe nodulocystic lesions occur that could lead to permanent scarring.",
        "precautions": "Use non-comedogenic skincare products, wash with a gentle soap-free cleanser twice daily, avoid scrubbing skin, and wear mineral-based sunscreen daily.",
        "risk": "Low"
    },
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": {
        "description": "Actinic Keratosis is a pre-cancerous scaly lesion caused by sun damage. Basal Cell Carcinoma (BCC) and Squamous Cell Carcinoma represent primary skin malignancies.",
        "symptoms": "Rough, scaly, sand-paper-like spots, pearl-like or translucent pink bumps with visible blood vessels, or firm red nodules that ulcerate or bleed.",
        "causes": "Chronic, cumulative ultraviolet (UV) radiation exposure from sunlight or tanning beds, leading to DNA mutations in epidermal keratinocytes.",
        "consult": "Immediately. Any new, changing, bleeding, or non-healing skin lesion must be evaluated by a dermatologist for a biopsy.",
        "precautions": "Strict sun protection is mandatory: wear broad-spectrum SPF 50+ sunscreen daily, wear wide-brimmed hats, and conduct monthly skin self-examinations.",
        "risk": "High"
    },
    "Atopic Dermatitis Photos": {
        "description": "A chronic, pruritic, inflammatory skin condition (eczema) typically beginning in childhood and associated with a personal or family history of allergies or asthma.",
        "symptoms": "Intense itching (pruritus), dry skin, erythematous papules or plaques with scaling, excoriations, and skin thickening (lichenification) in chronic stages.",
        "causes": "A complex interaction between genetic susceptibility (e.g., filaggrin gene mutations), immune system dysregulation, and epidermal barrier dysfunction.",
        "consult": "When itching interferes with sleep or daily activities, if skin appears infected (yellow crusting, warmth, pus), or if standard moisturizers fail to provide relief.",
        "precautions": "Moisturize multiple times daily with thick fragrance-free creams, take short lukewarm showers, use soap-free cleansers, and avoid harsh fibers like wool.",
        "risk": "Medium"
    },
    "Bullous Disease Photos": {
        "description": "A group of rare autoimmune disorders characterized by blisters (vesicles or bullae) forming on the skin and/or mucous membranes.",
        "symptoms": "Tense or flaccid fluid-filled blisters that form on normal-looking skin or red patches, which may easily rupture leaving painful open sores (erosions).",
        "causes": "Autoantibodies targeted against structural proteins (desmogleins or hemidesmosomes) that hold the skin layers together, causing epidermal splitting.",
        "consult": "Urgent. Autoimmune blistering diseases require formal dermatological evaluation, biopsy, and immunofluorescence studies for diagnosis.",
        "precautions": "Avoid popping blisters to minimize infection risk, cover open erosions with sterile non-adherent dressings, and maintain excellent hand hygiene.",
        "risk": "High"
    },
    "Cellulitis Impetigo and other Bacterial Infections": {
        "description": "Bacterial skin infections ranging from highly contagious superficial infections (Impetigo) to deep spreading subcutaneous infections (Cellulitis).",
        "symptoms": "Impetigo: Red sores that quickly rupture and ooze, forming honey-colored crusts. Cellulitis: Spreading area of redness, swelling, warmth, tenderness, and fever.",
        "causes": "Invasion of the skin by bacteria (most commonly Staphylococcus aureus or Streptococcus pyogenes) through micro-tears or cuts.",
        "consult": "Urgent. Spreading redness, fever, chills, or red streaks extending from the lesion warrant immediate medical attention to prevent systemic complications.",
        "precautions": "Wash the affected area gently with mild soap, apply prescription topical or oral antibiotics exactly as directed, and keep cuts covered.",
        "risk": "High"
    },
    "Eczema Photos": {
        "description": "An inflammatory skin reaction pattern presenting with erythema, pruritus, scaling, and occasionally vesicles, encompassing contact, nummular, and dyshidrotic forms.",
        "symptoms": "Itchy dry patches, scaling plaques, small water blisters on hands/feet (dyshidrosis), or localized rashes at sites of chemical contact.",
        "causes": "Epidermal barrier impairment, irritant exposure (soaps, chemicals), contact allergens (nickel, fragrances), stress, and climatic changes.",
        "consult": "If symptoms worsen, cause significant discomfort, or fail to respond to standard over-the-counter hydrocortisone creams.",
        "precautions": "Apply barrier-repair moisturizers, avoid contact with known allergens or irritants, and minimize direct contact with hot water.",
        "risk": "Medium"
    },
    "Exanthems and Drug Eruptions": {
        "description": "Widespread skin eruptions representing a hypersensitivity reaction to a systemic medication or an immunological reaction to a viral infection.",
        "symptoms": "Symmetrical red macules or papules starting on the trunk and spreading to extremities, sometimes accompanied by itching, mild fever, or joint pain.",
        "causes": "Drug-induced delayed T-cell mediated immune reaction or immunological reaction to viral pathogens.",
        "consult": "Immediately if the rash is accompanied by mucosal involvement (mouth/eye sores), skin peeling, blisters, facial swelling, breathing difficulties, or high fever.",
        "precautions": "Discontinue suspected trigger medications only under immediate medical supervision, and document the drug name for future medical records.",
        "risk": "Medium"
    },
    "Hair Loss Photos Alopecia and other Hair Diseases": {
        "description": "Disorders affecting the hair follicle, leading to temporary or permanent loss of hair (alopecia) or structural hair shaft abnormalities.",
        "symptoms": "Patchy bald spots (Alopecia Areata), gradual thinning at the crown or hairline, excessive daily hair shedding, or scarring/inflammation of the scalp.",
        "causes": "Autoimmune hair follicle attack, genetic susceptibility, physiological stressors, hormonal changes, or fungal scalp infections.",
        "consult": "If hair loss is sudden, patchy, accompanied by scalp pain, severe itching, scaling, or scarring, or if hair shedding is extensive.",
        "precautions": "Avoid tight hairstyles, use a gentle sulfate-free shampoo, limit heat styling/chemical treatments, and seek dermatologist evaluation.",
        "risk": "Low"
    },
    "Herpes HPV and other STDs Photos": {
        "description": "Infectious dermatoses caused by viral pathogens (such as Herpes Simplex Virus or Human Papillomavirus) or sexually transmitted infections manifest on the skin/mucosa.",
        "symptoms": "Grouped fluid-filled blisters on a red base, painful ulcers, or rough, raised cauliflower-like lesions (HPV warts) on genital or extragenital regions.",
        "causes": "Transmission of viral or bacterial pathogens through direct contact, micro-abrasions, or sexual contact.",
        "consult": "To confirm diagnosis via PCR or swab tests, obtain appropriate antiviral or antimicrobial therapy, and screen for other co-infections.",
        "precautions": "Avoid touching active lesions to prevent auto-inoculation, practice safe sexual behavior, and wash hands thoroughly after contact.",
        "risk": "Medium"
    },
    "Light Diseases and Disorders of Pigmentation": {
        "description": "Dermatoses triggered or exacerbated by ultraviolet radiation (Light Diseases) or abnormalities in melanin synthesis/distribution leading to discoloration.",
        "symptoms": "Itchy hives or rashes on sun-exposed skin, hyperpigmented facial patches (Melasma), or complete loss of pigment in patches (Vitiligo).",
        "causes": "Immunological reactions to UV light, hormonal influences (pregnancy/oral contraceptives), genetic factors, or autoimmune destruction of melanocytes.",
        "consult": "If changes in skin color occur suddenly, cover large body areas, or if sun exposure triggers hives, blisters, or systemic symptoms.",
        "precautions": "Use high-protection broad-spectrum sunscreen (SPF 50+), wear sun-protective clothing, and avoid tanning beds.",
        "risk": "Low"
    },
    "Lupus and other Connective Tissue diseases": {
        "description": "Autoimmune diseases where the immune system attacks connective tissue proteins, resulting in multi-systemic or localized cutaneous manifestations.",
        "symptoms": "Malar 'butterfly' rash across nose and cheeks, disc-like scaly plaques leaving scars, skin tightening on fingers, or purplish rashes on eyelids.",
        "causes": "Genetic predisposition combined with environmental triggers (particularly UV light, viral infections, and certain medications) leading to systemic autoimmunity.",
        "consult": "Promptly. Cutaneous lupus requires comprehensive medical evaluation, lab workups, and coordination between rheumatologists and dermatologists.",
        "precautions": "Avoid all sun exposure, wear UV-protective clothing, apply sunscreen strictly every 2 hours, and adhere to prescribed therapies.",
        "risk": "High"
    },
    "Melanoma Skin Cancer Nevi and Moles": {
        "description": "Nevi are benign melanocytic proliferation (moles). Melanoma is a highly aggressive malignant tumor of melanocytes, representing the deadliest form of skin cancer.",
        "symptoms": "Moles displaying ABCDE criteria: Asymmetry, Border irregularity, Color variation, Diameter > 6mm, or Evolving size/shape/color; or lesions that itch, bleed, or ooze.",
        "causes": "DNA damage in melanocytes caused by acute, intense UV radiation exposure (such as sunburns) and genetic susceptibility.",
        "consult": "Urgent. Any evolving, bleeding, or highly asymmetric mole requires immediate dermatological evaluation and excision/biopsy.",
        "precautions": "Perform monthly skin checks, receive annual professional full-body skin examinations, protect skin from UV radiation, and avoid artificial tanning.",
        "risk": "High"
    },
    "Nail Fungus and other Nail Disease": {
        "description": "Fungal infection of the nail plate (Onychomycosis) or other non-infectious inflammatory nail disorders (psoriatic nails, nail dystrophy).",
        "symptoms": "Nail thickening, yellow or brown discoloration, brittle or crumbling edges, separation of the nail plate from the bed (onycholysis), or pitting.",
        "causes": "Dermatophyte fungi, yeasts, molds, or underlying systemic skin disorders (Psoriasis, Lichen Planus).",
        "consult": "If nail changes are painful, interfere with walking, show signs of secondary bacterial infection, or if the patient has diabetes.",
        "precautions": "Keep nails clipped short, dry feet thoroughly after bathing, wear breathable footwear, change socks daily, and wear sandals in public locker rooms.",
        "risk": "Low"
    },
    "Poison Ivy Photos and other Contact Dermatitis": {
        "description": "An acute, highly pruritic, inflammatory skin reaction caused by direct exposure of the skin to an allergen (e.g., urushiol in poison ivy) or chemical irritant.",
        "symptoms": "Erythema, severe itching, localized swelling, and linear streaks of vesicles or blisters that ooze clear fluid.",
        "causes": "Type IV cell-mediated hypersensitivity reaction (Allergic Contact Dermatitis) or direct chemical cytotoxicity (Irritant Contact Dermatitis).",
        "consult": "If the rash covers more than 25% of the body, affects the face, eyes, or genitals, or if blisters show signs of secondary infection.",
        "precautions": "Wash skin thoroughly with soap and water immediately after contact with the suspected plant or chemical, wash exposed clothing, and apply calamine lotion.",
        "risk": "Medium"
    },
    "Psoriasis pictures Lichen Planus and related diseases": {
        "description": "Chronic inflammatory dermatoses characterized by epidermal hyperproliferation (Psoriasis) or band-like lymphocytic interface dermatitis (Lichen Planus).",
        "symptoms": "Psoriasis: Well-demarcated pink plaques covered with silvery scales (elbows, knees, scalp). Lichen Planus: Pruritic, purple, polygonal, flat-topped papules (wrists, ankles).",
        "causes": "Autoimmune T-cell mediated response leading to accelerated keratinocyte turnover (Psoriasis) or autoimmune basal keratinocyte destruction (Lichen Planus).",
        "consult": "To establish an accurate diagnosis and treatment plan, which may range from topical corticosteroids to systemic biologics or light therapy.",
        "precautions": "Avoid skin trauma (Koebner phenomenon), moisturize dry skin regularly, avoid hot baths, and manage autoimmune triggers.",
        "risk": "Medium"
    },
    "Scabies Lyme Disease and other Infestations and Bites": {
        "description": "Cutaneous reactions resulting from vector-borne bacterial transmission (Lyme disease from ticks) or direct parasitic skin infestations (Scabies mites).",
        "symptoms": "Scabies: Intense nocturnal itching, erythematous papules, and linear burrows in web spaces. Lyme: Expanding target-like rash (Erythema Migrans) around tick bite.",
        "causes": "Infestation by the Sarcoptes scabiei mite (Scabies) or infection with Borrelia burgdorferi transmitted by Ixodes ticks (Lyme disease).",
        "consult": "Promptly. Both conditions require prescription medical therapy (permethrin for scabies; doxycycline or other antibiotics for Lyme disease).",
        "precautions": "Treat all household members simultaneously (Scabies), wash bedding and clothing in hot water, and wear insect repellent in wooded areas.",
        "risk": "Medium"
    },
    "Seborrheic Keratoses and other Benign Tumors": {
        "description": "Common, non-cancerous epidermal growths that increase in frequency with age, representing benign keratinocyte proliferation.",
        "symptoms": "Waxy, scaly, or 'stuck-on' brown, black, or tan papules or plaques with a verrucous surface, commonly on the trunk, face, or neck.",
        "causes": "Benign clonal proliferation of epidermal keratinocytes; genetic predisposition and age-related changes.",
        "consult": "If the growth changes rapidly, bleeds, becomes highly irritated, or to differentiate it from malignant melanoma.",
        "precautions": "Do not attempt to scratch, cut, or pick off the lesions to avoid infection and scarring. If cosmetically bothersome, they can be removed by a dermatologist.",
        "risk": "Low"
    },
    "Systemic Disease": {
        "description": "Skin manifestations of underlying internal illnesses, such as diabetes, thyroid disorders, or internal inflammatory conditions.",
        "symptoms": "Velvety hyperpigmented patches in skin folds (Acanthosis Nigricans), pretibial scaling, localized deposition, purpura, or yellowish nodules (xanthomas).",
        "causes": "Endocrine disorders (Diabetes, thyroid dysfunction), metabolic disease, internal organ failure, or systemic paraneoplastic syndromes.",
        "consult": "For comprehensive diagnostic evaluation including blood panels, imaging, and specialist consultations to identify the primary internal disease.",
        "precautions": "Follow targeted guidelines based on the underlying systemic disease, control blood glucose, and inspect skin regularly.",
        "risk": "Medium"
    },
    "Tinea Ringworm Candidiasis and other Fungal Infections": {
        "description": "Superficial fungal infections affecting the skin, scalp, or groin, leading to ring-like red rashes or itchy yeast infections.",
        "symptoms": "Annular (ring-like) red scaly patches with active borders and central clearing (Ringworm/Tinea Corporis), severe itching, or red macerated plaques in skin folds.",
        "causes": "Infection by dermatophytes or Candida albicans overgrowth in warm, moist environments.",
        "consult": "If the infection fails to resolve with over-the-counter antifungal creams, spreads extensively, or involves the scalp or face.",
        "precautions": "Keep skin clean and dry, avoid sharing personal clothing or towels, wear sandals in public showers, and complete the full course of antifungal treatment.",
        "risk": "Low"
    },
    "Urticaria Hives": {
        "description": "An acute or chronic vascular reaction characterized by transient wheals (hives) due to mast cell degranulation and histamine release.",
        "symptoms": "Raised, intensely pruritic, erythematous, edematous plaques (wheals) that migrate, fade within 24 hours, and leave no permanent marks.",
        "causes": "Allergic reactions to foods, drugs, or insect stings; physical triggers (pressure, temperature, exercise); infections; or autoimmune mechanisms.",
        "consult": "Immediate emergency care if hives are accompanied by angioedema (swelling of face, lips, tongue), throat tightness, wheezing, or difficulty breathing.",
        "precautions": "Avoid known allergens, take oral non-sedating antihistamines, apply cool compresses to relieve itching, and avoid hot showers.",
        "risk": "Medium"
    },
    "Vascular Tumors": {
        "description": "Benign or malignant growths composed of blood vessels or lymphatic vessels, such as hemangiomas or Kaposi's sarcoma.",
        "symptoms": "Cherry-red papules (Cherry Hemangiomas), expanding purple or blue patches/nodules (Kaposi's Sarcoma), or rubbery bright red lesions.",
        "causes": "Endothelial cell growth factor abnormalities, genetic vascular anomalies, or viral infections (e.g., HHV-8 in Kaposi's Sarcoma).",
        "consult": "To evaluate any new, growing, or ulcerating vascular lesion and rule out malignant vessel tumors.",
        "precautions": "Do not puncture or scratch vascular lesions to prevent significant bleeding, protect skin from irritation, and seek medical assessment.",
        "risk": "Medium"
    },
    "Vasculitis Photos": {
        "description": "Inflammation of blood vessel walls in the skin, which can lead to vessel destruction, ischemia, and necrosis, often reflecting systemic immune complex deposition.",
        "symptoms": "Palpable purpura (non-blanching purple spots), nodules, livedo reticularis (net-like rash), painful skin ulcers, or digital infarcts.",
        "causes": "Type III immune-complex mediated hypersensitivity triggered by drug allergies, infections, connective tissue diseases, or malignancies.",
        "consult": "Promptly. Cutaneous vasculitis can be a sign of internal organ involvement (kidneys, lungs, GI tract) and requires a comprehensive diagnostic workup.",
        "precautions": "Rest, elevate the lower extremities to reduce pressure, avoid cold exposure, and follow prescribed therapies.",
        "risk": "High"
    },
    "Warts Molluscum and other Viral Infections": {
        "description": "Benign epidermal proliferations caused by cutaneous viral pathogens, typically presenting as localized nodules or dome-shaped papules.",
        "symptoms": "Hyperkeratotic, rough papules with tiny black dots (Warts), or small, firm, dome-shaped papules with central umbilication (Molluscum Contagiosum).",
        "causes": "Infection of keratinocytes by Human Papillomavirus (HPV) for warts, or Molluscum Contagiosum Virus (MCV) for molluscum.",
        "consult": "If lesions are painful, spread rapidly, become infected, interfere with function, or fail to respond to standard home treatments.",
        "precautions": "Do not pick or scratch lesions to prevent autoinoculation, wash hands regularly, and wear sandals in public pools/showers.",
        "risk": "Low"
    }
}

# 3. Set the page configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="DermAI Diagnostics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 19. Modular function: load_model()
# 2. Load the trained model only once using @st.cache_resource
@st.cache_resource
def load_model():
    """
    Loads the compiled Keras model from the specified path.
    Caches the model to prevent reloading on every interaction.
    """
    path = MODEL_PATH if os.path.exists(MODEL_PATH) else ALT_MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at '{path}'.")
    return keras_load_model(path)


# 19. Modular function: preprocess_image()
def preprocess_image(uploaded_file):
    """
    Validates, loads, resizes, and preprocesses an uploaded image for MobileNetV2.
    """
    # Load the image using PIL
    img = Image.open(uploaded_file).convert('RGB')
    
    # 9. Resize exactly like predict.py: 224 x 224
    img_resized = img.resize((224, 224))
    
    # 9. Convert to NumPy array
    img_array = np.array(img_resized)
    
    # 9. Apply MobileNetV2 preprocessing (preprocess_input)
    img_array = preprocess_input(img_array)
    
    # 9. Expand dimensions for batch formatting (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    
    return img_batch, img


# Custom prediction pipeline returning top predictions
def predict_top_classes(model, img_batch, classes, top_k=5):
    """
    Predicts the disease using the model and returns the sorted list of top predictions.
    Each prediction is a tuple of (class_name, confidence).
    """
    # Predict using model.predict()
    preds = model.predict(img_batch, verbose=0)[0]
    
    # Sort predictions by confidence in descending order
    top_indices = np.argsort(preds)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        confidence = float(preds[idx])
        if classes and len(classes) > idx:
            disease_name = classes[idx]
        else:
            disease_name = f"Class Index {idx}"
        results.append((disease_name, confidence))
        
    return results


# Custom CSS styling function for shadcn-style premium interface
def apply_custom_styles():
    custom_css = """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    
    <style>
    /* Main app background & font settings */
    .stApp {
        background-color: #09090B !important;
        color: #FAFAFA !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #09090B !important;
        border-right: 1px solid #27272A !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #FAFAFA !important;
    }
    
    /* Hide standard Streamlit header, footer, and menu */
    header {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    #MainMenu {visibility: hidden !important;}
    [data-testid="stHeader"] {background: transparent !important;}
    
    /* Top Navigation Bar */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 0px;
        border-bottom: 1px solid #27272A;
        background-color: #09090B;
        margin-bottom: 36px;
    }
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .nav-logo-icon {
        color: #8B5CF6;
    }
    .nav-title {
        font-size: 18px;
        font-weight: 700;
        color: #FAFAFA;
        letter-spacing: -0.02em;
    }
    .nav-badge {
        background-color: rgba(139, 92, 246, 0.1);
        color: #8B5CF6;
        border: 1px solid rgba(139, 92, 246, 0.2);
        font-size: 10px;
        font-weight: 700;
        padding: 2px 6px;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .nav-actions {
        display: flex;
        align-items: center;
        gap: 18px;
    }
    .nav-link {
        font-size: 14px;
        color: #A1A1AA;
        text-decoration: none;
        transition: color 0.2s;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .nav-link:hover {
        color: #FAFAFA;
    }
    
    /* Hero Section */
    .hero-wrapper {
        margin-bottom: 40px;
        text-align: left;
    }
    .hero-title-gradient {
        font-size: 40px;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin-bottom: 8px;
    }
    .hero-desc {
        font-size: 16px;
        color: #A1A1AA;
        max-width: 600px;
        line-height: 1.5;
        margin: 0;
    }
    
    /* Custom Cards */
    .shadcn-card {
        background-color: #18181B;
        border: 1px solid #27272A;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1), 0 1px 2px rgba(0,0,0,0.06);
        margin-bottom: 24px;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .shadcn-card:hover {
        border-color: rgba(139, 92, 246, 0.3);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .card-header-title {
        font-size: 16px;
        font-weight: 600;
        color: #FAFAFA;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
        letter-spacing: -0.01em;
    }
    
    /* Uploader Dropzone Simulation */
    .uploader-dropzone {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 36px 20px;
        border: 2px dashed #27272A;
        background-color: #18181B;
        border-radius: 12px;
        text-align: center;
        cursor: pointer;
        transition: border-color 0.2s, background-color 0.2s;
        margin-bottom: -15px;
    }
    .uploader-dropzone:hover {
        border-color: #8B5CF6;
        background-color: rgba(139, 92, 246, 0.02);
    }
    .uploader-icon {
        color: #A1A1AA;
        margin-bottom: 12px;
    }
    .uploader-title {
        font-size: 14px;
        font-weight: 500;
        color: #FAFAFA;
        margin-bottom: 4px;
    }
    .uploader-desc {
        font-size: 12px;
        color: #A1A1AA;
    }
    
    /* Image metadata layout */
    .metadata-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-top: 16px;
    }
    .metadata-item {
        background-color: rgba(255, 255, 255, 0.02);
        border: 1px solid #27272A;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
    }
    .metadata-label {
        font-size: 10px;
        color: #A1A1AA;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
        font-weight: 600;
    }
    .metadata-value {
        font-size: 13px;
        color: #FAFAFA;
        font-weight: 700;
    }
    
    /* Badges */
    .badge-pill {
        font-size: 11px;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 9999px;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    .badge-pill-success {
        background-color: rgba(34, 197, 94, 0.1);
        color: #22C55E;
        border: 1px solid rgba(34, 197, 94, 0.2);
    }
    .badge-pill-warning {
        background-color: rgba(245, 158, 11, 0.1);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
    .badge-pill-error {
        background-color: rgba(239, 68, 68, 0.1);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
    
    .risk-badge {
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        display: inline-block;
        letter-spacing: 0.05em;
    }
    .badge-low {
        background-color: rgba(34, 197, 94, 0.1);
        color: #22C55E;
        border: 1px solid rgba(34, 197, 94, 0.2);
    }
    .badge-medium {
        background-color: rgba(245, 158, 11, 0.1);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
    .badge-high {
        background-color: rgba(239, 68, 68, 0.1);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.2);
        animation: pulse-border 2s infinite;
    }
    
    @keyframes pulse-border {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        70% { box-shadow: 0 0 0 6px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    
    .rank-badge {
        display: inline-flex;
        justify-content: center;
        align-items: center;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        font-size: 12px;
        font-weight: 700;
    }
    .rank-1 { background-color: rgba(251, 191, 36, 0.15); color: #FBBF24; border: 1px solid rgba(251, 191, 36, 0.3); }
    .rank-2 { background-color: rgba(156, 163, 175, 0.15); color: #D1D5DB; border: 1px solid rgba(156, 163, 175, 0.3); }
    .rank-3 { background-color: rgba(180, 83, 9, 0.15); color: #D97706; border: 1px solid rgba(180, 83, 9, 0.3); }
    .rank-other { background-color: rgba(255, 255, 255, 0.05); color: #A1A1AA; border: 1px solid rgba(255, 255, 255, 0.1); }
    
    /* Progress bars */
    .progress-container {
        margin-bottom: 16px;
    }
    .progress-info {
        display: flex;
        justify-content: space-between;
        margin-bottom: 6px;
        font-size: 13px;
    }
    .progress-name {
        font-weight: 600;
        color: #FAFAFA;
    }
    .progress-val {
        font-weight: 700;
        color: #8B5CF6;
    }
    .progress-track {
        background-color: #27272A;
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
    }
    .progress-thumb-gradient {
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
        height: 100%;
        border-radius: 4px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Metric Display card */
    .shadcn-metric {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: rgba(255, 255, 255, 0.02);
        border: 1px solid #27272A;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 20px;
    }
    .metric-lhs {
        display: flex;
        flex-direction: column;
    }
    .metric-lbl {
        font-size: 10px;
        color: #A1A1AA;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-v {
        font-size: 28px;
        font-weight: 800;
        color: #FAFAFA;
    }
    
    /* Dynamic details block */
    .condition-details-section {
        border-top: 1px solid #27272A;
        padding-top: 16px;
        margin-top: 16px;
    }
    .detail-block {
        margin-bottom: 16px;
    }
    .detail-title {
        font-size: 11px;
        font-weight: 600;
        color: #A1A1AA;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .detail-content {
        font-size: 14px;
        color: #FAFAFA;
        line-height: 1.5;
    }
    
    /* Warning Card Disclaimer */
    .warning-card {
        background-color: rgba(245, 158, 11, 0.02);
        border: 1px solid rgba(245, 158, 11, 0.15);
        border-left: 4px solid #F59E0B;
        border-radius: 8px;
        padding: 16px;
        margin-top: 24px;
        display: flex;
        gap: 12px;
        align-items: flex-start;
    }
    .warning-icon {
        color: #F59E0B;
        flex-shrink: 0;
        margin-top: 2px;
    }
    .warning-text {
        font-size: 13px;
        color: #A1A1AA;
        line-height: 1.5;
        margin: 0;
    }
    .warning-text strong {
        color: #FAFAFA;
    }
    
    /* Sidebar Custom List Items */
    .sidebar-list {
        display: flex;
        flex-direction: column;
        gap: 12px;
        padding-top: 10px;
    }
    .sidebar-item-card {
        background-color: #18181B;
        border: 1px solid #27272A;
        border-radius: 8px;
        padding: 12px 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .sidebar-lbl {
        font-size: 12px;
        font-weight: 500;
        color: #A1A1AA;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .sidebar-val {
        font-size: 13px;
        font-weight: 600;
        color: #FAFAFA;
    }
    
    /* Hide standard File Uploader elements */
    div[data-testid="stFileUploader"] section {
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    div[data-testid="stFileUploader"] label {
        display: none !important;
    }
    div[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] {
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    div[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] button {
        display: none !important;
    }
    div[data-testid="stFileUploader"] {
        background: #18181B !important;
        border: 2px dashed #27272A !important;
        border-radius: 12px !important;
        padding: 24px !important;
        text-align: center !important;
    }
    
    /* Action Button styling overrides */
    div.stButton > button {
        background: #8B5CF6 !important;
        color: #FAFAFA !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        width: 100% !important;
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.25) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div.stButton > button:hover {
        background: #7C3AED !important;
        box-shadow: 0 6px 16px rgba(139, 92, 246, 0.4) !important;
        transform: translateY(-1px) !important;
    }
    div.stButton > button:active {
        transform: translateY(0px) !important;
    }
    
    /* Scrollbar override */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #09090B;
    }
    ::-webkit-scrollbar-thumb {
        background: #27272A;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #3F3F46;
    }
    
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(34, 197, 94, 0.08);
        color: #22C55E;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid rgba(34, 197, 94, 0.15);
    }
    
    .status-dot {
        width: 6px;
        height: 6px;
        background-color: #22C55E;
        border-radius: 50%;
        display: inline-block;
        animation: status-pulse 1.5s infinite;
    }
    
    @keyframes status-pulse {
        0% { transform: scale(0.9); opacity: 0.6; }
        50% { transform: scale(1.15); opacity: 1; }
        100% { transform: scale(0.9); opacity: 0.6; }
    }
    
    .footer-text {
        text-align: center;
        color: #6b7280;
        font-size: 13px;
        margin-top: 60px;
        line-height: 1.6;
    }
    </style>
    """
    st.markdown(textwrap.dedent(custom_css), unsafe_allow_html=True)


# 19. Modular function: main()
def main():
    """
    Main Streamlit application pipeline. Handles UI structure, user interactions,
    and exception handling.
    """
    # Apply global styles
    apply_custom_styles()
    
    # 5. Sidebar Configuration
    st.sidebar.markdown(
        textwrap.dedent("""
        <div style='text-align: center; margin-bottom: 25px; padding-top: 15px;'>
            <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#8B5CF6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-shield-alert" style="margin-bottom: 12px;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
            <h2 style='margin: 0px 0 5px 0; color: #FAFAFA; font-size: 20px; font-weight: 700; letter-spacing: -0.02em;'>Engine Control</h2>
            <p style='color: #A1A1AA; font-size: 13px; margin: 0;'>Clinical Screening Pipeline</p>
        </div>
        """), 
        unsafe_allow_html=True
    )
    
    # Check model loading status
    model_loaded = False
    try:
        model = load_model()
        model_loaded = True
    except Exception as e:
        status_html = f"<span style='color: #ef4444;'>Error: {e}</span>"
        
    if model_loaded:
        st.sidebar.markdown(
            textwrap.dedent("""
            <div style='margin-bottom: 25px; text-align: center;'>
                <div class="status-indicator">
                    <span class="status-dot"></span> Pipeline Core Calibrated
                </div>
            </div>
            """), 
            unsafe_allow_html=True
        )
    else:
        st.sidebar.error(f"Failed to load model system: {e}")
        st.stop()
        
    st.sidebar.markdown("---")
    
    # Custom Sidebar List with Cards
    st.sidebar.markdown(
        textwrap.dedent(f"""
        <div class="sidebar-list">
            <div class="sidebar-item-card">
                <span class="sidebar-lbl">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>
                    Model
                </span>
                <span class="sidebar-val">MobileNetV2</span>
            </div>
            <div class="sidebar-item-card">
                <span class="sidebar-lbl">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m18 16 4-4-4-4"/><path d="m6 8-4 4 4 4"/><path d="m14.5 4-5 16"/></svg>
                    Framework
                </span>
                <span class="sidebar-val">TensorFlow 2.x</span>
            </div>
            <div class="sidebar-item-card">
                <span class="sidebar-lbl">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 22V4c0-.5.2-1 .6-1.4C5 2.2(5.5 2 6 2h8.5L20 7.5V20c0 .5-.2 1-.6 1.4-.4.4-.9.6-1.4.6H6c-.5 0-1-.2-1.4-.6C4.2 21 4 20.5 4 20v2z"/><path d="M14 2v6h6"/><path d="M16 13H8"/><path d="M16 17H8"/></svg>
                    Dataset
                </span>
                <span class="sidebar-val">DermNet</span>
            </div>
            <div class="sidebar-item-card">
                <span class="sidebar-lbl">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/></svg>
                    Classes
                </span>
                <span class="sidebar-val">{len(class_names)} Categories</span>
            </div>
            <div class="sidebar-item-card">
                <span class="sidebar-lbl">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                    Accuracy
                </span>
                <span class="sidebar-val">84.6%</span>
            </div>
            <div class="sidebar-item-card">
                <span class="sidebar-lbl">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2z"/></svg>
                    Model Size
                </span>
                <span class="sidebar-val">37.1 MB</span>
            </div>
            <div class="sidebar-item-card">
                <span class="sidebar-lbl">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                    Inference Latency
                </span>
                <span class="sidebar-val">~148ms</span>
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )
    
    # 4. Design a professional UI (Top Navigation Bar)
    st.markdown(
        textwrap.dedent("""
        <div class="top-nav">
            <div class="nav-brand">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#8B5CF6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="nav-logo-icon"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
                <span class="nav-title">DermAI Diagnostics</span>
                <span class="nav-badge">Core Engine</span>
            </div>
            <div class="nav-actions">
                <a href="https://github.com" target="_blank" class="nav-link">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"/></svg>
                    GitHub
                </a>
                <span class="nav-link" style="cursor: pointer;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg>
                    Dark Mode
                </span>
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )
    
    # Hero Section
    st.markdown(
        textwrap.dedent("""
        <div class="hero-wrapper">
            <h1 class="hero-title-gradient">AI Skin Disease Detection</h1>
            <p class="hero-desc">Upload a high-resolution dermoscopic image and receive deep learning-powered visual feature extraction and classification analysis within seconds.</p>
        </div>
        """),
        unsafe_allow_html=True
    )
    
    # Define primary layout columns
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    with col1:
        st.markdown('<div class="card-header-title">Input Channel</div>', unsafe_allow_html=True)
        
        # Custom upload dropzone HTML simulation
        st.markdown(
            textwrap.dedent("""
            <div class="uploader-dropzone">
                <div class="uploader-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                </div>
                <div class="uploader-title">Drag & Drop Image Here</div>
                <div class="uploader-desc">Supports JPG, JPEG, or PNG (Max 10MB)</div>
            </div>
            """),
            unsafe_allow_html=True
        )
        
        # 6. Allow image upload using st.file_uploader() (jpg, jpeg, png)
        uploaded_file = st.file_uploader("Select dermoscopic skin image file", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        original_img = None
        img_batch = None
        
        if uploaded_file is not None:
            # Exception handling for loading/processing images
            try:
                # Preprocess the file
                img_batch, original_img = preprocess_image(uploaded_file)
                
                # Render preview
                st.markdown("<div style='margin-top: 24px;'>", unsafe_allow_html=True)
                st.image(original_img, caption="Preprocessed Image (224 x 224)", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Retrieve metadata
                res_str = f"{original_img.width} x {original_img.height}"
                size_bytes = uploaded_file.size
                if size_bytes < 1024 * 1024:
                    size_str = f"{size_bytes / 1024:.1f} KB"
                else:
                    size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
                
                # Retrieve and normalize the file format safely
                fmt_str = "JPEG"
                if uploaded_file and hasattr(uploaded_file, "name") and uploaded_file.name:
                    parts = uploaded_file.name.split(".")
                    if len(parts) > 1:
                        fmt_str = parts[-1].upper()
                
                # Render metadata grid
                st.markdown(
                    textwrap.dedent(f"""
                    <div class="metadata-grid">
                        <div class="metadata-item">
                            <div class="metadata-label">Resolution</div>
                            <div class="metadata-value">{res_str}</div>
                        </div>
                        <div class="metadata-item">
                            <div class="metadata-label">File Size</div>
                            <div class="metadata-value">{size_str}</div>
                        </div>
                        <div class="metadata-item">
                            <div class="metadata-label">Format</div>
                            <div class="metadata-value">{fmt_str}</div>
                        </div>
                    </div>
                    """), 
                    unsafe_allow_html=True
                )
                
            except UnidentifiedImageError:
                st.error("Error: Unsupported image format or corrupted file. Please provide a valid JPG/PNG.")
                uploaded_file = None
            except Exception as e:
                st.error(f"Error processing image: {e}")
                uploaded_file = None
                
        # Action button under preview/metadata
        predict_button = False
        if uploaded_file is not None:
            predict_button = st.button("Run Diagnostic Inference", use_container_width=True)
            
    with col2:
        st.markdown('<div class="card-header-title">Diagnostic Inference</div>', unsafe_allow_html=True)
        
        # Determine current output state in column 2
        if uploaded_file is None:
            # Default state when no image is uploaded
            st.markdown(
                textwrap.dedent("""
                <div class="shadcn-card" style="text-align: center; padding: 50px 24px; border-style: dashed;">
                    <div style="color: #A1A1AA; margin-bottom: 16px; display: flex; justify-content: center;">
                        <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m8 12 4 4 6-6"/></svg>
                    </div>
                    <h3 style="font-size: 16px; font-weight: 600; color: #FAFAFA; margin-bottom: 8px;">Awaiting Visual Input</h3>
                    <p style="font-size: 14px; color: #A1A1AA; max-width: 320px; margin: 0 auto; line-height: 1.5;">
                        Please upload a dermoscopic image on the left panel to execute normalization and classification.
                    </p>
                </div>
                """),
                unsafe_allow_html=True
            )
        elif not predict_button:
            # Image loaded but prediction not triggered yet
            st.markdown(
                textwrap.dedent("""
                <div class="shadcn-card" style="text-align: center; padding: 50px 24px;">
                    <div style="color: #3B82F6; margin-bottom: 16px; display: flex; justify-content: center;">
                        <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z"/><path d="m9 12 2 2 4-4"/></svg>
                    </div>
                    <h3 style="font-size: 16px; font-weight: 600; color: #FAFAFA; margin-bottom: 8px;">Ready for Classification</h3>
                    <p style="font-size: 14px; color: #A1A1AA; max-width: 320px; margin: 0 auto; line-height: 1.5;">
                        Image preprocessing complete. Click the <strong>Run Diagnostic Inference</strong> action button under the image preview to begin neural inference.
                    </p>
                </div>
                """),
                unsafe_allow_html=True
            )
        else:
            # Predict button pressed: perform inference & show results
            with st.spinner("Analyzing visual features..."):
                try:
                    # Execute prediction pipeline to extract top 5 classes
                    predictions = predict_top_classes(model, img_batch, class_names, top_k=5)
                    
                    # Top class info
                    top_class, top_conf = predictions[0]
                    
                    # Fetch details from DISEASE_INFO
                    disease_meta = DISEASE_INFO.get(top_class, {
                        "description": "No localized description is currently configured for this classification category.",
                        "symptoms": "No documented symptom details available.",
                        "causes": "No documented etiology details available.",
                        "consult": "Consult with a certified clinical dermatologist.",
                        "precautions": "Seek professional clinical guidance.",
                        "risk": "Medium"
                    })
                    
                    # Determine confidence badge level
                    if top_conf >= 0.75:
                        conf_badge = '<span class="badge-pill badge-pill-success">High Confidence</span>'
                    elif top_conf >= 0.40:
                        conf_badge = '<span class="badge-pill badge-pill-warning">Medium Confidence</span>'
                    else:
                        conf_badge = '<span class="badge-pill badge-pill-error">Low Confidence</span>'
                        
                    # Determine styling class for risk
                    risk_lower = disease_meta["risk"].lower()
                    
                    # Output primary diagnosis card
                    primary_card_html = f"""
                    <div class="shadcn-card">
                        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 16px;">
                            <span class="risk-badge badge-{risk_lower}">{disease_meta["risk"]} Risk</span>
                            <div style="display: flex; gap: 8px;">
                                {conf_badge}
                            </div>
                        </div>
                        <h2 style="margin: 0 0 16px 0; font-size: 24px; font-weight: 700; color: #FAFAFA; letter-spacing: -0.02em;">
                            {top_class}
                        </h2>
                        
                        <div class="shadcn-metric">
                            <div class="metric-lhs">
                                <span class="metric-lbl">Analysis Match Rate</span>
                                <span class="metric-v">{top_conf * 100:.2f}%</span>
                            </div>
                            <div style="color: #8B5CF6;">
                                <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m4.93 4.93 4.24 4.24"/><path d="m14.83 9.17 4.24-4.24"/><path d="M14.83 14.83 19.07 19.07"/><path d="m9.17 14.83-4.24 4.24"/><path d="m15 12-3-3-3 3 3 3z"/></svg>
                            </div>
                        </div>
                        
                        <div class="condition-details-section">
                            <div class="detail-block">
                                <div class="detail-title">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                                    Condition Description
                                </div>
                                <div class="detail-content">{disease_meta["description"]}</div>
                            </div>
                            
                            <div class="detail-block">
                                <div class="detail-title">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
                                    Common Symptoms
                                </div>
                                <div class="detail-content">{disease_meta["symptoms"]}</div>
                            </div>
                            
                            <div class="detail-block">
                                <div class="detail-title">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                                    Possible Causes
                                </div>
                                <div class="detail-content">{disease_meta["causes"]}</div>
                            </div>
                            
                            <div class="detail-block">
                                <div class="detail-title">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2h-1a.3.3 0 1 0 .2.3"/></svg>
                                    Recommended Precautions
                                </div>
                                <div class="detail-content">{disease_meta["precautions"]}</div>
                            </div>
                            
                            <div class="detail-block" style="margin-bottom: 0;">
                                <div class="detail-title" style="color: #F59E0B;">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m22 8-6 4 6 4V8Z"/><rect width="14" height="12" x="2" y="6" rx="2" ry="2"/></svg>
                                    When to Consult a Doctor
                                </div>
                                <div class="detail-content" style="color: #F59E0B; font-weight: 500;">{disease_meta["consult"]}</div>
                            </div>
                        </div>
                    </div>
                    """
                    st.markdown(textwrap.dedent(primary_card_html), unsafe_allow_html=True)
                    
                    # Construct progress bars HTML for Top 5
                    progress_html = ""
                    ranks = ["🥇", "🥈", "🥉", "4", "5"]
                    rank_classes = ["rank-1", "rank-2", "rank-3", "rank-other", "rank-other"]
                    
                    for i, (class_name, prob) in enumerate(predictions):
                        prob_pct = prob * 100
                        progress_html += f"""
                        <div class="progress-container">
                            <div class="progress-info">
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <span class="rank-badge {rank_classes[i]}">{ranks[i]}</span>
                                    <span class="progress-name">{class_name}</span>
                                </div>
                                <span class="progress-val">{prob_pct:.2f}%</span>
                            </div>
                            <div class="progress-track">
                                <div class="progress-thumb-gradient" style="width: {prob_pct}%;"></div>
                            </div>
                        </div>
                        """
                    
                    # Output secondary predictions card
                    secondary_card_html = f"""
                    <div class="shadcn-card">
                        <h3 class="card-header-title">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>
                            Probability Distribution (Top 5)
                        </h3>
                        <div style="display: flex; flex-direction: column; gap: 4px;">
                            {progress_html}
                        </div>
                    </div>
                    """
                    st.markdown(textwrap.dedent(secondary_card_html), unsafe_allow_html=True)
                    
                    # Warning Disclaimer Card
                    disclaimer_html = """
                    <div class="warning-card">
                        <div class="warning-icon">
                            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                        </div>
                        <p class="warning-text">
                            <strong>Medical Screening Disclaimer:</strong> This AI system is intended for educational and research purposes only and must not replace consultation with a licensed dermatologist.
                        </p>
                    </div>
                    """
                    st.markdown(textwrap.dedent(disclaimer_html), unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error during diagnostic prediction: {e}")
                    
    # EXPANDABLE EXPLANATION SECTION
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("About the AI Diagnostic Backbone & Transfer Learning"):
        st.write("""
        This web application leverages deep learning transfer learning methodologies using a **MobileNetV2** backbone architecture.
        The system is optimized for mobile-friendly, low-latency execution while maintaining high accuracy thresholds.
        
        **Data Infrastructure:**
        The underlying model is trained on the comprehensive **DermNet Dataset**, classification spectrum covers 23 unique skin disease types.
        
        **Screening Disclaimer:**
        This platform represents a screening helper. Artificial intelligence classification is subject to statistical error. It does not replace a clinical examination, professional dermoscopic inspection, or tissue biopsy.
        """)
        
    # FOOTER
    st.markdown(
        textwrap.dedent("""
        <div class="footer-text">
            <hr style="border-top: 1px solid rgba(255,255,255,0.06); margin-bottom: 20px;">
            <p style="font-weight: 600; color: #FAFAFA;">AI Skin Disease Diagnostics Pipeline</p>
            <p style="font-size: 12px; color: #A1A1AA;">Built with TensorFlow, Keras, Pillow and Streamlit. Inspired by shadcn/ui.</p>
        </div>
        """),
        unsafe_allow_html=True
    )


# 21. The file should execute using `streamlit run streamlit_app.py`
if __name__ == "__main__":
    main()
