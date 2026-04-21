import gradio as gr
import os
import sys
import numpy as np

# Fix Windows console encoding
os.environ["PYTHONUTF8"] = "1"
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def train_model_if_needed():
    """Train the model if artifacts don't exist yet."""
    model_path = os.path.join('artifacts', 'model.pkl')
    preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

    if os.path.exists(model_path) and os.path.exists(preprocessor_path):
        print("[OK] Model artifacts found. Skipping training.")
        return True

    print("[TRAINING] Training model for the first time...")
    try:
        from src.components.data_ingestion import DataIngestion
        from src.components.data_transformation import DataTransformation
        from src.components.model_trainer import ModelTrainer

        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        transformation = DataTransformation()
        train_arr, test_arr, _ = transformation.initiate_data_transformation(train_path, test_path)

        trainer = ModelTrainer()
        r2_score, best_model_name, report = trainer.initiate_model_trainer(train_arr, test_arr)

        print(f"[OK] Training complete! Best model: {best_model_name} (R2 = {r2_score:.4f})")
        print(f"[REPORT] All model scores: {report}")
        return True
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_overall_grade(score):
    """Returns grade letter, color, icon, and description based on overall average."""
    if score >= 90:
        return "A+", "#22C55E", "Excellent", "Outstanding performance across all subjects!"
    elif score >= 80:
        return "A", "#22C55E", "Very Good", "Strong academic performance. Keep it up!"
    elif score >= 70:
        return "B", "#2563EB", "Good", "Solid understanding. Room to excel further."
    elif score >= 60:
        return "C", "#F59E0B", "Average", "Satisfactory performance. Focus on weak areas."
    elif score >= 50:
        return "D", "#F59E0B", "Below Average", "Needs improvement. Consider additional support."
    elif score >= 40:
        return "E", "#EF4444", "Poor", "Significant gaps in understanding. Intervention needed."
    else:
        return "F", "#EF4444", "Fail", "Critical attention required. Immediate support needed."


def get_subject_grade(score):
    """Get letter grade for individual subject."""
    if score >= 90: return "A+"
    elif score >= 80: return "A"
    elif score >= 70: return "B"
    elif score >= 60: return "C"
    elif score >= 50: return "D"
    elif score >= 40: return "E"
    else: return "F"


def get_insights(final_score, math_pre, bio_pre, chem_pre, phys_pre, eng_pre,
                 gender, parent_edu, test_prep, study_time):
    """Generate personalized insights based on overall performance."""
    insights = []

    # Overall assessment
    if final_score >= 80:
        insights.append("**Strong Academic Profile** - This student is well prepared. The model predicts a high final grade based on current metrics.")
    elif final_score >= 60:
        insights.append("**Moderate Performance** - The student shows a solid foundation. Targeted practice in weaker pre-exam subjects could significantly boost the final grade.")
    else:
        insights.append("**Early Intervention Recommended** - Current pre-exam scores and study habits indicate a high risk of failing. Consider additional tutoring.")

    # Subject-specific insights
    scores = {"Mathematics": math_pre, "Biology": bio_pre, "Chemistry": chem_pre, "Physics": phys_pre, "English": eng_pre}
    strongest = max(scores, key=scores.get)
    weakest = min(scores, key=scores.get)
    if scores[strongest] - scores[weakest] > 15:
        insights.append(f"**Subject Gap Detected** - Strongest in {strongest} ({scores[strongest]:.0f}) vs weakest in {weakest} ({scores[weakest]:.0f}). Focus on bridging this gap to improve the overall average.")

    # Study time insight
    if study_time in ["0-3 hrs", "3-5 hrs"]:
        insights.append("**Study Habits** - Bumping study time to 5-10 hours per week is statistically correlated with a 5-10 point increase in the final score.")

    # Test prep insight
    if test_prep == "none":
        insights.append("**Test Preparation** - Completing a standardized test preparation course provides a solid boost across all disciplines.")

    # Parent education insight
    higher_edu = ["bachelor's degree", "master's degree"]
    if parent_edu in higher_edu:
        insights.append("**Home Environment** - Higher parental education often provides strong academic support. Leverage this advantage with challenging materials.")

    return "\n\n".join(insights)


def predict_grade(gender, parental_level_of_education, test_preparation_course, study_time,
                  math_pre_score, biology_pre_score, chemistry_pre_score, physics_pre_score, english_pre_score):
    """Main prediction function — predicts final score using ML pipeline."""
    try:
        from src.pipeline.predict_pipeline import CustomData, PredictPipeline

        data = CustomData(
            gender=gender,
            parental_level_of_education=parental_level_of_education,
            test_preparation_course=test_preparation_course,
            study_time=study_time,
            math_pre_score=float(math_pre_score),
            biology_pre_score=float(biology_pre_score),
            chemistry_pre_score=float(chemistry_pre_score),
            physics_pre_score=float(physics_pre_score),
            english_pre_score=float(english_pre_score)
        )

        pred_df = data.get_data_as_data_frame()
        pipeline = PredictPipeline()
        result = pipeline.predict(pred_df)

        final_score = round(float(result[0]), 1)
        final_score = max(0, min(100, final_score))

        grade, grade_color, grade_label, grade_desc = get_overall_grade(final_score)

        insights = get_insights(final_score, math_pre_score, biology_pre_score, chemistry_pre_score, physics_pre_score, english_pre_score,
                                gender, parental_level_of_education, test_preparation_course, study_time)

        # ── Grade Card HTML ──
        grade_card_html = f"""
        <div style="text-align: center; padding: 32px 20px;">
            <div style="
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 120px; height: 120px;
                border-radius: 50%;
                background: linear-gradient(135deg, {grade_color}22, {grade_color}11);
                border: 4px solid {grade_color};
                margin-bottom: 16px;
                box-shadow: 0 8px 24px {grade_color}25;
            ">
                <span style="
                    font-size: 48px;
                    font-weight: 800;
                    color: {grade_color};
                    line-height: 1;
                ">{grade}</span>
            </div>

            <div style="
                font-size: 13px;
                color: #6B7280;
                text-transform: uppercase;
                letter-spacing: 2px;
                margin-bottom: 4px;
                font-weight: 600;
            ">Predicted Final Grade</div>

            <div style="
                display: inline-block;
                background: {grade_color}12;
                color: {grade_color};
                padding: 6px 16px;
                border-radius: 20px;
                font-size: 14px;
                font-weight: 600;
                margin-top: 8px;
                border: 1px solid {grade_color}30;
            ">{grade_label}</div>

            <div style="
                color: #6B7280;
                font-size: 14px;
                margin-top: 12px;
                line-height: 1.5;
            ">{grade_desc}</div>

            <div style="
                margin-top: 20px;
                background: #F3F4F6;
                border-radius: 10px;
                padding: 16px;
                border: 1px solid #E5E7EB;
            ">
                <div style="
                    font-size: 32px;
                    font-weight: 700;
                    color: #111827;
                    line-height: 1;
                ">{final_score}<span style="font-size: 16px; color: #6B7280; font-weight: 500;">/100</span></div>
                <div style="font-size: 12px; color: #6B7280; margin-top: 4px; text-transform: uppercase; letter-spacing: 1px;">Final Score Output</div>
            </div>
        </div>
        """

        # ── Subject Breakdown HTML ──
        def subject_bar(name, score, color):
            sgr = get_subject_grade(score)
            return f"""
            <div style="margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <div>
                        <span style="color: #111827; font-size: 14px; font-weight: 600;">{name}</span>
                        <span style="
                            display: inline-block;
                            background: {color}15;
                            color: {color};
                            padding: 2px 8px;
                            border-radius: 10px;
                            font-size: 11px;
                            font-weight: 600;
                            margin-left: 8px;
                        ">{sgr}</span>
                    </div>
                    <span style="color: #111827; font-weight: 700; font-size: 15px;">{score}</span>
                </div>
                <div style="background: #F3F4F6; border-radius: 8px; padding: 3px; border: 1px solid #E5E7EB;">
                    <div style="
                        height: 10px;
                        border-radius: 6px;
                        background: linear-gradient(90deg, {color}cc, {color});
                        width: {score}%;
                        transition: width 0.8s ease-out;
                    "></div>
                </div>
            </div>
            """

        subjects_html = f"""
        <div style="padding: 20px;">
            <div style="
                font-size: 15px;
                font-weight: 700;
                color: #111827;
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                gap: 8px;
            ">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M18 20V10"/><path d="M12 20V4"/><path d="M6 20v-6"/></svg>
                Pre-Exam Subject Scores
            </div>

            {subject_bar("Mathematics", math_pre_score, "#2563EB")}
            {subject_bar("Biology", biology_pre_score, "#22C55E")}
            {subject_bar("Chemistry", chemistry_pre_score, "#F59E0B")}
            {subject_bar("Physics", physics_pre_score, "#EF4444")}
            {subject_bar("English", english_pre_score, "#4F46E5")}
        </div>
        """

        return grade_card_html, subjects_html, insights

    except Exception as e:
        error_html = f"""
        <div style="text-align: center; padding: 40px; color: #EF4444;">
            <div style="font-size: 36px; margin-bottom: 12px;">&#9888;</div>
            <div style="font-size: 16px; font-weight: 600;">Prediction Error</div>
            <div style="font-size: 13px; color: #6B7280; margin-top: 8px;">{str(e)}</div>
        </div>
        """
        return error_html, "", "An error occurred during prediction. Please check your inputs and try again."


# ─── Custom CSS (Light Theme) ───────────────────────────────────────────────────
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

* {
    font-family: 'Inter', -apple-system, sans-serif !important;
}

.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 48px 20px 32px;
    background: linear-gradient(135deg, #2563EB, #4F46E5);
    border-radius: 20px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}

.app-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 60%);
    animation: shimmer 8s ease-in-out infinite;
}

@keyframes shimmer {
    0%, 100% { transform: translate(0, 0); }
    50% { transform: translate(20px, -20px); }
}

.app-header h1 {
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    color: #FFFFFF !important;
    margin: 0 0 8px !important;
    position: relative;
    line-height: 1.2 !important;
}

.app-header .subtitle {
    color: rgba(255,255,255,0.85);
    font-size: 1rem;
    font-weight: 400;
    max-width: 550px;
    margin: 0 auto 16px;
    line-height: 1.6;
    position: relative;
}

.app-header .badge-row {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    position: relative;
}

.app-header .badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 50px;
    font-size: 12px;
    font-weight: 600;
    background: rgba(255,255,255,0.15);
    color: #FFFFFF;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.2);
}

/* ── Cards ── */
.gr-group {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 16px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.04) !important;
}

/* ── Section Titles ── */
.section-title {
    font-size: 16px !important;
    font-weight: 700 !important;
    color: #111827 !important;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 2px;
}

.section-desc {
    font-size: 13px;
    color: #6B7280;
    margin-bottom: 16px;
}

/* ── Labels ── */
label {
    font-weight: 600 !important;
    font-size: 13px !important;
    color: #111827 !important;
}

/* ── Inputs ── */
input, select, .gr-input, textarea {
    border: 1.5px solid #E5E7EB !important;
    border-radius: 10px !important;
    background: #F9FAFB !important;
    color: #111827 !important;
    font-size: 14px !important;
}

input:focus, select:focus {
    border-color: #4F46E5 !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.12) !important;
}

/* ── Primary Button ── */
#predict-btn {
    background: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    color: white !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.35) !important;
    text-transform: uppercase !important;
}

#predict-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(37,99,235,0.45) !important;
}

#predict-btn:active {
    transform: translateY(0) !important;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 20px;
    color: #6B7280;
    font-size: 13px;
    margin-top: 16px;
}

.footer a {
    color: #2563EB;
    text-decoration: none;
    font-weight: 500;
}

.footer a:hover {
    text-decoration: underline;
}

/* ── About ── */
.about-section {
    padding: 20px !important;
}

.about-section h3 {
    color: #111827 !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    margin-bottom: 6px !important;
}

.about-section p, .about-section li {
    color: #6B7280 !important;
    font-size: 13px !important;
    line-height: 1.7 !important;
}

.about-section strong {
    color: #111827 !important;
}

/* ── Accordion ── */
.gr-accordion {
    border: 1px solid #E5E7EB !important;
    border-radius: 12px !important;
}

/* ── Slider ── */
input[type="range"] {
    accent-color: #2563EB !important;
}

/* ── Responsive ── */
@media (max-width: 768px) {
    .app-header h1 {
        font-size: 1.8rem !important;
    }
}
"""


# ─── Build the Gradio Interface ─────────────────────────────────────────────────

# Train model on startup
train_model_if_needed()

# Theme (Forced Light with Custom Palette)
custom_theme = gr.themes.Soft(
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#F9FAFB",
    body_background_fill_dark="#F9FAFB",
    background_fill_primary="#F9FAFB",
    background_fill_primary_dark="#F9FAFB",
    background_fill_secondary="#FFFFFF",
    background_fill_secondary_dark="#FFFFFF",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#FFFFFF",
    block_border_width="1px",
    block_border_color="#E5E7EB",
    block_border_color_dark="#E5E7EB",
    border_color_primary="#E5E7EB",
    border_color_primary_dark="#E5E7EB",
    block_radius="16px",
    block_shadow="0 1px 3px rgba(0,0,0,0.06)",
    input_background_fill="#F9FAFB",
    input_background_fill_dark="#F9FAFB",
    input_border_color="#E5E7EB",
    input_border_color_dark="#E5E7EB",
    input_border_width="1.5px",
    button_primary_background_fill="linear-gradient(135deg, #2563EB 0%, #4F46E5 100%)",
    button_primary_background_fill_dark="linear-gradient(135deg, #2563EB 0%, #4F46E5 100%)",
    button_primary_text_color="white",
    button_primary_text_color_dark="white",
    body_text_color="#111827",
    body_text_color_dark="#111827",
    body_text_color_subdued="#6B7280",
    body_text_color_subdued_dark="#6B7280",
    color_accent="#2563EB",
    color_accent_soft="#4F46E5",
    color_accent_soft_dark="#4F46E5",
    slider_color="#2563EB",
    slider_color_dark="#2563EB",
)

with gr.Blocks(
    title="Student Performance Prediction | AI-Powered Academic Analysis",
) as demo:

    # ── Header ──
    gr.HTML("""
    <div class="app-header">
        <h1>Student Final Grade Predictor</h1>
        <p class="subtitle">
            Predict the final unified grade using 5 pre-exam subject scores, study duration, and preparation metrics using an optimized ML Pipeline.
        </p>
        <div class="badge-row">
            <span class="badge">&#x1F916; Machine Learning</span>
            <span class="badge">&#x26A1; 6 Models Evaluated</span>
            <span class="badge">&#x1F3AF; R&sup2; Optimized</span>
        </div>
    </div>
    """)

    with gr.Row(equal_height=False):
        # ── Left: Input Form ──
        with gr.Column(scale=5):
            with gr.Group():
                gr.HTML("""
                <div style="padding: 16px 16px 0;">
                    <div class="section-title">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
                        Student Background & Preparation
                    </div>
                    <div class="section-desc">Enter the student's background and study engagement</div>
                </div>
                """)

                with gr.Row():
                    gender = gr.Dropdown(
                        choices=["male", "female"],
                        value="female",
                        label="Gender",
                        elem_id="gender-input",
                    )
                    study_time = gr.Dropdown(
                        choices=['0-3 hrs', '3-5 hrs', '5-10 hrs', '> 10 hrs'],
                        value="3-5 hrs",
                        label="Weekly Study Time",
                        elem_id="study-time-input",
                    )

                with gr.Row():
                    parental_education = gr.Dropdown(
                        choices=[
                            "some high school",
                            "high school",
                            "some college",
                            "associate's degree",
                            "bachelor's degree",
                            "master's degree",
                        ],
                        value="some college",
                        label="Parental Education Level",
                        elem_id="education-input",
                    )
                    test_prep = gr.Dropdown(
                        choices=["none", "completed"],
                        value="none",
                        label="Test Preparation Course",
                        elem_id="test-prep-input",
                    )

            with gr.Group():
                gr.HTML("""
                <div style="padding: 12px 16px 0;">
                    <div class="section-title">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#4F46E5" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>
                        Pre-Exam Subject Scores
                    </div>
                    <div class="section-desc">Set the preliminary scores for each major subject (0-100)</div>
                </div>
                """)

                with gr.Row():
                    math_pre = gr.Slider(minimum=0, maximum=100, value=65, step=1, label="Mathematics")
                    intro_bio = gr.Slider(minimum=0, maximum=100, value=70, step=1, label="Biology")
                with gr.Row():
                    chem_pre = gr.Slider(minimum=0, maximum=100, value=60, step=1, label="Chemistry")
                    phys_pre = gr.Slider(minimum=0, maximum=100, value=58, step=1, label="Physics")
                with gr.Row():
                    eng_pre = gr.Slider(minimum=0, maximum=100, value=75, step=1, label="English")

                predict_btn = gr.Button(
                    "Predict Final Grade",
                    variant="primary",
                    elem_id="predict-btn",
                    size="lg",
                )

        # ── Right: Results ──
        with gr.Column(scale=5):
            with gr.Group():
                grade_output = gr.HTML(
                    value="""
                    <div style="text-align: center; padding: 60px 20px;">
                        <div style="
                            display: inline-flex;
                            align-items: center;
                            justify-content: center;
                            width: 100px; height: 100px;
                            border-radius: 50%;
                            background: #F9FAFB;
                            border: 3px solid #E5E7EB;
                            margin-bottom: 16px;
                        ">
                            <span style="font-size: 36px; color: #6B7280;">?</span>
                        </div>
                        <div style="color: #111827; font-size: 15px; font-weight: 600;">Ready to Predict</div>
                        <div style="color: #6B7280; font-size: 13px; margin-top: 4px;">Fill in the details and click predict</div>
                    </div>
                    """,
                    elem_id="grade-output",
                )

            with gr.Group():
                subjects_output = gr.HTML(
                    value="""
                    <div style="text-align: center; padding: 28px; color: #6B7280;">
                        <div style="font-size: 13px;">Subject breakdown will appear here</div>
                    </div>
                    """,
                    elem_id="subjects-output",
                )

    # ── Insights Section ──
    with gr.Group():
        gr.HTML("""
        <div style="padding: 16px 16px 0;">
            <div class="section-title">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v4"/><path d="m15.5 8.5 2.8-2.8"/><path d="M20 12h4"/><path d="m15.5 15.5 2.8 2.8"/><path d="M12 18v4"/><path d="m4.9 19.1 2.8-2.8"/><path d="M0 12h4"/><path d="m4.9 4.9 2.8 2.8"/></svg>
                AI-Powered Insights
            </div>
            <div class="section-desc">Personalized analysis and academic recommendations</div>
        </div>
        """)
        insights_output = gr.Markdown(
            value="*Insights will appear after prediction...*",
            elem_id="insights-output",
        )

    # ── About Section ──
    with gr.Accordion("About This Model", open=False):
        gr.HTML("""
        <div class="about-section">
            <h3>How It Works</h3>
            <p>This application uses machine learning to predict a student's <strong>final, unified grade</strong> based on their
            background, study habits, and pre-exam scores across 5 core subjects. The system evaluates <strong>6 different algorithms</strong>
            to find the best predictive model:</p>
            <ul>
                <li><strong>Random Forest</strong> &mdash; Ensemble of decision trees</li>
                <li><strong>Gradient Boosting</strong> &mdash; Sequential tree boosting</li>
                <li><strong>AdaBoost</strong> &mdash; Adaptive boosting</li>
                <li><strong>K-Nearest Neighbors</strong> &mdash; Instance-based learning</li>
                <li><strong>Decision Tree</strong> &mdash; Single tree model</li>
                <li><strong>Linear Regression</strong> &mdash; Baseline linear model</li>
            </ul>

            <h3 style="margin-top: 14px;">Grading Scale</h3>
            <p>
                <strong style="color: #22C55E;">A+ (90-100)</strong> &bull;
                <strong style="color: #22C55E;">A (80-89)</strong> &bull;
                <strong style="color: #2563EB;">B (70-79)</strong> &bull;
                <strong style="color: #F59E0B;">C (60-69)</strong> &bull;
                <strong style="color: #F59E0B;">D (50-59)</strong> &bull;
                <strong style="color: #EF4444;">E (40-49)</strong> &bull;
                <strong style="color: #EF4444;">F (&lt;40)</strong>
            </p>

            <h3 style="margin-top: 14px;">Dataset Details</h3>
            <p>Trained on a synthetic dataset of 1,000 student records that captures correlations between preparation hours,
            parental education levels, and 5 independent subject proficiencies.</p>
        </div>
        """)

    # ── Footer ──
    gr.HTML("""
    <div class="footer">
        Built with <a href="https://gradio.app" target="_blank">Gradio</a> &
        <a href="https://scikit-learn.org" target="_blank">scikit-learn</a>
        &nbsp;&middot;&nbsp; Deploy on
        <a href="https://huggingface.co/spaces" target="_blank">Hugging Face Spaces</a>
    </div>
    """)

    # ── Wire up ──
    predict_btn.click(
        fn=predict_grade,
        inputs=[
            gender,
            parental_education,
            test_prep,
            study_time,
            math_pre,
            intro_bio,
            chem_pre,
            phys_pre,
            eng_pre
        ],
        outputs=[grade_output, subjects_output, insights_output],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False,
                css=custom_css, theme=custom_theme)
