import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import chatbot_pipeline

def test_hypoglycemia_case():
    q = "I'm sweating, shaky, and my glucometer reads 55 mg/dL—what should I do?"
    result = chatbot_pipeline(q)
    assert "hypoglycemia" in result.lower()
    assert "glucose" in result.lower()

def test_severe_hypoglycemia_case():
    q = "My diabetic uncle is unconscious after his sugar crashed."
    result = chatbot_pipeline(q)
    assert "severe hypoglycemia" in result.lower()
    assert "glucagon" in result.lower()

def test_gestational_diabetes_case():
    q = "I have gestational diabetes and my sugar is 130 after eating."
    result = chatbot_pipeline(q)
    assert "gestational diabetes" in result.lower()
    assert "glucose" in result.lower()

def test_myocardial_infarction_case():
    q = "I feel chest pain radiating to my left arm—what do I do?"
    result = chatbot_pipeline(q)
    assert "myocardial infarction" in result.lower()
    assert "aspirin" in result.lower() or "emergency" in result.lower()

def test_angina_case():
    q = "I was diagnosed with angina and have sudden chest tightness."
    result = chatbot_pipeline(q)
    assert "angina" in result.lower()

def test_heart_failure_case():
    q = "I'm short of breath and have a history of heart failure."
    result = chatbot_pipeline(q)
    assert "chronic heart failure" in result.lower()
    assert "sit upright" in result.lower() or "oxygen" in result.lower()

def test_aki_case():
    q = "My creatinine is high, I barely urinated today, and was in the sun."
    result = chatbot_pipeline(q)
    assert "acute kidney injury" in result.lower()

def test_drug_induced_aki_case():
    q = "I took ibuprofen and now my flanks hurt and I feel sick."
    result = chatbot_pipeline(q)
    assert "drug-induced aki" in result.lower()

def test_hyperkalemia_case():
    q = "My potassium level is 6.1—what does that mean?"
    result = chatbot_pipeline(q)
    assert "hyperkalemia" in result.lower()
    assert "heart" in result.lower() or "calcium gluconate" in result.lower()

def test_type2_diabetes_case():
    q = "I’m extremely thirsty and my glucose is very high."
    result = chatbot_pipeline(q)
    assert "type 2 diabetes" in result.lower()
    assert "hyperglycemia" in result.lower()
