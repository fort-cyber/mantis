import os

DEFAULT_MODEL = "vertex_ai/gemini-3.6-flash"

def get_llm_kwargs(model_id: str = None, default_model: str = DEFAULT_MODEL) -> tuple[str, dict]:
    """Resolves the LLM mapping details cleanly with precedence: node > MODEL_ID env > default."""
    model_id = model_id or os.environ.get("MODEL_ID") or default_model
        
    llm_kwargs = {"model": model_id}
    
    if model_id.startswith("vertex_ai/"):
        project = os.environ.get("VERTEXAI_PROJECT", os.environ.get("GOOGLE_CLOUD_PROJECT"))
        location = os.environ.get("VERTEXAI_LOCATION", os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
        
        if not project:
            try:
                import google.auth
                _, project = google.auth.default()
            except Exception:
                pass
                
        if not project:
            raise ValueError("ERROR: You must set VERTEXAI_PROJECT or GOOGLE_CLOUD_PROJECT env variables.")
        
        llm_kwargs["vertex_project"] = project
        llm_kwargs["vertex_location"] = location
        # Relax safety filters that sometimes trigger erroneously on defensive security
        # analysis and vulnerability remediation workflows.
        llm_kwargs["safety_settings"] = [
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        ]
        
    return model_id, llm_kwargs

