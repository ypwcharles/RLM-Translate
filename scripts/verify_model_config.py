
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    pass

from src.graphs.main_graph import create_main_graph

def verify_models():
    print("Verifying model configuration loading...")
    
    # Initialize graph (which initializes client manager)
    graph = create_main_graph()
    manager = graph.client_manager
    
    # Check if overrides are loaded
    print(f"Model Overrides: {manager.model_overrides}")
    
    # Check specific roles
    roles = ["analyzer", "drafter", "critic", "editor"]
    for role in roles:
        # Accessing private method for verification purposes or property if available
        # The manager has properties for each client, but the client itself might not expose the model name directly 
        # depending on implementation (ChatGoogleGenerativeAI).
        # However, we can check the manager's state.
        
        model_name = manager._get_model_name(role)
        print(f"Role '{role}' model: {model_name}")
        
    # Expected values based on user's last edit:
    # analyzer: gemini-3-pro-preview
    # drafter: gemini-3-flash-preview
    # critic: gemini-3-pro-preview
    # editor: gemini-3-flash-preview

if __name__ == "__main__":
    verify_models()
