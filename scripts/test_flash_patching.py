
import logging
import sys
import unittest
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.agents.critic import CriticAgent
from src.agents.patcher import PatcherAgent
from src.core.client import ChatGoogleGenerativeAI

# Mock Client
class MockClient:
    def __init__(self):
        pass

class TestFlashPatching(unittest.TestCase):

    def setUp(self):
        # We don't strictly need a real client for these logic tests
        self.mock_client = MockClient()
        self.critic = CriticAgent(client=self.mock_client)
        self.patcher = PatcherAgent(client=self.mock_client)

    def test_critic_json_parsing(self):
        """Test parsing of various JSON formats from Critic response."""
        
        # 1. Pure JSON
        json_resp = '[{"original_span": "old", "replacement": "new", "reason": "fix"}]'
        parsed = self.critic.parse_json_response(json_resp)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["original_span"], "old")
        
        # 2. Markdown JSON
        md_resp = 'Here is the JSON:\n```json\n[{"original_span": "foo", "replacement": "bar"}]\n```'
        parsed = self.critic.parse_json_response(md_resp)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["original_span"], "foo")
        
        # 3. Empty list
        empty_resp = "[]"
        parsed = self.critic.parse_json_response(empty_resp)
        self.assertEqual(len(parsed), 0)

    def test_patcher_application(self):
        """Test deterministic patching logic."""
        
        draft_text = "The quick brown fox jumps over the lazy dog."
        
        # Case 1: Simple replacement
        patches = [
            {"original_span": "brown", "replacement": "red"},
            {"original_span": "lazy", "replacement": "energetic"}
        ]
        result = self.patcher.apply_patches(draft_text, patches)
        expected = "The quick red fox jumps over the energetic dog."
        self.assertEqual(result, expected)
        
        # Case 2: Overlapping/Nested? (Patcher sorts by length, so longest match first)
        draft_text_2 = "Hello World. Hello Universe."
        patches_2 = [
            {"original_span": "Hello", "replacement": "Hi"}, # Found twice, count != 1, should skip (default logic warning)
        ]
        # Patcher logs warning for duplicates and skips if strictly 1 match required? 
        # Let's check logic: if count == 1: apply. else: skip.
        result_2 = self.patcher.apply_patches(draft_text_2, patches_2)
        self.assertEqual(result_2, draft_text_2) # No change
        
        # Case 3: Unique match
        draft_text_3 = "Chapter One: The Beginning."
        patches_3 = [{"original_span": "Chapter One", "replacement": "第一章"}]
        result_3 = self.patcher.apply_patches(draft_text_3, patches_3)
        self.assertEqual(result_3, "第一章: The Beginning.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
