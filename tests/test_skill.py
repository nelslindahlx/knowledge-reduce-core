import os
import json
import yaml
import unittest
import knowledge_graph_pkg

class TestHermesSkillPackaging(unittest.TestCase):

    def setUp(self):
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.skill_path = os.path.join(self.root_dir, ".agents", "skills", "knowledge-reduce-core", "SKILL.md")
        self.registry_path = os.path.join(self.root_dir, ".agents", "skills.json")

    def test_skill_manifest_file_exists_and_has_valid_frontmatter(self):
        self.assertTrue(os.path.isfile(self.skill_path), "SKILL.md file must exist in the canonical path.")
        
        with open(self.skill_path, "r", encoding="utf-8") as fh:
            content = fh.read()
            
        # Parse frontmatter
        self.assertTrue(content.startswith("---"), "SKILL.md must start with --- frontmatter delimiter.")
        parts = content.split("---")
        self.assertTrue(len(parts) >= 3, "SKILL.md must contain a YAML frontmatter block.")
        
        frontmatter_text = parts[1].strip()
        data = yaml.safe_load(frontmatter_text)
        
        # Verify required frontmatter keys
        required_keys = [
            "name", "description", "version", "author", "license", 
            "platforms", "category", "usage_hint", 
            "required_environment_variables", "required_commands", "metadata"
        ]
        for key in required_keys:
            self.assertIn(key, data, f"Required frontmatter key '{key}' is missing in SKILL.md.")

        # Version consistency
        self.assertEqual(data["version"], knowledge_graph_pkg.__version__, 
                         f"SKILL.md version ({data['version']}) must match package version ({knowledge_graph_pkg.__version__}).")

    def test_skills_registry_configuration(self):
        self.assertTrue(os.path.isfile(self.registry_path), "skills.json registry file must exist.")
        
        with open(self.registry_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            
        self.assertIn("skills", data, "skills.json must contain a list under the key 'skills'.")
        
        skills = data["skills"]
        kr_skill = next((s for s in skills if s.get("name") == "knowledge-reduce-core"), None)
        
        self.assertIsNotNone(kr_skill, "skills.json must contain a record for 'knowledge-reduce-core'.")
        self.assertEqual(kr_skill.get("canonical_path"), ".agents/skills/knowledge-reduce-core/SKILL.md")
        self.assertTrue(kr_skill.get("is_active"), "The skill must be marked active.")
