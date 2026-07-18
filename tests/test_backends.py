import sys
from unittest.mock import MagicMock, patch

# Setup dummy modules before any package import is evaluated
mock_llama_cpp = MagicMock()
mock_openai = MagicMock()
mock_sentence_transformers = MagicMock()

mock_Llama_class = MagicMock()
mock_llama_cpp.Llama = mock_Llama_class

mock_OpenAI_class = MagicMock()
mock_openai.OpenAI = mock_OpenAI_class

mock_SentenceTransformer_class = MagicMock()
mock_sentence_transformers.SentenceTransformer = mock_SentenceTransformer_class

import unittest
from knowledge_graph_pkg.model_probe import get_backend, LlamaCppBackend, OpenAICompatibleBackend
from knowledge_graph_pkg.embeddings import get_embedder, SentenceTransformersEmbedder

class TestPluggableBackends(unittest.TestCase):

    def setUp(self):
        self.sys_modules_patcher = patch.dict(sys.modules, {
            'llama_cpp': mock_llama_cpp,
            'openai': mock_openai,
            'sentence_transformers': mock_sentence_transformers
        })
        self.sys_modules_patcher.start()
        # Reset mocks before each test
        mock_Llama_class.reset_mock()
        mock_OpenAI_class.reset_mock()
        mock_SentenceTransformer_class.reset_mock()

    def tearDown(self):
        self.sys_modules_patcher.stop()

    def test_llama_cpp_backend(self):
        mock_instance = MagicMock()
        mock_Llama_class.return_value = mock_instance
        
        # Mock chat completion return value
        mock_instance.create_chat_completion.return_value = {
            "choices": [{
                "message": {
                    "content": '{"facts": [{"subject": "A", "predicate": "B", "object": "C"}]}'
                }
            }]
        }
        
        backend = LlamaCppBackend(model_path="/path/to/model.gguf")
        self.assertEqual(backend.model, "model.gguf")
        
        schema = {"type": "object"}
        res = backend.generate_structured("test prompt", schema=schema)
        
        self.assertEqual(len(res["facts"]), 1)
        self.assertEqual(res["facts"][0]["subject"], "A")
        
        # Assert client was called correctly
        mock_Llama_class.assert_called_once()
        mock_instance.create_chat_completion.assert_called_once()
        args, kwargs = mock_instance.create_chat_completion.call_args
        self.assertEqual(kwargs["response_format"]["type"], "json_object")
        self.assertEqual(kwargs["response_format"]["schema"], schema)

    def test_openai_compatible_backend(self):
        mock_instance = MagicMock()
        mock_OpenAI_class.return_value = mock_instance
        
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = '{"facts": [{"subject": "X", "predicate": "Y", "object": "Z"}]}'
        mock_instance.chat.completions.create.return_value = mock_resp
        
        backend = OpenAICompatibleBackend(model="gpt-4", api_key="test-key", base_url="https://api.openai.com/v1")
        self.assertEqual(backend.model, "gpt-4")
        
        schema = {"type": "object"}
        res = backend.generate_structured("test prompt", schema=schema)
        
        self.assertEqual(len(res["facts"]), 1)
        self.assertEqual(res["facts"][0]["subject"], "X")
        
        mock_OpenAI_class.assert_called_once()
        mock_instance.chat.completions.create.assert_called_once()
        args, kwargs = mock_instance.chat.completions.create.call_args
        self.assertEqual(kwargs["model"], "gpt-4")
        self.assertEqual(kwargs["response_format"]["type"], "json_object")

    def test_sentence_transformers_embedder(self):
        mock_instance = MagicMock()
        mock_SentenceTransformer_class.return_value = mock_instance
        
        import numpy as np
        mock_instance.encode.return_value = np.array([0.1, 0.2, 0.3])
        
        embedder = SentenceTransformersEmbedder(model="all-MiniLM-L6-v2")
        self.assertEqual(embedder.model, "all-MiniLM-L6-v2")
        
        vec1 = embedder.embed_one("hello")
        self.assertEqual(vec1, [0.1, 0.2, 0.3])
        
        vec2 = embedder.embed_one("hello")
        self.assertEqual(vec2, [0.1, 0.2, 0.3])
        
        mock_instance.encode.assert_called_once_with("hello", convert_to_numpy=True)
        
        mock_instance.encode.reset_mock()
        mock_instance.encode.return_value = np.array([0.2, 0.4, 0.6])
        
        sim = embedder.similarity("hello", "world")
        self.assertAlmostEqual(sim, 1.0)

    def test_factories(self):
        backend = get_backend("llama-cpp", "/path/to/model.gguf")
        self.assertIsInstance(backend, LlamaCppBackend)

        import os
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            backend_openai = get_backend("openai", "gpt-4")
            self.assertIsInstance(backend_openai, OpenAICompatibleBackend)

        embedder = get_embedder("sentence-transformers", "all-MiniLM-L6-v2")
        self.assertIsInstance(embedder, SentenceTransformersEmbedder)

        with self.assertRaises(ValueError):
            get_backend("invalid-backend", "model-name")
            
        with self.assertRaises(ValueError):
            get_embedder("invalid-embedder")
