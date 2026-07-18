import sys
from unittest.mock import MagicMock

# Setup dummy mlx modules before imports occur
mock_mlx_lm = MagicMock()
mock_lora = MagicMock()
mock_mlx_lm.lora = mock_lora
sys.modules['mlx_lm'] = mock_mlx_lm
sys.modules['mlx_lm.lora'] = mock_lora

import unittest
import os
from unittest.mock import patch
from knowledge_graph_pkg.train import MLXTrainer
from knowledge_graph_pkg.cli import main

class TestMLXTrainer(unittest.TestCase):

    @patch('os.path.isfile')
    def test_train_lora_success(self, mock_isfile):
        mock_isfile.return_value = True
        
        trainer = MLXTrainer(model_path="dummy_model", data_dir="dummy_data", adapter_path="dummy_adapters")
        
        mock_lora.run.reset_mock()
        
        trainer.train_lora(iters=5, batch_size=2)
        
        mock_lora.run.assert_called_once()
        args = mock_lora.run.call_args[0][0]
        self.assertEqual(args.model, "dummy_model")
        self.assertEqual(args.iters, 5)
        self.assertEqual(args.batch_size, 2)
        self.assertTrue(args.train)

    @patch('os.path.isfile')
    def test_train_lora_missing_files(self, mock_isfile):
        mock_isfile.return_value = False
        
        trainer = MLXTrainer(model_path="dummy_model", data_dir="dummy_data")
        
        with self.assertRaises(FileNotFoundError):
            trainer.train_lora()

    def test_cli_train_routing(self):
        with patch("sys.argv", ["knowledgereduce", "train", "--model", "dummy_model", "--data", "dummy_data", "--adapter-path", "dummy_adapters", "--iters", "5"]):
            with patch("knowledge_graph_pkg.train.MLXTrainer.train_lora") as mock_train:
                code = main()
                self.assertEqual(code, 0)
                mock_train.assert_called_once_with(iters=5, batch_size=4, lr=1e-05, num_layers=16)
