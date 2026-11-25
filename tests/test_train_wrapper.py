import sys
import unittest
from unittest.mock import MagicMock, patch

from knowledge_graph_pkg.cli import main
from knowledge_graph_pkg.train import MLXTrainer


class TestMLXTrainer(unittest.TestCase):

    def setUp(self):
        self.mock_lora = MagicMock()
        self.mock_lora.CONFIG_DEFAULTS = {
            "batch_size": 4,
            "num_layers": 16,
            "custom_mlx_flag": True,
        }
        self.mock_mlx_lm = MagicMock()
        self.mock_mlx_lm.lora = self.mock_lora
        self.sys_patcher = patch.dict(
            sys.modules,
            {"mlx_lm": self.mock_mlx_lm, "mlx_lm.lora": self.mock_lora},
        )
        self.sys_patcher.start()

    def tearDown(self):
        self.sys_patcher.stop()

    @patch("os.path.isfile")
    def test_train_lora_success_merges_defaults(self, mock_isfile):
        mock_isfile.return_value = True

        trainer = MLXTrainer(
            model_path="dummy_model", data_dir="dummy_data", adapter_path="dummy_adapters"
        )
        trainer.train_lora(iters=5, batch_size=2)

        self.mock_lora.run.assert_called_once()
        args = self.mock_lora.run.call_args[0][0]
        self.assertEqual(args.model, "dummy_model")
        self.assertEqual(args.iters, 5)
        self.assertEqual(args.batch_size, 2)
        self.assertTrue(args.train)
        self.assertTrue(args.custom_mlx_flag)

    @patch("os.path.isfile")
    def test_train_lora_missing_files(self, mock_isfile):
        mock_isfile.return_value = False

        trainer = MLXTrainer(model_path="dummy_model", data_dir="dummy_data")
        with self.assertRaises(FileNotFoundError):
            trainer.train_lora()

    def test_cli_train_routing(self):
        with patch(
            "sys.argv",
            [
                "knowledgereduce",
                "train",
                "--model",
                "dummy_model",
                "--data",
                "dummy_data",
                "--adapter-path",
                "dummy_adapters",
                "--iters",
                "5",
            ],
        ):
            with patch("knowledge_graph_pkg.train.MLXTrainer.train_lora") as mock_train:
                code = main()
                self.assertEqual(code, 0)
                mock_train.assert_called_once_with(
                    iters=5, batch_size=4, lr=1e-05, num_layers=16
                )

    def test_cli_train_sft_routing(self):
        with patch(
            "sys.argv",
            [
                "knowledgereduce",
                "train-sft",
                "--dataset",
                "dummy_train.jsonl",
                "--dry-run",
            ],
        ):
            with patch("scripts.train_sft.main") as mock_train:
                mock_train.return_value = 0
                code = main()
                self.assertEqual(code, 0)
                mock_train.assert_called_once()
