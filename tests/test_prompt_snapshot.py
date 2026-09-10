import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from symplex.agents import prompts


class PromptSnapshotTests(unittest.TestCase):
    def test_nested_invocations_freeze_instructions_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(prompts, 'ROOT', Path(directory)):
            source = Path(directory) / 'role.md'
            source.write_text('original')
            @prompts.frozen_prompts
            def nested():
                return prompts.prompt('role'), prompts.manifest()
            @prompts.frozen_prompts
            def run():
                before = prompts.manifest()
                source.write_text('revised')
                self.assertEqual(nested(), ('original', before))
                self.assertEqual(prompts.manifest(), before)
            run()
            self.assertEqual(nested()[0], 'revised')

    def test_exception_does_not_leak_prior_snapshot(self):
        @prompts.frozen_prompts
        def fail():
            raise RuntimeError('test')
        with self.assertRaises(RuntimeError):
            fail()
        self.assertIsNone(prompts._ACTIVE.get())
