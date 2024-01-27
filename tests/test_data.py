import unittest

from helpers.data import hate_speech_score_to_label


class TestData(unittest.TestCase):
    def test_hate_speech_score_to_label(self):
        self.assertEqual(hate_speech_score_to_label(-1.5), "Supportive")
        self.assertEqual(hate_speech_score_to_label(0.7), "Hate")
        self.assertEqual(hate_speech_score_to_label(0.2), "Neutral")
        self.assertEqual(hate_speech_score_to_label(-1), "Neutral")
        self.assertEqual(hate_speech_score_to_label(0.5), "Neutral")
        self.assertEqual(hate_speech_score_to_label(0), "Neutral")


if __name__ == "__main__":
    unittest.main()
