import unittest
from app import app
from app import preprocess_text

class TestApp(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        self.app = app.test_client()

    def test_lowercase_conversion(self):
        text = "Hello World"
        expected_result = "hello world"
        self.assertEqual(preprocess_text(text), expected_result)

    def test_url_removal(self):
        text = "Check out this website: https://example.com"
        expected_result = "check website"
        self.assertEqual(preprocess_text(text), expected_result)

    
    def test_extra_whitespace_removal(self):
        text = "This    has    extra   whitespace"
        expected_result = "extra whitespace"
        actual_result = preprocess_text(text).strip()
        self.assertEqual(actual_result, expected_result)        
        

    def test_stopword_removal(self):
        text = "This is a test sentence"
        expected_result = "test sentence"
        self.assertEqual(preprocess_text(text), expected_result)

    def test_empty_text(self):
        text = ""
        expected_result = " "
        self.assertEqual(preprocess_text(text), expected_result)

    def test_text_with_stopwords_only(self):
        text = "the and is"
        expected_result = " "
        self.assertEqual(preprocess_text(text), expected_result)


    def test_get_news(self):
        response = self.app.get("/news?country=us")
        self.assertEqual(response.status_code, 200)
        # You can add more assertions to check the content of the response.

    def test_news_emotions_single_graph(self):
        response = self.app.get("/newsEmotionsSingleGraph?country=us")
        self.assertEqual(response.status_code, 200)
        # You can add more assertions to check the content of the response.

    def test_news_emotions_single_graph_db_save_connectivity(self):
        response = self.app.get("/newsEmotionsSingleGraphDBSave?country=us")
        self.assertEqual(response.status_code, 200)
        # You can add more assertions to check the content of the response.

    def test_get_news_emotions_single_graph_db_connectivity(self):
        response = self.app.get("/getNewsEmotionsSingleGraphDB?start_date=2022-01-01&end_date=2022-12-31&news_source=The%20Tribune%20India")
        self.assertEqual(response.status_code, 200)
        # You can add more assertions to check the content of the response.

    def test_news_emotions_single_graph_db_save(self):
        response = self.app.get("/newsEmotionsSingleGraphDBSave?country=us")
        data = response.get_json()

        # Check if the response status code is 200
        self.assertEqual(response.status_code, 200)

        # Check if the response is a dictionary
        self.assertIsInstance(data, dict)

        if data:
            for source, emotions in data.items():
                # Check if each source is a string
                self.assertIsInstance(source, str)

                # Check if emotions is a dictionary
                self.assertIsInstance(emotions, dict)

                # Check if the emotions dictionary contains expected keys
                self.assertIn("compound", emotions)
                self.assertIn("neg", emotions)
                self.assertIn("neu", emotions)
                self.assertIn("pos", emotions)

                # Check if the values corresponding to each key are lists
                self.assertIsInstance(emotions["compound"], list)
                self.assertIsInstance(emotions["neg"], list)
                self.assertIsInstance(emotions["neu"], list)
                self.assertIsInstance(emotions["pos"], list)

                # Check if the lists are not empty
                self.assertGreater(len(emotions["compound"]), 0)
                self.assertGreater(len(emotions["neg"]), 0)
                self.assertGreater(len(emotions["neu"]), 0)
                self.assertGreater(len(emotions["pos"]), 0)

    def test_get_news_emotions(self):
        response = self.app.get("/news")
        data = response.get_json()

        # Check if the response status code is 200
        self.assertEqual(response.status_code, 200)

        # Check if the response contains the expected keys
        self.assertIn("sentiment_scores", data)
        self.assertIn("word_frequencies", data)
        self.assertIn("ngrams", data)
        self.assertIn("tfidf", data)

        # Check if the sentiment_scores key contains a list
        self.assertIsInstance(data["sentiment_scores"], list)

        # Check if the word_frequencies key contains a nested list
        self.assertIsInstance(data["word_frequencies"], list)
        self.assertIsInstance(data["word_frequencies"][0], list)

        # Check if the ngrams key contains a nested list
        self.assertIsInstance(data["ngrams"], list)
        self.assertIsInstance(data["ngrams"][0], list) 

        # Check if the tfidf key contains a nested list
        self.assertIsInstance(data["tfidf"], list)
        self.assertIsInstance(data["tfidf"][0], list)

if __name__ == "__main__":
    unittest.main()
