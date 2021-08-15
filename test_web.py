import unittest
import myWeb
from myWeb import app, paused

class TestWeb(unittest.TestCase):

    def test_home(self):
        tester = app.test_client(self)
        response = tester.get('/', content_type='html/text')
        self.assertEqual(response.status_code, 200)

    def test_generate_page(self):
        tester = app.test_client(self)
        response = tester.get('/generate-page', content_type='html/text')
        self.assertEqual(response.status_code, 200)

    def test_other(self):
        tester = app.test_client(self)
        response = tester.get('a', content_type='html/text')
        self.assertEqual(response.status_code, 404)

    def test_play(self):
        print(paused)
        myWeb.play_song()
        print(paused)


if __name__ == "__main__":
    unittest.main()