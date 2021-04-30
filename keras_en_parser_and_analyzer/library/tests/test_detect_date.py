from unittest import TestCase
from datetime import date
from keras_en_parser_and_analyzer.library.pipmp_my_cv_classify import detect_date


class DetectDate(TestCase):
    def test_detect_date(self):
        dates_to_test = ['10-1990', '09/12/2020', 'jan 1990', 'feb 2012', '9-12-2020']
        res = detect_date(dates_to_test[0])
        self.assertEqual(10, res.month)
        self.assertEqual(1990, res.year)
        res = detect_date(dates_to_test[1])
        self.assertEqual(9, res.month)
        self.assertEqual(2020, res.year)
        res = detect_date(dates_to_test[2])
        self.assertEqual(1, res.month)
        self.assertEqual(1990, res.year)
        res = detect_date(dates_to_test[3])
        self.assertEqual(2, res.month)
        self.assertEqual(2012, res.year)
        res = detect_date(dates_to_test[4])
        self.assertEqual(9, res.month)
        self.assertEqual(2020, res.year)

