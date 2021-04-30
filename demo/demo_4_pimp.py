from keras_en_parser_and_analyzer.library.pipmp_my_cv_classify import ResumeParser, read_pdf
from keras_en_parser_and_analyzer.library.prepare_train_dataset import TextPreprocessor
import pandas as pd


def pre_analyze_cv(file_name: str):
    """
    Pre classify into Experience, Education and Skills groups
    """
    parser = ResumeParser()
    parser.load_model('models')
    rows = read_pdf(file_name)[file_name]
    preds = parser.parse(rows)

    df = pd.DataFrame(preds)
    parser.define_header_lines(df)

predictions = pre_analyze_cv('data/resume_samples/502661448+Daniil.pdf')
print(predictions)
