from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from math import log
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
import os
import platform
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import Screen
from kivymd.app import MDApp
from kivymd.uix.filemanager import MDFileManager

nltk.download("punkt")
nltk.download("stopwords")

class TermsLensApp(MDApp):
    # Integration of elements into the graphic interface.
    file_manager = None
    def build(self):
        self.terms = pd.read_csv("data.csv", encoding="utf-8")
        self.terms.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)
        self.terms.rename(columns={"v1": "labels", "v2": "message"}, inplace=True)
        self.terms["label"] = self.terms["labels"].map({"neutral": 0, "suspicious": 1})
        self.terms.drop(["labels"], axis=1, inplace=True)

        self.trainIndex, self.testIndex = list(), list()
        for i in range(self.terms.shape[0]):
            if np.random.uniform(0, 1) < 0.75:
                self.trainIndex += [i]
            else:
                self.testIndex += [i]
        self.trainData = self.terms.loc[self.trainIndex]
        self.testData = self.terms.loc[self.testIndex]

        self.trainData.reset_index(inplace=True)
        self.trainData.drop(["index"], axis=1, inplace=True)

        self.testData.reset_index(inplace=True)
        self.testData.drop(["index"], axis=1, inplace=True)

        return Builder.load_file('interface.kv')

    # Function to save the output in a txt file.
    def save_results_to_file(self, lines):
        with open("termslens_analysis_filter_for_suspect_content.txt", "w") as file:
            file.write("\n".join(lines))
        system_name = platform.system()
        if system_name == "Windows":
            os.system("start notepad termslens_analysis_filter_for_suspect_content.txt")  # Opening the file with notepad on Windows.
        elif system_name == "Linux":
            os.system("xdg-open termslens_analysis_filter_for_suspect_content.txt")  # Opening the file with the default app on Linux.
        elif system_name == "Darwin":  # macOS
            os.system("open termslens_analysis_filter_for_suspect_content.txt")  # Opening the file with the default app on MacOS.

    # Set of classes to manage the file explorer to select the pdf file (it took me a while to get all this to work ;)).
    def show_file_manager(self):
        if not self.file_manager:
            self.file_manager = MDFileManager(
                exit_manager=self.exit_file_manager,  # Method to call when closing the file explorer.
                select_path=self.select_path,  # Method to call when selecting a file.
            )
        self.file_manager.show('/')  # Opening the file manager at the root.

    def exit_file_manager(self, *args):
        self.file_manager.close()

    def select_path(self, path):
        # Processing the pdf file path.
        print("Selected path:", path)
        # Method to call when analyzing the pdf file.
        self.analyze_pdf(path)

    def analyze_pdf(self, pdf_path):
        # Algorithm to call when analyzing pdf file.
        sc_tf_idf = TFIDFCLassifier(self.trainData)
        sc_tf_idf.train()
        # Extract text from pdf file.
        pdf_text = extract_text_from_pdf(pdf_path)
        lines = pdf_text.split('\n')
        suspicious_lines = []
        # Saving output inside a txt file.
        for line in lines:
            processed_line = process_message(line)
            is_suspicious = sc_tf_idf.classify(processed_line)
            if is_suspicious:
                suspicious_lines.append(line)

        self.save_results_to_file(suspicious_lines)

        output_label = self.root.ids.output_label
        output_label.text = ("The suspect content has been filtered and saved in "
                             "a txt file."
                             "This has now been opened.")

    def classify_text(self, text):
        # Algorithm to call when classifying text.
        sc_tf_idf = TFIDFCLassifier(self.trainData)
        sc_tf_idf.train()

        processed_text = process_message(text)
        is_suspicious = sc_tf_idf.classify(processed_text)

        output_label = self.root.ids.output_label
        output_label.text = "Suspicious? : " + str(is_suspicious)

    def test_algorithm(self):
        # Algorithm to call when testing the algorithm.
        sc_tf_idf = TFIDFCLassifier(self.trainData)
        sc_tf_idf.train()
        preds_tf_idf = sc_tf_idf.predict(self.testData["message"])
        pass
        precision, recall, fscore, accuracy = self.calculate_metrics(self.testData["label"], preds_tf_idf)

        # Showing statistics inside gui.
        output_label = self.root.ids.output_label
        output_label.text = f"Precision: {precision}\nRecall: {recall}\nF-score: {fscore}\nAccuracy: {accuracy}"

    def calculate_metrics(self, labels, predictions):
        true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
        for i in range(len(labels)):
            true_pos += int(labels.get(i) == 1 and predictions.get(i) == 1)
            true_neg += int(labels.get(i) == 0 and predictions.get(i) == 0)
            false_pos += int(labels.get(i) == 0 and predictions.get(i) == 1)
            false_neg += int(labels.get(i) == 1 and predictions.get(i) == 0)
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        fscore = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg) if (true_pos + true_neg + false_pos + false_neg) > 0 else 0
        return precision, recall, fscore, accuracy

# Extraction function for the pdf file.
def extract_text_from_pdf(pdf_file):
    with open(pdf_file, 'rb') as file:
        pdf_reader = PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

def process_message(
    message, lower_case=True, stem=True, stop_words=True, gram=1
):
    """
This function transforms messages into a list of "stemmed" keywords, 
which is the heart of the algorithm's operation. If the "gram" value 
is less than 1 (gram > 1), not keywords but pairs of keywords will 
be taken into account. This can vary the efficiency of data processing.
    """
    if lower_case:
        message = message.lower()
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 2]
    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [" ".join(words[i : i + gram])]
        return w
    if stop_words:
        sw = stopwords.words("english")
        words = [word for word in words if word not in sw]
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    return words


class suspicious_terms_classifier:
    def __init__(self, trainData):
        self.terms, self.labels = trainData["message"], trainData["label"]

    def train(self):
        pass

    def classify(self, message):
        pass

    def calc_TF_and_IDF(self):
        number_of_lines = self.terms.shape[0]  # Defines the number of lines.
        self.suspicious_terms, self.neutral_terms = (
            self.labels.value_counts()[1],
            self.labels.value_counts()[0],
        )
        self.total_terms = self.suspicious_terms + self.neutral_terms
        self.suspicious_words = 0
        # Number of words flagged as suspect.
        self.neutral_words = 0
        # Number of words flagged as neutral.
        self.tf_suspicious = dict()
        # Dictionary with the TF of each word in the suspect data.
        self.tf_neutral = dict()
        # Dictionary with the TF of each word in the neutral data.
        self.idf_suspicious = dict()
        # Dictionary with the IDF of each word in the suspect data.
        self.idf_neutral = dict()

        # Dictionary with the IDF of each word in the neutral data.
        for i in range(number_of_lines):
            # Invoking nltk libraries :)
            message_processed = process_message(self.terms.get(i))
            count = list()
            # Save whether or not a word has appeared in the message.
            # IDF
            for word in message_processed:
                if self.labels[i]:
                    self.tf_suspicious[word] = self.tf_suspicious.get(word, 0) + 1
                    self.suspicious_words += 1
                    # Calculates the TF of a word in the suspect data.
                else:
                    self.tf_neutral[word] = self.tf_neutral.get(word, 0) + 1
                    self.neutral_words += 1
                    # Calculates the TF of a word in the neutral data.
                if word not in count:
                    count += [word]
            for word in count:
                if self.labels[i]:
                    self.idf_suspicious[word] = self.idf_suspicious.get(word, 0) + 1
                    # Calculates the idf (the number of suspect data items containing this word).
                else:
                    self.idf_neutral[word] = self.idf_neutral.get(word, 0) + 1
                    # Calculates the idf (the number of neutral data items containing this word).

    def predict(self, testData):
        # Invokes the classifier for Test Set messages
        result = dict()
        for (i, message) in enumerate(testData):
            processed_message = process_message(message)
            result[i] = int(self.classify(processed_message))
        return result


class TFIDFCLassifier(suspicious_terms_classifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        self.calc_TF_and_IDF()
        self.calc_TF_IDF()

    def calc_TF_IDF(self):
        # Performs the overall calculation (tf_idf).
        self.prob_suspicious = dict()
        self.prob_neutral = dict()
        self.sum_tf_idf_suspicious = 0
        self.sum_tf_idf_neutral = 0
        for word in self.tf_suspicious:
            self.prob_suspicious[word] = self.tf_suspicious[word] * log(
                (self.suspicious_terms + self.neutral_terms)
                / (self.idf_suspicious[word] + self.idf_neutral.get(word, 0))
            )
            self.sum_tf_idf_suspicious += self.prob_suspicious[word]

        for word in self.tf_suspicious:
            self.prob_suspicious[word] = (self.prob_suspicious[word] + 1) / (
                self.sum_tf_idf_suspicious + len(self.prob_suspicious.keys())
            )

        for word in self.tf_neutral:
            self.prob_neutral[word] = (self.tf_neutral[word]) * log(
                (self.suspicious_terms + self.neutral_terms)
                / (self.idf_suspicious.get(word, 0) + self.idf_neutral[word])
            )
            self.sum_tf_idf_neutral += self.prob_neutral[word]

        for word in self.tf_neutral:
            self.prob_neutral[word] = (self.prob_neutral[word] + 1) / (
                self.sum_tf_idf_neutral + len(self.prob_neutral.keys())
            )

        self.prob_suspicious_entry, self.prob_neutral_entry = (
            self.suspicious_terms / self.total_terms,
            self.neutral_terms / self.total_terms,
        )

    def classify(self, processed_message):
        # Data classification.
        pSpam, pHam = 0, 0
        for word in processed_message:
            if word in self.prob_suspicious:
                pSpam += log(self.prob_suspicious[word])
            else:
                pSpam -= log(self.sum_tf_idf_suspicious + len(self.prob_suspicious.keys()))
            if word in self.prob_neutral:
                pHam += log(self.prob_neutral[word])
            else:
                pHam -= log(self.sum_tf_idf_neutral + len(self.prob_neutral.keys()))
            pSpam += log(self.prob_suspicious_entry)
            pHam += log(self.prob_neutral_entry)
        return pSpam >= pHam


class BowClassifier(suspicious_terms_classifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        self.calc_TF_and_IDF()
        self.calc_prob()

    def calc_prob(self):
        self.prob_suspicious = dict()
        self.prob_neutral = dict()
        for word in self.tf_suspicious:
            # calcule la proba qu'un mot apparaisse dans les spams
            self.prob_suspicious[word] = (self.tf_suspicious[word] + 1) / (
                self.suspicious_words + len(self.tf_suspicious.keys())
            )
        for word in self.tf_neutral:
            # calcule la proba qu'un mot apparaisse dans les non spams
            self.prob_neutral[word] = (self.tf_neutral[word] + 1) / (
                self.neutral_words + len(self.tf_neutral.keys())
            )
        self.prob_suspicious_entry, self.prob_neutral_entry = (
            self.suspicious_terms / self.total_terms,
            self.neutral_terms / self.total_terms,
        )

    def classify(self, processed_message):
        # Data classification.
        pSpam, pHam = 0, 0
        for word in processed_message:
            if word in self.prob_suspicious:
                pSpam += log(self.prob_suspicious[word])
            else:
                pSpam -= log(self.suspicious_words + len(self.prob_suspicious.keys()))
            if word in self.prob_neutral:
                pHam += log(self.prob_neutral[word])
            else:
                pHam -= log(self.neutral_words + len(self.prob_neutral.keys()))
            pSpam += log(self.prob_suspicious_entry)
            pHam += log(self.prob_neutral_entry)
        return pSpam >= pHam


def metrics(labels, predictions):  # Calculates metrics:
    # True Positive, True Negative, False Positive, False Negative.
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    for i in range(len(labels)):
        true_pos += int(labels.get(i) == 1 and predictions.get(i) == 1)
        true_neg += int(labels.get(i) == 0 and predictions.get(i) == 0)
        false_pos += int(labels.get(i) == 0 and predictions.get(i) == 1)
        false_neg += int(labels.get(i) == 1 and predictions.get(i) == 0)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    Fscore = 2 * precision * recall / (precision + recall)
    accuracy = (true_pos + true_neg) / (
        true_pos + true_neg + false_pos + false_neg
    )

    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F-score: ", Fscore)
    print("Accuracy: ", accuracy)

if __name__ == "__main__":
    TermsLensApp().run()