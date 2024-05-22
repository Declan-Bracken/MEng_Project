import json
from pathlib import Path

class SubjectKeywordsManager:
    def __init__(self, titles_path='Dashboard/Transcript_Processor/subjects/subject_keywords_titles.json', codes_path='Dashboard/Transcript_Processor/subjects/subject_keywords_codes.json'):
        self.titles_path = Path(titles_path)
        self.codes_path = Path(codes_path)
        self.subject_keywords_titles = self.load_keywords(self.titles_path)
        self.subject_keywords_codes = self.load_keywords(self.codes_path)

    def load_keywords(self, path):
        if path.exists():
            with open(path, 'r') as file:
                return json.load(file)
        return {}

    def save_keywords(self, keywords, path):
        with open(path, 'w') as file:
            json.dump(keywords, file, indent=4)

    def save(self):
        self.save_keywords(self.subject_keywords_titles, self.titles_path)
        self.save_keywords(self.subject_keywords_codes, self.codes_path)

    def add_subject(self, subject, type='title'):
        if type == 'title':
            if subject not in self.subject_keywords_titles:
                self.subject_keywords_titles[subject] = []
        elif type == 'code':
            if subject not in self.subject_keywords_codes:
                self.subject_keywords_codes[subject] = []

    def add_sub_subject(self, subject, sub_subject, type='title'):
        if type == 'title':
            if subject in self.subject_keywords_titles:
                self.subject_keywords_titles[subject].append(sub_subject)
        elif type == 'code':
            if subject in self.subject_keywords_codes:
                self.subject_keywords_codes[subject].append(sub_subject)

# Example usage:
if __name__ == "__main__":
    manager = SubjectKeywordsManager()
    manager.add_subject('Chemistry', type='title')
    manager.add_sub_subject('Mathematics', 'Real Analysis', type='title')
    manager.save()
    # print(manager.subject_keywords_titles)
