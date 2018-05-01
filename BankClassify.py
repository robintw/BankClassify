import pandas as pd
import numpy as np
from textblob.classifiers import NaiveBayesClassifier
import re
from colorama import init, Fore, Style
import dateutil
from tabulate import tabulate

class BankClassify():

    def __init__(self, data="AllData.csv"):
        self.prev_data = pd.read_csv(data)
        self.classifier = NaiveBayesClassifier(self._get_training(self.prev_data), self._extractor)

    def add_data(self, filename):
        self.new_data = self._read_santander_file(filename)

        self._ask_with_guess(self.new_data)

        self.prev_data = pd.concat([self.prev_data, self.new_data])
        self.prev_data.to_csv("AllData.csv", index=False)

    def _prep_for_analysis(self):
        self.prev_data = self._make_date_index(self.prev_data)

        self.prev_data['cat'] = self.prev_data['cat'].str.strip()

        self.inc = self.prev_data[self.prev_data.amount > 0]
        self.out = self.prev_data[self.prev_data.amount < 0]
        self.out.amount = self.out.amount.abs()

        self.inc_noignore = self.inc[self.inc.cat != 'Ignore']
        self.inc_noexpignore = self.inc[(self.inc.cat != 'Ignore') & (self.inc.cat != 'Expenses')]

        self.out_noignore = self.out[self.out.cat != 'Ignore']
        self.out_noexpignore = self.out[(self.out.cat != 'Ignore') & (self.out.cat != 'Expenses')]

    def _read_categories(self):
        categories = {}

        with open('categories.txt') as f:
            for i, line in enumerate(f.readlines()):
                categories[i] = line.strip()

        return categories

    def _add_new_category(self, category):
        with open('categories.txt', 'a') as f:
            f.write('\n' + category)

    def _ask_with_guess(self, df):
        # Initialise colorama
        init()

        df['cat'] = ""

        categories = self._read_categories()

        for index, row in df.iterrows():

            # Generate the category numbers table from the list of categories
            cats_list = [[idnum, cat] for idnum, cat in categories.items()]
            cats_table = tabulate(cats_list)

            stripped_text = self._strip_numbers(row['desc'])

            # Guess a category using the classifier (only if there is data in the classifier)
            if len(self.classifier.train_set) > 1:
                guess = self.classifier.classify(stripped_text)
            else:
                guess = ""


            # Print list of categories
            print(chr(27) + "[2J")
            print(cats_table)
            print("\n\n")
            # Print transaction
            print("On: %s\t %.2f\n%s" % (row['date'], row['amount'], row['desc']))
            print(Fore.RED  + Style.BRIGHT + "My guess is: " + str(guess) + Fore.RESET)

            input_value = input("> ")

            if input_value.lower() == 'q':
                # If the input was 'q' then quit
                return df
            if input_value == "":
                # If the input was blank then our guess was right!
                df.ix[index, 'cat'] = guess
                self.classifier.update([(stripped_text, guess)])
            else:
                # Otherwise, our guess was wrong
                try:
                    # Try converting the input to an integer category number
                    # If it works then we've entered a category
                    category_number = int(input_value)
                    category = categories[category_number]
                except ValueError:
                    # Otherwise, we've entered a new category, so add it to the list of
                    # categories
                    category = input_value
                    self._add_new_category(category)
                    categories = self._read_categories()

                # Write correct answer
                df.ix[index, 'cat'] = category
                # Update classifier
                self.classifier.update([(stripped_text, category)   ])

        return df

    def _make_date_index(self, df):
        df.index = pd.DatetimeIndex(df.date.apply(dateutil.parser.parse,dayfirst=True))

        return df

    def _read_santander_file(self, filename):
        with open(filename, errors='replace') as f:
            lines = f.readlines()

        dates = []
        descs = []
        amounts = []

        for line in lines[4:]:

            line = "".join(i for i in line if ord(i)<128)
            if line.strip() == '':
                continue

            splitted = line.split(":")

            category = splitted[0]
            data = ":".join(splitted[1:])

            if category == 'Date':
                dates.append(data.strip())
            elif category == 'Description':
                descs.append(data.strip())
            elif category == 'Amount':
                just_numbers = re.sub("[^0-9\.-]", "", data)
                amounts.append(just_numbers.strip())

        df = pd.DataFrame({'date':dates, 'desc':descs, 'amount':amounts})

        df['amount'] = df.amount.astype(float)
        df['desc'] = df.desc.astype(str)
        df['date'] = df.date.astype(str)

        return df

    def _get_training(self, df):
        train = []
        subset = df[df['cat'] != '']
        for i in subset.index:
            row = subset.ix[i]
            new_desc = self._strip_numbers(row['desc'])
            train.append( (new_desc, row['cat']) )

        return train

    def _extractor(self, doc):
        # TODO: Extend to extract words within words
        # For example, MUSICROOM should give MUSIC and ROOM
        tokens = self._split_by_multiple_delims(doc, [' ', '/'])

        features = {}

        for token in tokens:
            if token == "":
                continue
            features[token] = True

        return features

    def _strip_numbers(self, s):
        return re.sub("[^A-Z ]", "", s)

    def _split_by_multiple_delims(self, string, delims):
        regexp = "|".join(delims)

        return re.split(regexp, string)