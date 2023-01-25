import re
import dateutil
import os
from datetime import datetime

import pandas as pd
from nltk.classify import NaiveBayesClassifier
from colorama import init, Fore, Style
from tabulate import tabulate


class BankClassify():

    def __init__(self, data="AllData.csv"):
        """Load in the previous data (by default from `data`) and initialise the classifier"""

        # allows dynamic training data to be used (i.e many accounts in a loop)
        self.trainingDataFile = data

        if os.path.exists(data):
            self.prev_data = pd.read_csv(self.trainingDataFile)
        else:
            self.prev_data = pd.DataFrame(
                columns=['date', 'desc', 'amount', 'cat'])
        # Prepare the dataset for training
        dataset = [(self._extractor(row['desc']), row['cat'])
                   for _, row in self.prev_data.iterrows()]
        self.classifier = NaiveBayesClassifier.train(dataset)

    def add_data(self, filename, bank="usbank"):
        """Add new data and interactively classify it.

        Arguments:
         - filename: filename of Travel - Petrol file
        """
        if bank == "usbank":
            print("adding usbank data!")
            self.new_data = self._read_usbank_file(filename)
        elif bank == "boa":
            print("adding boa data!")
            self.new_data = self._read_boa_file(filename)
        elif bank == "chase":
            print("adding chase Bank data!")
            self.new_data = self._read_chase_file(filename)
        elif bank == "blockfi":
            print("adding blockfi Bank data!")
            self.new_data = self._read_blockfi_file(filename)
        else:
            raise ValueError(
                'new_data appears empty! probably tried an unknown bank: ' + bank)

        self._ask_with_guess(self.new_data)

        self.prev_data = pd.concat([self.prev_data, self.new_data])
        # save data to the same file we loaded earlier
        self.prev_data.to_csv(self.trainingDataFile, index=False)

    def _prep_for_analysis(self):
        """Prepare data for analysis in pandas, setting index types and subsetting"""
        self.prev_data = self._make_date_index(self.prev_data)

        self.prev_data['cat'] = self.prev_data['cat'].str.strip()

        self.inc = self.prev_data[self.prev_data.amount > 0]
        self.out = self.prev_data[self.prev_data.amount < 0]
        self.out.amount = self.out.amount.abs()

        self.inc_noignore = self.inc[self.inc.cat != 'Ignore']
        self.inc_noexpignore = self.inc[(
            self.inc.cat != 'Ignore') & (self.inc.cat != 'Expenses')]

        self.out_noignore = self.out[self.out.cat != 'Ignore']
        self.out_noexpignore = self.out[(
            self.out.cat != 'Ignore') & (self.out.cat != 'Expenses')]

    def _read_categories(self):
        """Read list of categories from categories.txt"""
        categories = {}

        with open('categories.txt') as f:
            for i, line in enumerate(f.readlines()):
                categories[i] = line.strip()

        return categories

    def _add_new_category(self, category):
        """Add a new category to categories.txt"""
        with open('categories.txt', 'a') as f:
            f.write('\n' + category)

    def _ask_with_guess(self, df):
        """Interactively guess categories for each transaction in df, asking each time if the guess
        is correct"""
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
            # if len(self.classifier.train_set) > 1:
            #     guess = self.classifier.classify(stripped_text)
            # else:
            #     guess = ""
            guess = self.classify(stripped_text)

            # Print list of categories
            print(chr(27) + "[2J")
            print(cats_table)
            print("\n\n")
            # Print transaction
            print("On: %s\t %.2f\n%s" %
                  (row['date'], row['amount'], row['desc']))
            print(Fore.RED + Style.BRIGHT +
                  "My guess is: " + str(guess) + Fore.RESET)

            input_value = input("> ")

            if input_value.lower() == 'q':
                # If the input was 'q' then quit
                return df
            if input_value == "":
                # If the input was blank then our guess was right!
                df.at[index, 'cat'] = guess
                train = self._get_training(self.prev_data)
                self.classifier = NaiveBayesClassifier.train(train)

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
                df.at[index, 'cat'] = category
                # Update classifier
                train = self._get_training(self.prev_data)
                self.classifier = NaiveBayesClassifier.train(train)

        return df

    def _make_date_index(self, df):
        """Make the index of df a Datetime index"""
        df.index = pd.DatetimeIndex(df.date.apply(
            dateutil.parser.parse, dayfirst=True))

        return df

    def _read_usbank_file(self, filename):
        """Read a file in the csv file that usbank provides downloads in.

        Returns a pd.DataFrame with columns of 'date', 'desc' and 'amount'."""

        with open(filename) as f:
            lines = f.readlines()

            dates = []
            descs = []
            amounts = []

            for line in lines[1:]:

                line = "".join(i for i in line if ord(i) < 128)
                if line.strip() == '':
                    continue

                splits = line.split("\",\"")
                """
                0 = Date yyyy-mm-dd
                1 = Transaction type
                2 = Name
                3 = Memo
                4 = Amount
                """
                date = splits[0].replace("\"", "").strip()
                date = datetime.strptime(date, '%Y-%m-%d').strftime('%d/%m/%Y')
                dates.append(date)

                amounts.append(splits[4].replace('\n', '').replace("\"", ""))

                # Description
                descs.append(splits[2] + splits[3])

            df = pd.DataFrame(
                {'date': dates, 'desc': descs, 'amount': amounts})

            df['amount'] = df.amount.astype(float)
            df['desc'] = df.desc.astype(str)
            df['date'] = df.date.astype(str)

        return df

    def _read_boa_file(self, filename):
        """Read a file in the plain text format that boa provides downloads in.

        Returns a pd.DataFrame with columns of 'date', 'desc' and 'amount'."""
        with open(filename, errors='replace') as f:
            lines = f.readlines()

        dates = []
        descs = []
        amounts = []

        for line in lines[1:]:

            line = "".join(i for i in line if ord(i) < 128)
            if line.strip() == '':
                continue

            splits = line.split(",")
            """
            0 = Date mm/dd/yyy
            1 = Reference Number
            2 = payee
            3 = address
            4 = Amount
            """

            date = splits[0].strip()
            date = datetime.strptime(date, '%m/%d/%Y').strftime('%d/%m/%Y')
            dates.append(date)

            amounts.append(splits[4])

            # Description
            descs.append(splits[2].replace("\"", '') +
                         splits[3].replace("\"", ''))

        df = pd.DataFrame({'date': dates, 'desc': descs, 'amount': amounts})

        df['amount'] = df.amount.astype(float)
        df['desc'] = df.desc.astype(str)
        df['date'] = df.date.astype(str)

        return df

    def _read_blockfi_file(self, filename):
        """Read a file in the plain text format that blockfi provides downloads in.

        Returns a pd.DataFrame with columns of 'date', 'desc' and 'amount'."""

        with open(filename) as f:
            lines = f.readlines()

        dates = []
        descs = []
        amounts = []

        for line in lines:

            line = "".join(i for i in line if ord(i) < 128)
            if line.strip() == '':
                continue

            splits = line.split("\",\"")
            """
            0 = NNNN first four of cc
            1 = Date mm-dd-yy
            2 = tx date
            3 = Posted date
            4 = Description
            5 = Amount
            """
            date = splits[3].replace("\"", "").strip()
            date = datetime.strptime(date, '%m/%d/%y').strftime('%d/%m/%Y')
            dates.append(date)

            amounts.append(splits[5].replace(
                '\n', '').replace("\"", "").replace(",", ""))

            # Description
            descs.append(splits[4])

        df = pd.DataFrame({'date': dates, 'desc': descs, 'amount': amounts})

        df['amount'] = df.amount.astype(float)
        df['desc'] = df.desc.astype(str)
        df['date'] = df.date.astype(str)

        return df

    def _get_training(self, df):
        """Get training data for the classifier, consisting of tuples of
        (text, category)"""
        train = []
        subset = df[df['cat'] != '']
        for i in subset.index:
            row = subset.iloc[i]
            new_desc = self._strip_numbers(row['desc'])
            features = self._extractor(new_desc)
            train.append((features, row['cat']))

        return train

    def _extractor(self, doc):
        """Extract tokens from a given string"""
        # TODO: Extend to extract words within words
        # For example, MUSICROOM should give MUSIC and ROOM
        tokens = self._split_by_multiple_delims(doc, [' ', '/', '.'])

        features = {}

        for token in tokens:
            if token == "":
                continue
            features[token] = True

        return features

    def _strip_numbers(self, s):
        """Strip numbers from the given string"""
        return re.sub("[^A-Z ]", "", s)

    def _split_by_multiple_delims(self, string, delims):
        """Split the given string by the list of delimiters given"""
        regexp = "|".join(delims)

        return re.split(regexp, string)

    def classify(self, text):
        stripped_text = self._strip_numbers(text)
        features = self._extractor(stripped_text)
        guess = self.classifier.classify(features)
        return guess
