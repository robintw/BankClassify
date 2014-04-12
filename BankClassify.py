import pandas as pd
from text.classifiers import NaiveBayesClassifier
import re
from colorama import init, Fore, Style
import dateutil

class BankClassify():

    def __init__(self, data="AllData.csv", incomedata="AllIncome.csv"):
        self.prev_data = pd.read_csv(data)
        self.classifier = NaiveBayesClassifier(self._get_training(self.prev_data), self._extractor)


    def add_data(self, filename):
        self.new_data, self.new_income = self._read_santander_file(filename)

        self._ask_with_guess(self.new_data)

        self.prev_data = pd.concat([self.prev_data, self.new_data])
        self.prev_data.to_csv("AllData.csv", index=False)

    def _prep_for_analysis(self):
        self.prev_data = self._make_date_index(self.prev_data)

        self.noignore = self.prev_data[self.prev_data.cat != 'Ignore']
        self.noexpignore = self.prev_data[(self.prev_data.cat != 'Ignore') & (self.prev_data.cat != 'Expenses')]


    def _ask_with_guess(self, df):
        init()

        df['cat'] = ""

        categories = {1: 'Bill',
                      2: 'Supermarket',
                      3: 'Cash',
                      4: 'Petrol',
                      5: 'Eating Out',
                      6: 'Travel',
                      7: 'Unclassified',
                      8: 'House',
                      9: 'Books',
                      10: 'Craft',
                      11: 'Charity Shop',         
                      12: 'Presents',
                      13: 'Toiletries',
                      14: 'Car',
                      15: 'Cheque',
                      16: 'Rent',
                      17: 'Paypal',
                      18: 'Ignore',
                      19: 'Expenses'
                      }

        for index, row in df.iterrows():
            #print Fore.GREEN + "-" * 72 + Fore.RESET

            

            # TODO: Make interface nicer
            # Ideas:
            # * Give list of categories at the end
            cats_list = ["%d: %s" % (id, cat) for id,cat in categories.iteritems()]
            new_list = []
            for item in cats_list:
                if len(item.split(":")[1].strip()) < 5:
                    new_list.append(item + "\t\t\t")
                else:
                    new_list.append(item + "\t\t")
            new_list[2::3] = map(lambda x: x+"\n", new_list[2::3])
            cats_joined = "".join(new_list)

            stripped_text = self._strip_numbers(row['desc'])

            if len(self.classifier.train_set) > 1:
                guess = self.classifier.classify(stripped_text)
            else:
                guess = ""


            # PRINTING STUFF
            print chr(27) + "[2J"
            print cats_joined
            print "\n\n"
            print "On: %s\t %.2f\n%s" % (row['date'], row['amount'], row['desc'])
            print Fore.RED  + Style.BRIGHT + "My guess is: " + guess + Fore.RESET

            res = raw_input("> ")

            if res.lower().startswith('q'):
                # Q = Quit
                return df
            if res == "":
                # Our guess was right!
                df.ix[index, 'cat'] = guess
                self.classifier.update([(stripped_text, guess)])
            else:
                # Our guess was wrong

                # Write correct answer
                df.ix[index, 'cat'] = categories[int(res)]
                # Update classifier
                self.classifier.update([(stripped_text, categories[int(res)])])

        return df

    def _make_date_index(self, df):
        df.index = pd.DatetimeIndex(df.date.apply(dateutil.parser.parse,dayfirst=True))

        return df

    def _read_santander_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()

        dates = []
        descs = []
        amounts = []

        for line in lines[4:]:
            #print line

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

        outgoings = df[df.amount < 0]
        income = df[df.amount > 0]

        outgoings['amount'] = -1 * df['amount']
        return outgoings, income

    def _get_training(self, df):
        train = []
        subset = df[df['cat'] != '']
        for i in subset.index:
            row = subset.ix[i]
            new_desc = self._strip_numbers(row['desc'])
            train.append( (new_desc, row['cat']) )

        # classifier = NaiveBayesClassifier(train)
        # return classifier
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