import pandas as pd
from text.classifiers import NaiveBayesClassifier, DecisionTreeClassifier
import re
from colorama import init, Fore, Style
import dateutil

# TODO: Add removal of non-ASCII characters first

def read_santander_file(filename):
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

	df = df[df.amount < 0]

	df['amount'] = -1 * df['amount']
	return df

def make_date_index(df):
	df.index = pd.DatetimeIndex(df.date.apply(dateutil.parser.parse,dayfirst=True))
	del df['date']

	return df

def ask_with_guess(df, c=None):
	init()

	df['cat'] = ""

	if c is None:
		c = NaiveBayesClassifier([], extractor)

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

		stripped_text = strip_numbers(row['desc'])

		if len(c.train_set) > 1:
			guess = c.classify(stripped_text)
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
			return df,c
		if res == "":
			# Our guess was right!
			df.ix[index, 'cat'] = guess
			c.update([(stripped_text, guess)])
		else:
			# Our guess was wrong

			# Write correct answer
			df.ix[index, 'cat'] = categories[int(res)]
			# Update classifier
			c.update([(stripped_text, categories[int(res)])])

	return df,c

def strip_numbers(s):
	return re.sub("[^A-Z ]", "", s)

def get_training(df):
	train = []
	subset = df[df['cat'] != '']
	for i in subset.index:
		row = subset.ix[i]
		new_desc = strip_numbers(row['desc'])
		train.append( (new_desc, row['cat']) )

	# classifier = NaiveBayesClassifier(train)
	# return classifier
	return train

def classify(df, c):
	df['cat'] = None

	for i in df.index:
		df.ix[i, 'cat'] = c.classify(df.ix[i, 'desc'])

def classifier_from_labelled_csv(filename):
	df = pd.read_csv(filename)

	return NaiveBayesClassifier(get_training(df), extractor)

def split_by_multiple_delims(string, delims):
	regexp = "|".join(delims)

	return re.split(regexp, string)

def extractor(doc):
	# TODO: Extend to extract words within words
	# For example, MUSICROOM should give MUSIC and ROOM
	tokens = split_by_multiple_delims(doc, [' ', '/'])

	features = {}

	for token in tokens:
		if token == "":
			continue
		features[token] = True

	return features

def by_month(x):
    return (x.resample('M', sum))

def group_by_cats_and_month(df):
	g = df.groupby('cat')
	return g.apply(by_month).unstack()