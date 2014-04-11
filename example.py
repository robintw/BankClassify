from classify import *

# Create a classifier from previously-labelled data
c = classifier_from_labelled_csv('FromJan2013_Classified.csv')

# Import Statements from Santander
df = read_santander_file("/Users/robin/Downloads/Statements09012800760389 (1).txt")

# Do the asking for categories, with Bayesian guesses
ask_with_guess(df, c)
# Save results!
df.to_csv("Sep2013-Apr2014.csv", index=False)

# Do bits of analysis
group_by_cats_and_month(df).ix['Bill']
group_by_cats_and_month(df).ix['Bill'].plot()
group_by_cats_and_month(df).ix['Supermarket']
group_by_cats_and_month(df).ix['Supermarket'].plot()
df.apply(by_month)
clf()
df.apply(by_month).plot()
df[df.index > pd.Timestamp('2013-10-01')]
df[(df.index > pd.Timestamp('2013-10-01')) & (df.index < pd.Timestamp('2013-11-01'))]
oct = df[(df.index > pd.Timestamp('2013-10-01')) & (df.index < pd.Timestamp('2013-11-01'))]

oct.groupby('cat').sum()