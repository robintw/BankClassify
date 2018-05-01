# BankClassify - automatically classify your bank statement entries

**Note:** This is not 'finished' software. I use it for dealing with my bank statements, but it is not 'production-ready' and may crash or do strange things. It is also set up for my particular usage, so may not work for you. However, I hope it will be a useful resource.

This code will classify each entry in your bank statement into categories such as 'Supermarket', 'Petrol', 'Eating Out' etc. It learns from previously classified data, and corrections you make when it guesses a category incorrectly, and improves its performance over time.

## How to use
1. Install the required libraries:
  `pip install -r requirements.txt`

2. Run the code in `example.py` as a demonstration. This will interactively classify the example bank statement data in `Statement_Example.txt` and save the results in `AllData.csv`. In the interactive classification you will be presented with a list of categories (with ID numbers), the details of a transaction, and a guessed category. You have three choices:
   - To accept the guessed category, just press `Enter`
   - To correct the classifier to a category that is in the list shown, enter the ID number of the category and press `Enter`
   - To add a new category, type the name of the category and press `Enter`

3. Examine the output in `AllData.csv` manually, or run `bc._prep_for_analysis()` and look at `bc.in` and `bc.out` for incomings and outgoings respectively. You will see there is a `cat` column with the category in it.

To use it with your own data:

- *If you use Santander UK as your bank:* just run `bc.add_data(filename)` with the filename of your downloaded statement file. Delete `AllData.py` first though, or the example data will be used as part of the training data.
- *If you use another bank:* Write your own function to read in your statement data from your bank. It must return a pandas dataframe with columns of `date`, `desc` and `amount`. Add this to the `BankClassify` class and call it instead of `_read_santander_file`.

