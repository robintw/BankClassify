from bankclassify.BankClassify import BankClassify

bc = BankClassify(data="budget2023.csv")

bc.add_data("2023-01-03 thru 2023-04-02 transactions.csv", "personal_capital")