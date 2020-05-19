# Communicate Data Findings - Loan Data from Prosper
## by Abha Ramchandani


## Dataset

> The loan dataset has been provided in a CSV file. It contains 113,937 records and 81 variables about each loan data, including loan amount, borrower rate (or interest rate), current loan status, etc. It also contains those listings which didn't transformed into a loan. There is also information on partially funded loans in the dataset.

> My focus variables were - 'BorrowerAPR', 'DebtToIncomeRatio', 'StatedMonthlyIncome', 'LoanOriginalAmount''Term', 'ProsperRating (Alpha)'

> Basic cleaning was performed on the data to get rid of NULL / NA values.


## Summary of Findings

> - As ProsperRating gets better HR>AA, the BorrowerAPR decreases.
> - Generally speaking, for ProsperRating HR>C, BorrowerAPR generally decreses with increase in length of loan Term. This reverses as we go from ProsperRating B>AA.
> - 36-month Term has most number of loans, followed by 60-month and 12-month Terms.
> - There are more loans for lower LoanOriginalAmount in all the terms.
> - As LoanOriginalAmount increases BorrowerAPR decreases.tween BorrowerAPR and LoanOriginalAmount. But as LoanOriginalAmount increases BorrowerAPR decreases.
> - Higher the StatedMonthlyIncome, lower the BorrowerAPR.
> - Better ProsperRating means better BorrowerAPR
> - Better ProsperRating means higher loan amounts are approved


## Key Insights for Presentation

> The most pronounced relationship is between ProsperRating (Alpha) and BorrowerAPR. Other factors play a weaker role in determining BorrowerAPR, nonetheless, they cannot be neglected. The crux is that people with better ProsperRating get lower interest rates on their loans.

> Surprisingly, DebtToIncomeRatio did not have a meaningful correlation to interest rates. Other column behavioprs were as expected. I did not explore all the features in the dataset, but I thing it is a good idea to explore some more like IncomeRange, MonthlyPayment, EstimatedLoss, etc.