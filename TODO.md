# TODO List

## create_bronze_dataset.py
- [X] Try larger percentage of invalid financial transaction to see the change in silver dataset
- [X] risk segment "unknown" for new players (16.01.26)
- [X] no sessions for new players (16.01.26)


## create_silver_dataset.py
- [x] All transactions must happen after player registration (12.01.25)
- [X] All transactions must respect player balance (12.01.25)
- [X] Deleting of invalid transactions using Pandas UDF  (13.01.26)
- [x] Update session and transaction dataframes with after_txn balance (15.01.26)
- [x] Create a column with recent balance for each player in player dataframe (15.01.26)
- [ ] Drop invalid sessions when bet>balance with win result.

## create_gold_dataset.py

- [X] last_session_date = null for new players (16.01.26)


