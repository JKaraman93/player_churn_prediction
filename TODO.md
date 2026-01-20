# TODO List
## generate_sessions
- [ ] .withColumn("session_seq", F.explode(F.sequence(F.lit(1), F.col("daily_sessions")))) 
 Check if it works


## generate_transactions 
- [X] Try larger percentage of invalid financial transaction to see the change in silver dataset (16.01.26)
- [ ]  .withColumn("transaction_ts", F.current_timestamp())  select ts between start and end date like sessions

## create_bronze_dataset.py
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
- [X] Create a unified money-chnaging event table
- [X] Compute net_amount_7d, net_game_result_7d, balance_7d_ago, balance_change_30d
- [X] Compute failed_withdrawals_30d, deposit_count_30d, withdrawal_ratio	
- [ ] balance_change_7d  = current_balance - balance_7d_ago
- [ ] balance_change_30d = current_balance - balance_30d_ago
- [ ] Boolean /  no_sessions_7d   no_transactions_30d   no_deposits_30d


## ground_truth
rolling window of 7 inactive consecutive days (1 or 0)
for each day look if 1 exists in the next 7 days -> player will churn GOLD labels