# TODO List

## General 
- [ ] Transfer your code from .ipynb to .py files 
- [X] df_sessions_rolling : convert null values to 0 
- [X] old_player_behavior : have to pass the var first event to filter the first 30days
- [ ] more assertions about data consistency 


## Logistic Regrsession
- [X] lr_model = model.stages[-1]
max(abs(lr_model.coefficients.toArray()))
If you see extreme values (>50), scaling or regularization is off.
- [X] construct a pipeline
- [X] finetune hyperparameter  (gridsearch)
- [X] churn threshold
- [X] CV kfold=3 makes senese ? 
- [X] How can you handle imbalance data ? weightCol -> no resonable improvement
- [X] check test run : best_threshold
- [ ] Drop highly correlated features

## generate_sessions.py
- [X] .withColumn("session_seq", F.explode(F.sequence(F.lit(1), F.col("daily_sessions")))) 
 Check if it works


## generate_transactions.py
- [X] Try larger percentage of invalid financial transaction to see the change in silver dataset (16.01.26)
- [X]  .withColumn("transaction_ts", F.current_timestamp())  select ts between start and end date like sessions
- [ ] Each player complete only one session / should be more.

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


## gold_data_generation.ipynb
- [X] rolling window of 7 inactive consecutive days (1 or 0)
for each day look if 1 exists in the next 7 days -> player will churn GOLD labels
- [X] create data for ML/ beforehand or before fitting

    3. One important clarification (not a bug, but critical)
    ⚠ Your churn_7d definition includes the current day
    You defined:
    “inactive for the last 7 days, including the current day”
    This is valid, but you must be consistent everywhere.
    That means:
    num_sessions_7d == 0 → churn state
    next_7d_churn == true means
    “there exists a future day where churn_7d == true”
    ✔ This is acceptable
    ✔ Just document it clearly in your README / notebook
    Many teams instead define:
    inactivity window = previous 7 full days excluding today
    But your choice is consistent and defensible.


