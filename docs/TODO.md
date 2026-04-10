# TODO List

- [X] Refine draft.ipynb, split it into sections

## Threshold selection 
- [ ] For a player with predicted churn probability p: EV=p×r×S−C

## General 
- [X] Transfer your code from .ipynb to .py files 
- [X] df_sessions_rolling : convert null values to 0 
- [X] old_player_behavior : have to pass the var first event to filter the first 30days
- [ ] more assertions about data consistency 
- [X] multiple sessions/transactions per day  cause inconsistency between inference and training 
- [X] find who players are in inference dataset and not in training, and then explore in which stage they are excluded from the training dataset -> there is difference because in training the related rows are sum, so the results will be 0 while in inference the zero rows are excluded. 
- [X] transactions has full date including minutes which cause prooblem compairing with a specific date e.g 2025-06-20 05:30 > 2025-06-20 i dont want it
- [X] #.filter(F.datediff(F.col("reference_date"), F.col("first_event_date")) > 30) it cause discrepancy between inference and training -> if comment this create columnswith null values -> why? -> because cant compute balances later  than 30 days


## draft.ipynb
- [X] data_inference = prepare_data_inference('2024-06-25') must have null values only in transactions columns

## batch_inference.py
- [X] check if null values should be replaced by 0

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
- [X] convert ohe derived features importance to categorical feature importance 
- [X] store the exported items like figures and tables include in mlflow in a separated folder 
- [X] **produce player_idx | p_churn | risk_level | scoring_date**

## generate_sessions.py
- [X] .withColumn("session_seq", F.explode(F.sequence(F.lit(1), F.col("daily_sessions")))) 
 Check if it works

## generate_transactions.py
- [X] Try larger percentage of invalid financial transaction to see the change in silver dataset (16.01.26)
- [X] .withColumn("transaction_ts", F.current_timestamp())  select ts between start and end date like sessions
- [ ] Each player complete only one transaction / should be more.

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


