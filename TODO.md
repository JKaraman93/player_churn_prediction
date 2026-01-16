# TODO List

## create_bronze_dataset.py
- [ ] Try larger percentage of invalid financial transaction to see the change in silver dataset


## create_silver_dataset.py
- [x] All transactions must happen after player registration (12.01.25)
- [X] All transactions must respect player balance (12.01.25)
- [X] Deleting of invalid transactions using Pandas UDF / No account for problems whilebet > balance and win.  (13.01.26)
- [x] Update session and transaction dataframes with after_txn balance (15.01.26)
- [x] Create a column with recent balance for each player in player dataframe (15.01.26)
- [ ] Create a unified “money events” stream
        player_id
        event_id
        event_ts
        event_type   (deposit, withdrawal, session)
        signed_amount
        balance_after_txn
- [ ] Gold data



Generate raw transactions in Bronze
Ignore balance feasibility at this stage (for simplicity)
But record timestamps

Silver transformation:
Join with registration_date → drop transactions/sessions before registration
Compute running balance → drop infeasible bets/withdrawals
Normalize signed amounts
Compute balance_after_txn

Silver sessions:
Drop sessions before registration
Optional: drop sessions with zero duration
s
