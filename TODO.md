# TODO List

## create_silver_dataset.py
- [x] All transactions must happen after player registration
- [X] All transactions must respect player balance
- [ ] Handling of invalid transactions(sessions must be deleted, while financial change success_flag value )
- [ ] Update session and transaction dataframes with after_txn balance
- [ ] Create a column with recent balance for each player in player dataframe
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
