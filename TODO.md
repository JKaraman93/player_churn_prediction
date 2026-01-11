# TODO List

## create_silver_dataset.py
- [ ] All transactions must happen after player registration
- [ ] All transactions must respect player balance
- [ ] The balance must be updated after every money-changing event

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
