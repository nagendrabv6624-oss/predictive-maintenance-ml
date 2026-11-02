# Data

Add your licensed machine sensor dataset as `maintenance.csv`.

Required columns: `air_temperature`, `process_temperature`, `rotational_speed`, `torque`, `tool_wear`, `failure`. The target must be 0/1.

No real industrial records are included. The training script uses synthetic demonstration data when no CSV is provided. Record the dataset source, license, sampling process and label definition in your project notes.
