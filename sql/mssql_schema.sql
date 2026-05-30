use Finance_Project;
go

CREATE TABLE fraud_transactions (
    transaction_id INT IDENTITY(1,1) PRIMARY KEY,
    transaction_time FLOAT NULL,
    amount FLOAT NULL,
    v1 FLOAT NULL,
    v2 FLOAT NULL,
    v3 FLOAT NULL,
    v4 FLOAT NULL,
    v5 FLOAT NULL,
    v6 FLOAT NULL,
    v7 FLOAT NULL,
    v8 FLOAT NULL,
    v9 FLOAT NULL,
    v10 FLOAT NULL,
    v11 FLOAT NULL,
    v12 FLOAT NULL,
    v13 FLOAT NULL,
    v14 FLOAT NULL,
    v15 FLOAT NULL,
    v16 FLOAT NULL,
    v17 FLOAT NULL,
    v18 FLOAT NULL,
    v19 FLOAT NULL,
    v20 FLOAT NULL,
    v21 FLOAT NULL,
    v22 FLOAT NULL,
    v23 FLOAT NULL,
    v24 FLOAT NULL,
    v25 FLOAT NULL,
    v26 FLOAT NULL,
    v27 FLOAT NULL,
    v28 FLOAT NULL,
    class_label INT NULL
);

CREATE TABLE stock_prices (
    id INT IDENTITY(1,1) PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    trade_date DATE NOT NULL,
    open_price FLOAT NULL,
    high_price FLOAT NULL,
    low_price FLOAT NULL,
    close_price FLOAT NULL,
    adj_close FLOAT NULL,
    volume BIGINT NULL
);

CREATE TABLE macro_indicators (
    id INT IDENTITY(1,1) PRIMARY KEY,
    indicator_name VARCHAR(100) NOT NULL,
    indicator_date DATE NOT NULL,
    value FLOAT NULL
);

CREATE TABLE model_predictions (
    id INT IDENTITY(1,1) PRIMARY KEY,
    record_type VARCHAR(50) NOT NULL,
    record_key VARCHAR(100) NOT NULL,
    prediction_label VARCHAR(50) NULL,
    prediction_score FLOAT NULL,
    created_at DATETIME DEFAULT GETDATE()
);
go