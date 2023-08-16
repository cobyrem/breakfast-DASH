CREATE TABLE  PAYMENTS
	(	
	ORDER_ID INTEGER NOT NULL PRIMARY KEY, 
	METHOD TEXT, 
	CARD_TYPE TEXT, 
	LAST_FOUR_CARD_NUMBER TEXT, 
	ZIP TEXT, 
	CARDHOLDER TEXT
	);

-- FK incase error in ETL.
CREATE TABLE  ITEMS
   (
	ORDER_ID INTEGER,
	NAME TEXT, 
	PRICE REAL,
	FOREIGN KEY(ORDER_ID) REFERENCES PAYMENTS(ORDER_ID)
	 );

-- In the future I would consider simply converting datetime to YYYY-MM-DD HH:MM:SS format.
CREATE TABLE  CHARGES
   (	
	ORDER_ID INTEGER NOT NULL PRIMARY KEY, 
	FULL_DATE TEXT,
	FULL_TIME TEXT,
	MONTH INTEGER,
	DAY INTEGER,
	DAY_OF_WEEK TEXT, 
	YEAR INTEGER,
	HOUR INTEGER,
	MINUTE INTEGER,
	SUBTOTAL REAL, 
	TAXES REAL, 
	TOTAL REAL
	); -- 



