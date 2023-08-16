import sqlite3
import json
from sqlite3 import Error
import datetime
import calendar
import os

def weekday_from_date(day, month, year):
    return calendar.day_name[datetime.date(day=day, month=month, year=year).weekday()]

#https://www.sqlitetutorial.net/sqlite-python/creating-database/
def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    cursor = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()

def read_json_data(file, connection, cursor):
    # file gets closed when out of scope.
    with open(file,'r') as f:
        data = json.load(f)
        orders = data["orders"]

        return orders

def parse_orders(orders, connection, cursor):
    order_num = 0

    for order in orders:
        order_num += 1
        
        items = order["items"]
        parse_items(items, order_num, connection, cursor)
        
        charges = order["charges"]
        parse_charges(charges, order_num, connection, cursor)
        
        payment = order["payment"]
        parse_payment(payment, order_num, connection, cursor)

def parse_items(items, order_num, connection, cursor):
    for item in items:
        item_name = item["name"]
        item_price = float(item["price"])

        item_insert_command = """INSERT INTO ITEMS (ORDER_ID, NAME, PRICE) 
        VALUES (?, ?, ?)"""

        fields = (order_num,item_name,item_price)

        cursor.execute(item_insert_command, fields)
        connection.commit()

def parse_charges(charges, order_num, connection, cursor):
    charges_date_time = charges["date"]
    charges_date_time_split = charges_date_time.split(" ") # Split date and time
    
    charges_date = str(charges_date_time_split[0]) # Split date
    charges_date_split = charges_date.split("/")

    charges_month = int(charges_date_split[0].lstrip("0")) # Remove leading zeros as won't store in casted int with leading zero.
    charges_day = int(charges_date_split[1].lstrip("0"))
    charges_year = int("20"+charges_date_split[2]) #Add full year.
    charges_weekday = str(weekday_from_date(day = int(charges_day), month = int(charges_month), year = int(charges_year))) # Find weekday from month, day, year. Can also do this in sql
    
    charges_time = str(charges_date_time_split[1]) # Split time
    charges_time_split = charges_time.split(":")

    charges_hour = int(charges_time_split[0])
    charges_minute = int(charges_time_split[1])

    charges_subtotal = float(charges["subtotal"])
    charges_taxes = float(charges["taxes"])
    charges_total = float(charges ["total"])

    charges_card_insert_command = """INSERT INTO CHARGES (ORDER_ID, FULL_DATE, FULL_TIME, MONTH, DAY, DAY_OF_WEEK, YEAR, HOUR, MINUTE, SUBTOTAL, TAXES, TOTAL) 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

    fields = (order_num, charges_date, charges_time, charges_month, charges_day, charges_weekday, charges_year, charges_hour, charges_minute, charges_subtotal, charges_taxes, charges_total)

    cursor.execute(charges_card_insert_command, fields)
    connection.commit()

def parse_payment(payment, order_num, connection, cursor):
    payment_method = payment["method"]

    payment_insert_command = """INSERT INTO PAYMENTS (ORDER_ID, METHOD, CARD_TYPE, LAST_FOUR_CARD_NUMBER, ZIP, CARDHOLDER) 
        VALUES (?, ?, ?, ?, ?, ?)"""

    if payment_method == "credit_card":
        payment_card_type = str(payment["card_type"])
        payment_last_4_card_number = str(payment["last_4_card_number"])
        payment_zip = str(payment["zip"])
        payment_cardholder = str(payment["cardholder"])

        fields = (order_num,payment_method,payment_card_type,payment_last_4_card_number,payment_zip,payment_cardholder)

    else: #cash payment method
        payment_card_type = None
        payment_last_4_card_number = None
        payment_zip = None
        payment_cardholder = None

        fields = (order_num,payment_method,payment_card_type,payment_last_4_card_number,payment_zip,payment_cardholder)

    cursor.execute(payment_insert_command, fields)
    connection.commit()

if __name__ == '__main__':
    
    #Only doing this to show how it would be manually done. Comment out if needed.
    if os.path.exists("reids.db"):
        os.remove("reids.db")
    else:
        print("The file does not exist")
    # Create basic tables from database
    print("Making reids.db")
    os.system("sqlite3 reids.db < reids.sql")

    #Could probably include more error checking, if connection not made. Short lab, so not going to add error checking.
    connection = sqlite3.connect('reids.db')
    cursor = connection.cursor()

    orders = read_json_data('reids.json', connection, cursor)
    orders = parse_orders(orders, connection, cursor) #Parse order data into items, charges, and payment, and further breaks down json. Also inserts into DB.
