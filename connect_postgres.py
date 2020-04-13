#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:58:39 2019

This program takes a sql query as an argument, connects to the datawh
database in recherche-data1. Once it executes the query, it returns the 
relevant dataframe.

@author: rdas

"""
import psycopg2
import pandas as pd
import os
import pwd

def connectDB_returnDF(query):   
 
 username = pwd.getpwuid(os.getuid())[ 0 ]
 homedirectory = pwd.getpwuid(os.getuid())[ 5 ]
 sslkeyPath = homedirectory + '/.postgresql/postgresql-' + username + '.key'
 sslcertPath = homedirectory + '/.postgresql/postgresql-' + username + '.crt'

 # Create the connection, the files postgresql-username.key and  postgresql-username.crt should be present for the 
 # connection to succeed.   

 conn = psycopg2.connect(host="recherche-data1", port = 5432, dbname="datawh",  user= username, sslkey=sslkeyPath ,sslcert=sslcertPath)
 cur = conn.cursor()

 # Execute the query and print the result without assigning to a 
 # dataframe.
 
 #cur.execute(query)
 #query_results = cur.fetchall()


 # Assigning the query result to a dataframe.

 rows = pd.read_sql_query(query, conn)

 # Close the cursor and connection. 

 cur.close()
 conn.close()
 
 return rows
