#!/usr/bin/env python
# coding: utf-8

# # Book Recommendation System

# The goal is to build a very simple recommendation engine based on the concept of 
# Amazon's Customers Who Bought This Item Also Bought (CWBTIAB).

# In[311]:


# Importing required libraries
from __future__ import print_function
import sys
import math
import pyspark
from pyspark.sql import SparkSession
# for sorting
from operator import itemgetter
import pyspark
from pyspark import SparkContext


# Here we are considering book name as the key and user as the value

# In[312]:


#------------------------------------------------
# Define a function to emit the value as key- value pair, here we are using bookname,userID
def book_data(book_record):
    tokens = book_record.split(':')
    userID = tokens[0]
    bookname = tokens[1]
    print(bookname)
    return (bookname, userID)
#end-def
#------------------------------------------------


# In[313]:


# STEP-1: create an instance of a SparkSession object
spark = SparkSession.builder.getOrCreate()
#to reduce the verbosity of Spark's runtime output
#spark.sparkContext.setLogLevel("WARN")

# STEP-2: Read input parameters. If the usage is incorrect, exit the program.
if len(sys.argv) != 2:
    print("Usage is quiz3.py <inputfile>")
    exit(1)
input_path = sys.argv[1]



# In[314]:


# Creating a RDD called data
data = spark.sparkContext.textFile(input_path)


# In[315]:


# Find the number of records in the input file
bookrecords = data.map(book_data)
## print("The number of records in the input file: ", bookrecords.count())


# In[316]:


# Remove the duplicate records (if any) present in the file
bookrecords = data.map(book_data)
bookrecords=bookrecords.distinct()
## print("The number of distinct records in the file: ", bookrecords.count())


# In[317]:


# Find the number of users who have read the book
bookrecords.collect()
count_of_users = bookrecords.map(lambda x : x[1]).distinct().collect()
count_of_users=len(count_of_users)
## print("The unique number of users in the file: ",count_of_users)


# In[318]:


# Sort the book based on the key value
bookrecords=bookrecords.sortByKey(ascending=True)


# In[319]:


# Group the user based on books(key) and the users who read it(value).
bookrecords=bookrecords.groupByKey().map(lambda x: (x[0],list(x[1])))
bookrecords.collect()


# In[320]:


# Print the total count of books
## print("The number of unique books in the file: ",bookrecords.count())


# # Cartesian Product

# In[321]:


# Find the Cartesian products for all the pairs. This is needed so we
# can combine and correlate each book and its corresponding user list.
book_cartesian=bookrecords.cartesian(bookrecords).filter(lambda x : x[0] < x[1] )


# In[322]:


book_cartesian.collect()


# In[323]:


# Define a function to compute the phi-correlation. We do this by looking at each record and
# using the formula defined in https://measuringu.com/affinity-analysis/
def phi_corr(book_corr):
    # Compute the total combination of users based on each book.
    yes_yes=len(set(book_corr[0][1]).intersection(set(book_corr[1][1])))
    yes_no=len(set(book_corr[0][1])-(set(book_corr[1][1])))
    no_yes=len(set(book_corr[1][1])-(set(book_corr[0][1])))
    no_no = count_of_users -(yes_yes+yes_no+no_yes)
    
    # we not compute the correlation value and return both books along with the correlation.
    denominator=math.sqrt((yes_yes+yes_no) * (no_yes+no_no) * (yes_yes+no_yes) * (yes_no+no_no))
    phi_corr=((yes_yes * no_no) - (yes_no * no_yes))/denominator
    phi_corr=round(phi_corr,2)
    return(book_corr[0][0],book_corr[1][0],phi_corr)
    
# Calling the function
phi_corr_matrix=book_cartesian.map(phi_corr)
    


# In[324]:


# Print elements from the phi-correlation matrix
print(phi_corr_matrix.collect())


# In[325]:


# We have two books in each record, in order to determine the relation between book 2 and
# book 1, we duplicate the list by adding book 2 first followed by book 1. After this, both
# the lists are combined together so that the recommender system can suggest books for 
# either case.

corr_matrix_first_half = phi_corr_matrix.map(lambda x: (x[0], (x[1], x[2])))
corr_matrix_second_half = phi_corr_matrix.map(lambda x: (x[1], (x[0], x[2])))


# In[326]:


# Print the 4 key-value pair of the matrix
corr_matrix_first_half.take(4)


# In[327]:


# Print the 4 key-value pair of the transposed matrix
corr_matrix_second_half.take(4)


# In[328]:


# Combining the the phi correlation matrix with the transposed matrix
corr_matrix = corr_matrix_first_half.union(corr_matrix_second_half)


# # Sparse Matrix

# Sparse similarity matrix we find the similarities between every pair of books. 
# 

# In[329]:


# Group a book with all its pair along with its phi-correlation
sparse_matrix= corr_matrix.groupByKey().map(lambda x: (x[0], list(x[1])))


# In[330]:


# Let's sort the pair of books based on the descending order of phi-correlation. Since the 
# rounded figure is taken, there can be two or more books with equal co-relation, but we choose
# one out of it based on the order it sees.
customer_rdd=sparse_matrix.map(lambda x: (x[0], sorted(x[1], key=lambda x: x[1], reverse=True)))
customer_rdd.collect()


# # Recommendation

# In[331]:


#Finding the top 2 recommendation books for a type of book
def get_top_two_books(book_corr):
    list_books = book_corr[1][:2]
    
    book_1 = list_books[0][0]
    book_2 = ""
    if len(list_books) > 1:
        book_2 = list_books[1][0]
    return(book_corr[0], (book_1, book_2))
    
    
customer_rdd_top_2=customer_rdd.map(get_top_two_books)
customer_rdd_top_2.collect()


# In[332]:

# Helper function to just display the book list
def print_book_recommendation(record):
    return(record[0]+ " : " + record[1][0] + ", " + record[1][1])

# Helper function to print the customer details
def print_recommendation_text(record):
    return("Customers Who Bought " + record[0] + " also Bought: " + record[1][0] + " and " + record[1][1])


# In[333]:

# Let's print just the book recommendations.
print(customer_rdd_top_2.map(print_book_recommendation).collect())

# Let's print customer details along with the recommendation.
print(customer_rdd_top_2.map(print_recommendation_text).collect())


# Reference : https://stackoverflow.com/questions/26557873/spark-produce-rddx-x-of-all-possible-combinations-from-rddx
# https://stackoverflow.com/questions/34618029/how-to-sort-rdd-of-nested-list-structure-by-value-in-spark

# In[ ]:




