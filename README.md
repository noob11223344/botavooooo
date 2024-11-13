# ashok
1.Perform Creation, indexing, slicing, concatenation 
and repetition operations on python built-in data 
types: Strings, List, Tuples, Dictionary, Set
# 1. Creation of Data Types

# String
str1 = "Hello"
str2 = "World"

# List
list1 = [1, 2, 3, 4]
list2 = ['a', 'b', 'c']

# Tuple
tuple1 = (10, 20, 30)
tuple2 = ('x', 'y', 'z')

# Dictionary
dict1 = {1: 'one', 2: 'two'}
dict2 = {3: 'three', 4: 'four'}

# Set
set1 = {1, 2, 3}
set2 = {3, 4, 5}


# 2. Indexing and Slicing

# String
print("Indexing in String:", str1[1])  # Accessing the 2nd character (index 1)
print("Slicing in String:", str1[1:4])  # Slicing the string from index 1 to 3 (exclusive)

# List
print("Indexing in List:", list1[2])  # Accessing the 3rd element (index 2)
print("Slicing in List:", list1[1:3])  # Slicing the list from index 1 to 2 (exclusive)

# Tuple
print("Indexing in Tuple:", tuple1[0])  # Accessing the 1st element (index 0)
print("Slicing in Tuple:", tuple1[1:])  # Slicing the tuple from index 1 to the end

# Dictionary (indexing by key)
print("Accessing Dictionary by Key:", dict1[1])  # Accessing value associated with key 1

# Set (No indexing or slicing, but can be looped)
for item in set1:
    print("Item in Set:", item)


# 3. Concatenation

# String
concat_str = str1 + " " + str2
print("Concatenated String:", concat_str)

# List
concat_list = list1 + list2
print("Concatenated List:", concat_list)

# Tuple
concat_tuple = tuple1 + tuple2
print("Concatenated Tuple:", concat_tuple)

# Dictionary
dict1.update(dict2)  # Merging dict2 into dict1
# Alternatively, in Python 3.9+, you can use dict3 = dict1 | dict2 for merging
print("Merged Dictionary:", dict1)

# Set
concat_set = set1 | set2  # Union of two sets
print("Union of Sets:", concat_set)


# 4. Repetition

# String
repeat_str = str1 * 3  # Repeats the string 3 times
print("Repeated String:", repeat_str)

# List
repeat_list = list1 * 2  # Repeats the list 2 times
print("Repeated List:", repeat_list)

# Tuple
repeat_tuple = tuple1 * 2  # Repeats the tuple 2 times
print("Repeated Tuple:", repeat_tuple)
_____________________________________________________________________________
# 2.List of tuples with student names and their marks
2.Apply Python built-in data types: Strings, List,     
Tuples, Dictionary, Set and their methods to solve any 
given problem 
students = [ 
    ("Alice", 85), 
    ("Bob", 67), 
    ("Charlie", 92), 
    ("David", 76), 
    ("Alice", 85),  # Duplicate entry 
    ("Eva", 88) 
] 

# 1. Dictionary Creation (Student names as keys, marks as values)
# If a name occurs more than once, the last occurrence is stored in the dictionary
student_dict = {name: marks for name, marks in students}

# 2. Find the student with the highest marks
highest_student = max(student_dict, key=student_dict.get)
print(f"Student with the highest marks: {highest_student}, Marks: {student_dict[highest_student]}")

# 3. Display all students with marks above 75
students_above_75 = [name for name, marks in student_dict.items() if marks > 75]
print(f"Students with marks above 75: {students_above_75}")

# 4. Remove duplicate names (if any) using a set
unique_students = set([name for name, marks in students])
print(f"Unique student names: {unique_students}")

# 5. Concatenate all student names into a single string separated by commas
student_names = ', '.join(unique_students)
print(f"Concatenated student names: {student_names}")

# 6. Sort the students by their marks in descending order
sorted_students = sorted(students, key=lambda x: x[1], reverse=True)
print("Students sorted by marks (descending):")
for name, marks in sorted_students:
    print(f"{name}: {marks}")
    _________________________________________________________________________________________
    3.Handle numerical operations using math and random 
number functions
import math
import random

# Taking input from the user and converting it to a float for mathematical operations
x = float(input("Enter a number: "))

print("\nNumerical Operations using math:")
print("Square root:", math.sqrt(x))           # Square root of x
print("e^2:", math.exp(2))                    # e^2
print("Natural log of 10:", math.log(10))     # Natural log of 10
print("Sin(π/2):", math.sin(math.pi / 2))     # Sin of π/2, which is 1.0
print("Factorial of 5:", math.factorial(5))   # Factorial of 5, which is 120
print("Floor of 3.9:", math.floor(3.9))       # Floor value of 3.9, which is 3
print("Ceil of 3.1:", math.ceil(3.1))         # Ceil value of 3.1, which is 4

print("\nNumerical Operations using random number functions:")
print("Random float between 0 and 1:", random.random())  # Random float between 0 and 1
print("Random integer between 1 and 100:", random.randint(1, 100))  # Random integer between 1 and 100
print("Random float between 1 and 10:", random.uniform(1, 10))      # Random float between 1 and 10

items = ['apple', 'banana', 'cherry']
print("Randomly selected item from the list:", random.choice(items))  # Randomly select an item from the list
random.shuffle(items)  # Shuffle the list in place
print("Shuffled list:", items)
____________________________________________________________________________________________________________
4.Create User-Defined Functions with Different Types 
of Function Arguments
# Function with positional arguments
def positional_args(a, b):
    print("Positional Arguments:")
    print("a =", a)
    print("b =", b)

# Function with default arguments
def default_args(a, b=10):
    print("\nDefault Arguments:")
    print("a =", a)
    print("b =", b)

# Function with keyword arguments
def keyword_args(a, b):
    print("\nKeyword Arguments:")
    print("a =", a)
    print("b =", b)

# Function with variable-length arguments
def variable_length_args(*args, **kwargs):
    print("\nVariable-length Arguments:")
    print("Positional (args):", args)
    print("Keyword (kwargs):", kwargs)

# Main program to call functions with different types of arguments
if __name__ == "__main__":
    print("Calling Positional Arguments Function:")
    positional_args(5, 15)

    print("\nCalling Default Arguments Function:")
    default_args(7)           # Uses default value for b
    default_args(7, 20)      # Overrides default value for b

    print("\nCalling Keyword Arguments Function:")
    keyword_args(a=25, b=35)
    keyword_args(b=50, a=40) # Order can be changed with keyword arguments

    print("\nCalling Variable-length Arguments Function:")
    variable_length_args(1, 2, 3, name="Alice", age=30)
    ________________________________________________________________________
    5.Create NumPy arrays from Python Data Structures, 
Intrinsic NumPy objects and Random Functions 
import numpy as np
# 1. Create NumPy arrays from Python Data Structures
python_list = [1, 2, 3, 4, 5]
numpy_array_from_list = np.array(python_list)

python_tuple = (10, 20, 30, 40, 50)
numpy_array_from_tuple = np.array(python_tuple)

nested_list = [[1, 2, 3], [4, 5, 6]]
numpy_array_from_nested_list = np.array(nested_list)

# 2. Create NumPy arrays from intrinsic NumPy objects
numpy_array_arange = np.arange(0, 10, 2)  # Creates an array of evenly spaced values within a given range
numpy_array_linspace = np.linspace(0, 1, 5)  # Creates an array of 5 evenly spaced values between 0 and 1
numpy_array_ones = np.ones((3, 3))  # Creates a 3x3 array filled with ones
numpy_array_zeros = np.zeros((2, 4))  # Creates a 2x4 array filled with zeros
numpy_identity_matrix = np.eye(3)  # Creates a 3x3 identity matrix

# 3. Create NumPy arrays using random functions
numpy_random_uniform = np.random.rand(3, 3)  # Creates a 3x3 array with random values from a uniform distribution (0 to 1)
numpy_random_integers = np.random.randint(0, 10, size=(2, 3))  # Creates a 2x3 array with random integers between 0 and 9
numpy_random_normal = np.random.randn(4, 4)  # Creates a 4x4 array with random values from a normal distribution
numpy_random_sample = np.random.random_sample((2, 3))  # Creates a 2x3 array with random float values between 0 and 1

# Print results
print("NumPy array from list:", numpy_array_from_list)
print("NumPy array from tuple:", numpy_array_from_tuple)
print("NumPy array from nested list:\n", numpy_array_from_nested_list)
print("Array using arange:", numpy_array_arange)
print("Array using linspace:", numpy_array_linspace)
print("Array of ones:\n", numpy_array_ones)
print("Array of zeros:\n", numpy_array_zeros)
print("Identity matrix:\n", numpy_identity_matrix)
print("Random array (uniform distribution):\n", numpy_random_uniform)
print("Random array of integers:\n", numpy_random_integers)
print("Random array (normal distribution):\n", numpy_random_normal)
print("Random array (random_sample):\n", numpy_random_sample)
________________________________________________________________________________________
6.Manipulation of NumPy arrays- Indexing, Slicing, 
Reshaping, Joining and Splitting.
import numpy as np

# Creating a sample NumPy array
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 1. Indexing8r
print("Element at row 2, column 3:", arr[1, 2])  # Indexing to get element at 2nd row, 3rd column (0-indexed)
print("First row:", arr[0])                      # Getting the entire first row
print("First column:", arr[:, 0])                # Getting the entire first column

# 2. Slicing
print("\nArray slice (first two rows and two columns):\n", arr[0:2, 0:2])  # Slicing to get a subarray (top left 2x2)
print("Last two elements of the second row:", arr[1, 1:])                  # Slicing the second row

# 3. Reshaping
reshaped_arr = arr.reshape(1, 9)  # Reshaping 3x3 array into a 1x9 array
print("\nReshaped array (3x3 to 1x9):\n", reshaped_arr)

reshaped_back = reshaped_arr.reshape(3, 3)  # Reshaping back to 3x3
print("\nReshaped back to 3x3:\n", reshaped_back)

# 4. Joining (Concatenation)
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

# Joining along axis 0 (vertical stacking)
joined_vertically = np.vstack((arr1, arr2))
print("\nJoined vertically:\n", joined_vertically)

# Joining along axis 1 (horizontal stacking)
joined_horizontally = np.hstack((arr1, arr2))
print("\nJoined horizontally:\n", joined_horizontally)

# 5. Splitting
arr_split = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]])

# Splitting into two equal arrays along rows (axis 0)
split_arr = np.split(arr_split, 2, axis=0)
print("\nSplit along rows (axis 0):")
for part in split_arr:
    print(part)

# Splitting into three arrays along columns (axis 1)
split_arr_columns = np.split(arr_split, 3, axis=1)
print("\nSplit along columns (axis 1):")
for part in split_arr_columns:
    print(part)
______________________________________________________________________
7.Load an image file and do crop and flip  
Operation using NumPy Indexing

from PIL import Image
import numpy as np

# Load the image
image = Image.open("D://31165195ad7305447dbbe279951736b4.jpg")

# Convert the image to a NumPy array
image_array = np.array(image)

# Crop the image (example: crop 100 pixels from each side)
# Syntax: array[start_row:end_row, start_column:end_column]
cropped_image = image_array[100:-100, 100:-100]

# Flip the image horizontally (axis=1 for left-right flip)
flipped_image = np.flip(cropped_image, axis=1)

# Convert the NumPy array back to an image
result_image = Image.fromarray(flipped_image)

# Save or display the resulting image
result_image.show()  # To display the image
result_image.save("output_image.jpg")  # To save the image
_______________________________________________________________________________________________
8.Creating Pandas Series and DataFrame  
from Various Inputs

import pandas as pd 
import numpy as np 
# Series from a list 
list_data = [10, 20, 30, 40, 50] 
series_from_list = pd.Series(list_data) 
print("Series from List:\n", series_from_list) 
#Series from a dictionary 
dict_data = {'a': 100, 'b': 200, 'c': 300} 
series_from_dict = pd.Series(dict_data) 
print("\nSeries from Dictionary:\n", series_from_dict) 
#Series from a NumPy array 
numpy_array = np.array([1, 2, 3, 4, 5]) 
series_from_array = pd.Series(numpy_array) 
print("\nSeries from NumPy Array:\n", series_from_array) 
#Create a DataFrame from various inputs 
# DataFrame from a dictionary 
data_dict = { 
'Name': ['Alice', 'Bob', 'Charlie'], 
'Age': [24, 27, 22], 
'City': ['New York', 'Los Angeles', 'Chicago'] 
} 
df_from_dict = pd.DataFrame(data_dict) 
print("\nDataFrame from Dictionary:\n", df_from_dict) 
# DataFrame from a list of lists 
data_list = [ 
['John', 28, 'London'], 
['Anna', 24, 'Paris'], 
['Mike', 32, 'Berlin'] 
] 
df_from_list = pd.DataFrame(data_list, columns=['Name', 'Age', 'City']) 
print("\nDataFrame from List of Lists:\n", df_from_list) 
# DataFrame from a NumPy array 
numpy_array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) 
df_from_numpy = pd.DataFrame(numpy_array_2d, columns=['A', 'B', 'C']) 
print("\nDataFrame from NumPy Array:\n", df_from_numpy) 
_________________________________________________________________________________
9.Import any CSV file to Pandas Data Frame and 
perform the given operations 

import pandas as pd 

# Load the CSV file into a DataFrame
# Assuming 'sample.csv' is the file to be loaded
df = pd.read_csv('D:\python\example.csv') 

# (a) Visualize the first and last 10 records
print("First 10 records:\n", df.head(10)) 
print("\nLast 10 records:\n", df.tail(10)) 

# (b) Get the shape, index, and column details
print("\nShape of the DataFrame:", df.shape) 
print("\nIndex of the DataFrame:", df.index) 
print("\nColumns of the DataFrame:", df.columns) 

# (c) Select/Delete records based on conditions
# Selecting records where 'Age' column is greater than 25
selected_records = df[df['Age'] > 25] 
print("\nRecords where Age > 25:\n", selected_records) 

# Deleting the 'City' column
df_deleted_column = df.drop(columns=['City']) 
print("\nDataFrame after deleting 'City' column:\n", df_deleted_column) 

# (d) Perform ranking and sorting operations
# Ranking based on the 'Age' column
df['Age Rank'] = df['Age'].rank(ascending=False) 
print("\nDataFrame with 'Age' ranked:\n", df[['Name', 'Age', 'Age Rank']]) 

# Sorting by 'Age' column in ascending order
sorted_df = df.sort_values(by='Age', ascending=True) 
print("\nDataFrame sorted by 'Age':\n", sorted_df) 

# (e) Perform statistical operations on a given column
print("\nMean of 'Age':", df['Age'].mean()) 
print("Median of 'Salary':", df['Salary'].median()) 
print("Standard Deviation of 'Salary':", df['Salary'].std()) 

# (f) Find the count and uniqueness of the given categorical values
print("\nCount of unique values in 'City':\n", df['City'].value_counts()) 
print("\nUnique values in 'City':", df['City'].unique()) 

# (g) Rename single/multiple columns
df_renamed = df.rename(columns={'Name': 'Employee Name', 'Age': 'Employee Age'}) 
print("\nDataFrame after renaming columns:\n", df_renamed) 
______________________________________________________________________________________________
10.Import any CSV file to Pandas Data Frame and 
perform the given operations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
# Replace 'sample.csv' with the actual path to your file
df = pd.read_csv('D:\python\example.csv')

# (a) Handle Missing Data
# Detect missing values
print("\nMissing Data:\n", df.isnull().sum())

# Drop rows with missing values
df_cleaned = df.dropna()
print("\nDataFrame after dropping missing values:\n", df_cleaned)

# Fill missing values with the median for numeric columns
df_filled = df.fillna(df.median(numeric_only=True))
print("\nDataFrame after filling missing values with median:\n", df_filled)

# (b) Transform Data Using apply() and map()
# Applying a custom function to square the 'Age' column
df['Age Squared'] = df['Age'].apply(lambda x: x ** 2)
print("\nDataFrame after applying function (Age Squared):\n", df[['Name', 'Age', 'Age Squared']])

# Using map() to substitute values in the 'City' column
city_mapping = {'New York': 'NY', 'Los Angeles': 'LA', 'Chicago': 'CHI'}
df['City Short'] = df['City'].map(city_mapping)
print("\nDataFrame after using map() to transform 'City':\n", df[['City', 'City Short']])

# (c) Detect and Filter Outliers
# Detect outliers using the IQR (Interquartile Range) method
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Age'] < (Q1 - 1.5 * IQR)) | (df['Age'] > (Q3 + 1.5 * IQR))]
print("\nDetected outliers in 'Age':\n", outliers)

# Filter out the outliers
df_filtered = df[~((df['Age'] < (Q1 - 1.5 * IQR)) | (df['Age'] > (Q3 + 1.5 * IQR)))]
print("\nDataFrame after filtering out outliers:\n", df_filtered)

# (d) Perform Vectorized String Operations on Pandas Series
# Convert 'Name' column to lowercase
df['Name Lower'] = df['Name'].str.lower()
print("\nDataFrame after converting 'Name' to lowercase:\n", df[['Name', 'Name Lower']])

# Check if 'Name' contains a specific substring (e.g., 'a')
df['Contains A'] = df['Name'].str.contains('a', case=False)
print("\nDataFrame after checking if 'Name' contains the letter 'a':\n", df[['Name', 'Contains A']])

# (e) Visualize Data Using Different Plots
plt.figure(figsize=(10, 8))

# Line Plot: Age vs Salary
plt.subplot(2, 2, 1)
plt.plot(df['Age'], df['Salary'], marker='o')
plt.title('Line Plot: Age vs Salary')
plt.xlabel('Age')
plt.ylabel('Salary')

# Bar Plot: City Count
plt.subplot(2, 2, 2)
df['City'].value_counts().plot(kind='bar', color='orange')
plt.title('Bar Plot: City Count')
plt.xlabel('City')
plt.ylabel('Count')

# Histogram: Age Distribution
plt.subplot(2, 2, 3)
df['Age'].plot(kind='hist', bins=10, color='lightblue', edgecolor='black')
plt.title('Histogram: Age Distribution')
plt.xlabel('Age')

# Density Plot: Salary Distribution
plt.subplot(2, 2, 4)
df['Salary'].plot(kind='density', color='green')
plt.title('Density Plot: Salary Distribution')
plt.xlabel('Salary')

# Display all plots
plt.tight_layout()
plt.show()
________________________________________
