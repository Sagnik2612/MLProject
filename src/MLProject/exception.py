#We want to make a customexception handler file

import sys
from src.MLProject.logger import logging


def er_mes_detail(error,error_detalis):

    _,_,ex_tb=error_detalis.exc_info()

    #sys.exc_info()
    #This function returns the old-style representation of the handled
    #  exception. If an exception e is currently handled 
    # (so exception() would return e), exc_info() returns the
    #  tuple (type(e), e, e.__traceback__). That is, a tuple 
    # containing the type of the exception (a subclass of BaseException),
    #  the exception itself, and a traceback object which typically 
    # encapsulates the call stack at the point where the exception last 
    # occurred.
    #If no exception is being handled anywhere on the stack, this function 
    # return a tuple containing three None values.
    #We only need the last entry of the tuple from the sys.exc_info()
    #function hence we do it like-->
    #_,_,exc_tb=error_details.exc_info()
    #Here exc_tb is e.__traceback__
    #Traceback is a python module that provides a standard interface to extract, 
    # format and print stack traces of a python program. When it prints the stack trace
    #  it exactly mimics the behaviour of a python interpreter. Useful when you want to print
    #  the stack trace at any step. They are usually seen when an exception occurs. 
    # Since a traceback gives all the
    # information regarding the exception it becomes easier to track one and fix it.

    filename=ex_tb.tb_frame.f_code.co_filename #Gives the filename of file 
    #causing the error
    #This line extracts the filename where the error occurred from 
    # the traceback information.
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(filename,ex_tb.tb_lineno,str(error))
    #This line formats an error message using the extracted error details, 
    # including the filename, line number, and error message string.


    return error_message



#We make a class on Custom Exception 
# that will inherit the exception

#Why Define Custom Exceptions?
#Custom exceptions are useful in the following scenarios:

#Clarity: They provide clear, specific error messages that 
# are relevant to your application.
#Granularity: They allow for more fine-grained error handling,
#  making it easier to pinpoint and address specific issues.
#Reusability: They can be reused across different parts of
#  your application or even in different projects.


#Format of Custom Exception-->

#class MyCustomError(Exception):
#    """Exception raised for custom error scenarios.

#    Attributes:
#        message -- explanation of the error
#    """

#    def __init__(self, message):
#        self.message = message
#        super().__init__(self.message)

#Example-
#class FileProcessingError(Exception):
#    def __init__(self, message, filename, lineno):
#        super().__init__(message)
#        self.filename = filename
#        self.lineno = lineno#

#    def __str__(self):
#        return f"{self.message} in {self.filename} at line {self.lineno}"


#try:
#    raise FileProcessingError("Syntax error", "example.txt", 13)
#except FileProcessingError as e:
#    print(f"Caught an error: {e}")

#Role of super()-->

#In object-oriented programming, inheritance 
# plays a crucial role in creating
# a hierarchy of classes. Python, being an object-oriented language,
# provides a built-in function called super() that allows 
# a child class to refer to its parent class. 
# When it comes to initializing instances of classes, 
# the __init__() method is often used. 
# Combining super() with __init__() can be particularly 
# useful when you want to extend the behavior 
# of the parent class's constructor while maintaining its functionality.
#The super() function in Python is used to refer to the parent class.
#When used in conjunction with the __init__() method, it allows
#  the child class 
# to invoke the constructor of its parent class (here Exception class)
#This is especially useful when you want to add 
# functionality to the child class's constructor without 
# completely overriding the parent class's constructor.

#Format for super()-->

#class ChildClass(ParentClass):
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)
        # Child class specific initialization code
#Here, super().__init__(*args, **kwargs) calls the __init__() 
# method of the parent class with the arguments and
#  keyword arguments passed to the child class's constructor.

#Example of super()-->
#class Animal:
#    def __init__(self, name):
#        self.name = name

#class Dog(Animal):
#    def __init__(self, name, breed):
#        super().__init__(name)
#        self.breed = breed

# Creating an instance of the Dog class
#my_dog = Dog("Buddy";, "Labrador")
#print(f"My dog's name is {my_dog.name} and it's a {my_dog.breed}"")
#Output-My dog's name is Buddy and it's a Labrador.

#Exception class-
#The base class for all non-exit exceptions.
#try:
#    raise Exception("This is a generic exception")
#except Exception as e:
#    print(e)

#We made the Error class and all it's methods because
#we want a detailed error- which will have the line no., error details
#filename of the file that is causing the error

class CustomException(Exception):
    #Inherits the Exception
    #Now we use an initializer
    def __init__(self,error_message,error_detalis:sys):
        #In Python, the __init__ method is a special function that
        #  initializes a new object when it is created. 
        #It acts like a contructor but it is NOT a constructor
        # The self parameter in the __init__ method
        #  refers to the instance of the object being created. 
        # Most object-oriented languages pass self as a hidden parameter
        #  to the methods defined on an object; Python does not.
        #  You have to declare it explicitly. 
        #This self represents the object of the class itself.
        #  Like in any other method of a class, in case of __init__ also ‘self’ 
        # is used 
        # as a dummy object variable 
        # for assigning values to the data members of an object. 
        #self acts like a placeholder that "becomes" the immediate object that is
        #created outside the class.The object gets passed to the class as 
        #"itself" or "self"."self" is also like a pointer to "this current object"
        #sys will help us track the error details
        #Error details is assigned as sys type

        super().__init__(error_message) 
        #inheriting parent Exception class's
        #constructor __init__() method by passing error message
        self.error_message=er_mes_detail(error_message,error_detalis)           
        #we will assign self.error_message
        #to a function that will bring /throw the error message
        #by taking input the error message,error details


    def __str__(self):
        #__str__() is an inbuilt method of a python class
        #This method decides how thw objects of that class 
        #can be fetched in a humanly readable string format
        #Directly printing the object would give the type of the object and a
        #memory location.
        #But __str__() prints it in a humanly understandable format
        #Example-
        #class Student:
        #      def __init__(self,name,score):
        #          self.name=name
        #          self.score=score
        #      def __str__(self):
        #          return f"{self.name},{self.score}"
        #
        #Mary=Student("Mary","98")
        #print(Mary)

        #output--> Mary,98
        #_str__()in someways is the same as the type
        # casting function (str())

        return self.error_message #We return the error message


