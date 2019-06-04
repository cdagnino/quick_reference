# Logging basics

+ [Basic tutorial](https://www.youtube.com/watch?v=-ARI4Cz-awo)
+ [Advanced tutorial](https://www.youtube.com/watch?v=jxmzY9soFXg)

Four levels to display messages. If you set `info` then all more serious messages will be displayed

+ DEBUG: detailed information
+ INFO: confirmation that things are working as expected
+ WARNING: indication that something unexpected happened
+ ERROR: software hasn't been able to perform some function
+ CRITICAL: serious error, the program itself may be unable to continue running.

Logging appends to the file, doesn't cancel it!

```py
import logging

#This sends to stdout
logging.basicConfig(level=logging.DEBUG)
#This saves to a file
logging.basicConfig(filename='test.log', level=logging.DEBUG)
```

```py
add_result = add(num_1, num_2)
logging.debug(f"Add: {num1} + {num_2} = ")
logging.info(f"Add: {num1} + {num_2} = ")
```

### Changing format of messages

[See the docs](https://docs.python.org/3/library/logging.html#logrecord-attributes) to check what are the things you can include in each message.

```py

logging.basicConfig(filename='test.log', level=logging.DEBUG, 
format='%(asctime)s:%(levelname)s:%message)s'  )
```

### Having multiple files and multiple loggers

Get a new logger for each running file. Delete the BasicConfig, but you need to define a file handler and a formatter. The formatter belongs to the file handler.

```py
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#Handler and its formatter
formatter = logging.Formatter(%(asctime)s:%(levelname)s:%message)s')
file_handler = logging.FileHandler('employee.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

#
# Code that does stuff goes here
#
logger.info("message blabla")
```

In the example above, the logging level is set at the `logger` level, but you can also have it set at the file handler level (I guess that one inherits the `logger` parameters if left unspecified)

```py
file_handler = logging.FileHandler('employee.log')
file_handler.setLevel(logging.ERROR)
```

You could have a file handler to send some types of messages to std_out and one to send to a file.

In this example, I send the same messages to both:

```py
file_handler = logging.FileHandler('sample.log')
stream_handler = logging.StreamHandler()
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
```

### Get nice error traceback

Use `logger.exception` instead of `logger.error`


```py
def divide(x, y):
	try:
		result = x / y
	except: ZeroDivisionError:
	    logger.exception("Tried dividing by zero")
	 else:
	     return result
```



