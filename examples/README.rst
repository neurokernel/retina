Running the examples
--------------------

Each example runs with the default parameters by executing the shell script.

Shell script copies or creates some necessary files the first time
it is executed but otherwise it is a convenient wrapper 
for the respective python script.

Configuration of examples
-------------------------

Simulations can be configured by a configuration file with the fields shown in
specification file. This file is the specification of a valid configuration file
and in addition documents parameters and provides default values in case a field
is missing.

Scripts generate an empty configuration file named <example>_default.cfg if
it does not exist which is a valid configuration that uses all default values.
User can edit this file or use a different one but in the latter case should
provide the name to the script.

Try
$ ./run_<example>_demo.sh -h
or
$ python <example>_demo.py -h
for all the available options

