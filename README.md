# word2vec

Project is implemented in luaJIT scripting language. Torch is used to implement Neural Net in project.

main.lua is the main file of the project. Input is given in form text file where each word is seperated by newline only. Specify input file in train.lua file.

Command to run:
th main.lua

Output will be text file containing words and their vectors seperated by space.
Add first line numberofwords and vector size seperated by space in the file.
To compare accuracy with google code, output file should be in binary. 
To do this use 'convertvec.c' file.
command: convertvec.out txt2bin textfilename binaryfilename

To check the accuracy use 'compute-accuracy.c' file.
