# C++ Quirks

**`main.cpp` Usage**

```bash
git clone https://github.com/zhermin/cardscanner.git
cd cardscanner/cardscanner-custom
make
./main.o card5 1; echo $?  # runs main.o on card5.pgm image and print stdout
>> 3  # number of corners found in card5.pgm
```

The output grayscale `.pgm` images can be found in `cardscanner/cardscanner-custom/assets/outputs`.

## Binary Mode

Input: `ifstream var(filename, ios::binary)`

Output: `ofstream var(filename, ios::binary)`

From the default cpp library `<fstream>`, these two functions reads and writes a file in binary format (byte stream). In both cases, the parameters following the variable name are the filename and `ios::binary` to indicate binary mode.

## Double :: Variables

Example, `::sig = stoi(argv[3])`, the `::` operator is the "score-resolution" operator which resolves the scope of a variable. Most of the time, the double colon will be preceded with the name of a library, such as in `cv::Mat img`, we are resolving the variable to the OpenCV library.

In the case of a naked double colon, it resolves to the global scope or top level scope of the project. This is essentially a global variable across all files and if a new variable with the same name is defined in a local scope, they will point to different variables under the hood.

## Double ** Variables

From `main.cpp`, the variable `pic` from `double **pic = new double*[height]` is used to deference twice to point to the first memory space of an array.

```cpp
// The following are equivalent
int main(int argc, char* argv[])
int main(int argc, char** argv)
```

## Operators `<<` and `>>`

These operators push a stream of bytes into an underlying buffer. In the most common case, with the help of the default cpp library `<iostream>`, we get access to both `cout` and `cin` to write and read outputs and inputs to allow things to be printed.

```cpp
#include <iostream>
int main() { cout << "Hello World" << endl; }
```

We can also push data to a bunch of variables in one line, such as the header information from a `.pgm` file, which contains 4 ASCII text in its header, allowing us to store that data directly to variables.

```cpp
infile >> ::type >> ::width >> ::height >> ::intensity;
```

## Command Line Arguments

To pass command line arguments, we typically define main() with two arguments: first argument is the number of command line arguments and second is list of command-line arguments.

```cpp
int main(int argc, char *argv[]) { /* ... */ }
// or
int main(int argc, char **argv) { /* ... */ }
```

- `argv(ARGument Vector)` is an array of character pointers listing all the arguments
- If `argc` is greater than zero, the array elements from `argv[0]` to `argv[argc-1]` will contain pointers to **strings**
- `argv[0]` is the name of the program while the rest are the command line arguments until `argv[argc-1]`

## `stoi()` Function

S-TO-I stands for String TO Integer conversion. This can be used to easily parse the character arrays `char *`, aka strings, from command line arguments to integers, such as in the case of `int lo = stoi(argv[3])`.
