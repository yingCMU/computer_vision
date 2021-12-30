# Compiling
    $ g++ main.cpp -o main.out

There are many tools available to help you compile, ranging from barebones tools, such as g++ on Unix, to complex build systems that are integrated into IDEs like Visual Studio and Eclipse.

There is a high-level build tool called CMake that is fairly popular and cross-platform. CMake in and of itself, however, does not compile code. CMake results in compilation configurations. It depends on a lower-level build tool called Make to manage compiling from source. And then Make depends on a compiler to do the actual compiling.

## linking
Linking is the process of creating an executable by effectively combining object files. During the linking process, the linker (the thing that does the linking) resolves symbolic references between object files and outputs a self-contained binary with all the machine code needed to execute.

As an aside, linking is not required for all programs. Most operating systems allow dynamic linking, in which symbolic references point to libraries that are not compiled into the resulting binary. With dynamic linking, these references are resolved at runtime. An example of this is a program that depends on a system library. At runtime, the symbolic references of the program resolve to the symbols of the system library.

There are pros and cons of dynamic linking. On the one hand, dynamically linked libraries are not linked within binaries, which keeps the overall file size down. However, if the library changes at any point, your code will automatically link to the new version. Like any changing dependency, difficult to fix and surprising bugs sometimes arise when versions change.

Compiling to Executable with a Compiler
Technically, you only need a compiler to compile C++ source code to a binary. A compiler does the dirty work of writing machine code for a given processor architecture. There are many compilers available. For this course, we picked the open source GNU Compiler Collection, more commonly called G++ or GCC. gcc is a command line tool.

There are two challenges with using gcc alone, both of which relate to the fact that most C++ projects are large. For one thing, you need to pass the paths for all of the project's source header files and cpp files to gcc. This is in addition to any compiler flags or options. You can easily end up with single call to gcc that spans multiple lines on a terminal, which is unruly and error-prone.

Secondly, large projects will usually contain multiple linked binaries, each of which is compiled individually. If you're working in large project and only change one .cpp file, you generally only need to recompile that one binary - the rest of your project does not need to be compiled again! Compiling an entire project can take up to hours for large projects, and as such being intelligent about only compiling binaries with updated source code can save lots of time. GCC in and of itself is not smart enough to recognize what files in your project have changed and which haven't, and as such will recompile binaries needlessly - you'd need to manually change your gcc calls for the same optimizations. Luckily, there are tools that solve both of these problems!

## resources
[c++ on windows](https://arachnoid.com/cpptutor/setup_windows.html)

# Namespae & Basic Types
http://www.cplusplus.com/doc/tutorial/other_data_types/

    using namespace std;
    int main()
    {
        std::cout<<"Hello World";
        cout << "";
        cout << 23;
        int integer = 4543;
        std::cout<<”The value of integer is “<<integer;
        //Output: The value of integer is 4543
        const int weightGoal = 100;
    }
## Enum
    enum type_name {
        value1,
        value2,
        value3,
        .
        .
    } object_names;

    enum MONTHS {Jan, Feb, Mar, Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec};

    //define bestMonth as a variable type MONTHS
    MONTHS bestMonth = Jan;
    if(bestMonth == Jan)
    {
        cout<<"I'm not so sure January is the best month\n";
    }
## Header Files
    #include "my.hpp"
## Stringstream
    #include <iostream>
    #include <string>
    #include <sstream>

    int main ()
    {
    std::string stringLength, stringWidth;
    float length = 0;
    float width = 0;
    float area = 0;

    std::cout << "Enter the length of the room: ";
    //get the length as a string
    std::getline (std::cin,stringLength);
    //convert to a float
    std::stringstream(stringLength) >> length;
    //get the width as a string
    std::cout << "Enter width: ";
    std::getline (std::cin,stringWidth);
    //convert to a float
    std::stringstream(stringWidth) >> width;
    area = length * width;
    std::cout << "\nThe area of the room is: " << area << std::endl;
    return 0;
    }
## Pointers
http://www.cplusplus.com/doc/tutorial/pointers/

Java is all pointers, they call references.
C++ use a lot of local variables, actually fewer pointers, so it run faster because no need to dereference pointers.
Pointers are the addresses of variables.
54 is the value of the variable. where is a? The location of 'a' can be found using a pointer!

    int a = 54;
     // this is an integer variable with value = 54
    std::cout<< &a<<"\n";
    //This will print the LOCATION of 'a'

    // this is a pointer that holds the address of the variable 'a'.
    // if 'a' was a float, rather than int, so should be its pointer.
    int * pointerToA = &a;

    // If we were to print pointerToA, we'd obtain the address of 'a':
    std::cout << "pointerToA stores " << pointerToA << '\n';

    // If we want to know what is stored in this address, we can dereference pointerToA:
    std::cout << "pointerToA points to " << * pointerToA << '\n';

    int *pointerGivenInt;
    int **pointerPointerGivenInt;
    pointerGivenInt = &givenInt;
    pointerPointerGivenInt = &pointerGivenInt;
    std::cout<< "pointer of givenInt = " << *pointerGivenInt<<"\n";
    std::cout<< "pointer of pointer of givenInt = " << **pointerPointerGivenInt<< "\n";

if we have a pointer and want to access the value stored in that address? That process is called dereferencing, and it is indicated by adding the operator * before the variable's name. This same operator should be used to declare a variable that is meant to store a pointer

compare these two:

    int a = 54;
    int * pointerToA = &a;
    // in decleration it should assign an address
and

    int * pointerI;
    int number;
    pointerI = &number;
    *pointerI = 45;
    // now it means dereference

modify value in a pointer

    #include<iostream>
    #include<string>

    int main ()
    {
    int * pointerI;
    int number;
    pointerI = &number;
    * pointerI = 45;
    std::cout << "number = "<<number<<"\n";
    std::cout << "*pointerI = "<<*pointerI<<"\n";
    * pointerI = 46;
    std::cout << "number = "<<number<<"\n";
    number = 47;
    std::cout << "number = "<<number<<"\n";
    std::cout << "*pointerI = "<<*pointerI<<"\n";
    return 0;
    }
    // output following:
    number = 45
    *pointerI = 45
    number = 46
    number = 47
    *pointerI = 47
## Class
the default is to make all members private.
Private members are listed first. If you do this, there is no need to use the 'private' keyword. If you list them after the public keyword, you will need to identify them using the private keyword.
So we add the keyword "public" and all members listed after it are accessible:

    #include<iostream>
    using namespace std;

    class Dog
    {
        private:
            int license;
        public:
            Dog();
            Dog(int licenseIn);
            void setLicense(int licenseIn);
            int getLicense();
            ~Dog();
    };

    Dog::Dog()
    {
    license = 0;
    }

    Dog::~Dog()
    {
        cout<<"\nDeleting the dog";
    }
    Dog::Dog(int licenseIn)
    {
    license = licenseIn;
    }
    void Dog::setLicense(int licenseIn)
    {
        license = licenseIn;
    }
    int Dog::getLicense()
    {
        return license;
    }
    int main()
    {
        Dog d2(666666);
        cout<<d2.getLicense();
        return 0;
    }
It is conventional to put classes in a header file.
## Memory Mangement
Start braces signify the start of memory management and end braces cleanup.
### Constructors
A constructor is special function that is executed whenever we create a new instance of the class. It is used to set initial values of data members of the class. **Constructors do not return a value, including void.**

    // The declaration for a constructor is:
    ClassName::ClassName();
    // The definition of a constructor is:
    ClassName::ClassName()
    {
        dataMemberName1 = value;
        dataMemberName2 = value;
        ...
    }
### Destructors
Destructors are special class functions that are called whenever an object goes out of scope. Just like a constructor, a destructor is called automatically. Destructors must have the same name as the class.. Destructors cannot:
- return a value
- accept parameters
```
// Declaring a destructor:
~className()  //this is a destructor
// Defining a destructor:
classname::~classname()
{
     //tasks to be completed before going out of scope
}
```
One of the more important tasks of a destructor is releasing memory that was allocated by the class constructor and member functions.


## IDE
http://www.codeblocks.org/

## cheatsheet
https://en.cppreference.com/w/cpp/language/operator_incdec
