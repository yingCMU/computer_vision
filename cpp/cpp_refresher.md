# Data Types
- [Why size_t matters](https://www.embedded.com/why-size_t-matters/)
- uint32_t
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
## Namespace scope
The potential scope of a name declared in a namespace begins at the point of declaration and includes the rest of the namespace and all namespace definitions with an identical namespace name that follow, plus, for any using-directive that introduced this name or its entire namespace into another scope, the rest of that scope.

The top-level scope of a translation unit ("file scope" or "global scope") is also a namespace and is properly called "global namespace scope". The potential scope of a name declared in the global namespace scope begins at the point of declaration and ends at the end of the translation unit.

The potential scope of a name declared in an unnamed namespace or in an inline namespace includes the potential scope that name would have if it were declared in the enclosing namespace.

```
namespace N { // scope of N begins (as a member of global namespace)
    int i; // scope of i begins
    int g(int a) { return a; } // scope of g begins
    int j(); // scope of j begins
    void q(); // scope of q begins
    namespace {
        int x; // scope of x begins
    } // scope of x continues (member of unnamed namespace)
    inline namespace inl { // scope of inl begins
        int y; // scope of y begins
    } // scope of y continues (member of inline namespace)
} // scopes of i, g, j, q, inl, x, and y pause
```
```
namespace {
    int l = 1; // scope of l begins
} // scope of l continues (member of unnamed namespace)
```
```
namespace N { // scopes of i, g, j, q, inl, x, and y resume
    int g(char a) { // overloads N::g(int)
        return l + a; // l from unnamed namespace is in scope
    }
//  int i; // error: duplicate definition (i is already in scope)
    int j(); // OK: duplicate function declaration is allowed
    int j() { // OK: definition of the earlier-declared N::j()
        return g(i); // calls N::g(int)
    }
//  int q(); // error: q is already in scope with a different return type
} // scopes of i, g, j, q, inl, x, and y pause

int main() {
    using namespace N; // scopes of i, g, j, q, inl, x, and y resume
    i = 1; // N::i is in scope
    x = 1; // N::(anonymous)::x is in scope
    y = 1; // N::inl::y is in scope
    inl::y = 2; // N::inl is also in scope
} // scopes of i, g, j, q, inl, x, and y end
```
## Class scope
The potential scope of a name declared in a class begins at the point of declaration and includes the rest of the class body, all the derived classes bodies, the function bodies (even if defined outside the class definition or before the declaration of the name), function default arguments, function exception specifications, in-class brace-or-equal initializers, and all these things in nested classes, recursively.

## Unqualified name lookup
For an unqualified name, that is a name that does not appear to the right of a scope resolution operator ::, name lookup examines the scopes as described below, until it finds at least one declaration of any kind, at which time the lookup stops and no further scopes are examined.

For a name used in a user-declared namespace outside of any function or class, this namespace is searched before the use of the name, then the namespace enclosing this namespace before the declaration of this namespace, etc until the global namespace is reached.
```
int n = 1; // declaration

namespace N
{
    int m = 2;

    namespace Y
    {
        int x = n; // OK, lookup finds ::n
        int y = m; // OK, lookup finds ::N::m
        int z = k; // Error: lookup fails
    }

    int k = 3;
}
```
## Understand function declarations
Because you have defined your function in a separate file outside of main.cpp, you can more easily re-use the function in other parts of your code.

Notice that you still had to declare the distance function at the top of main.cpp to be able to use the function.

```
#include <iostream>

float distance(float velocity, float acceleration, float time_elapsed);

int main() {

    std::cout << distance(3, 4, 5) << std::endl;
    std::cout << distance(7.0, 2.1, 5.4) << std::endl;

    return 0;
}
```
-- distance.cpp--
```
float distance(float velocity, float acceleration, float time_elapsed) {
    return velocity*time_elapsed + 0.5*acceleration*time_elapsed*time_elapsed;
}
```
### Header Files
The function declaration `float distance(float velocity, float acceleration, float time_elapsed);`
is oftentimes put into its own file as well. The declaration is kept in what's called a header file because the header is the information above the main() function. Header files generally have either a .h or .hpp extension. Here is the same code above but with the function declaration in a header file.
```
#include <iostream>
#include "distance.h"

int main() {

    std::cout << distance(3, 4, 5) << std::endl;
    std::cout << distance(7.0, 2.1, 5.4) << std::endl;

    return 0;
}
```
`#include "distance.h"`
will paste the contents of distance.h into main.cpp.

To compile the code, you only need to compile the .cpp files but not the .h file: `g++ main.cpp distance.cpp`
### File Naming
Naming conventions dictate that the header file and associated cpp file have the same name.
### Include syntax
You might be wondering why there are two different types of include statements:
```
#include <iostream>
#include "distance.h"
```
The include statement with quotes tells the program to look for the distance.h file in the current directory.
The <> syntax will depend on your C++ environment. Generally, environments are set up to look for the file where the C++ libraries are stored like the Standard Library.
## Python vs C++ speed
In this example, we'll be comparing the execution speed of C++ and Python implementations of the move function that Kalman filters use to update their estimate of a car's location as it moves.

The move function does two things:

It shifts a set of prior beliefs (about the car's location) in whichever direction the car moves.
It adds some uncertainty to the beliefs because our model for car movement is not perfect.

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
## When do we pass arguments by reference or pointer?
https://www.geeksforgeeks.org/when-do-we-pass-arguments-by-reference-or-pointer/
1) To modify local variables of the caller function
2) For passing large sized arguments: If an argument is large, passing by reference (or pointer) is more efficient because only an address is really passed, not the entire object. For example, let us consider the following Employee class and a function printEmpDetails() that prints Employee details.
3) To avoid Object Slicing: If we pass an object of subclass to a function that expects an object of superclass then the passed object is sliced if it is pass by value. For example, consider the following program, it prints “This is Pet Class”.  This point is valid only for struct and class variables as we don’t get any efficiency advantage for basic types like int, char, etc.
```
#include <iostream>

using namespace std;

class Pet {
public:
    virtual string getDescription() const
    {
        return "This is Pet class";
    }
};

class Dog : public Pet {
public:
    virtual string getDescription() const
    {
        return "This is Dog class";
    }
};

void describe(Pet p)
{ // Slices the derived class object
    cout << p.getDescription() << '\n';
}

int main()
{
    Dog d;
    describe(d);
    return 0;
}
```
Output:
This is Pet Class <br>
```
#include <iostream>

using namespace std;

class Pet {
public:
    virtual string getDescription() const
    {
        return "This is Pet class";
    }
};

class Dog : public Pet {
public:
    virtual string getDescription() const
    {
        return "This is Dog class";
    }
};

void describe(const Pet& p)
{ // Doesn't slice the derived class object.
    cout << p.getDescription() << '\n';
}

int main()
{
    Dog d;
    describe(d);
    return 0;
}
```
Output:
This is Dog Class <br>

4) To achieve Run Time Polymorphism in a function: We can make a function polymorphic by passing objects as reference (or pointer) to it. For example, in the following program, print() receives a reference to the base class object. Function print() calls the base class function show() if base class object is passed, and derived class function show() if derived class object is passed. This point is also not valid for basic data types like int, char, etc.
```
#include <iostream>
using namespace std;

class base {
public:
    virtual void show()
    { // Note the virtual keyword here
        cout << "In base\n";
    }
};

class derived : public base {
public:
    void show() { cout << "In derived\n"; }
};

// Since we pass b as reference, we achieve run time
// polymorphism here.
void print(base& b) { b.show(); }

int main(void)
{
    base b;
    derived d;
    print(b);
    print(d);
    return 0;
}
```
Output:
```
In base
In derived
```
## Pointers
- [pointer basics](http://www.cplusplus.com/doc/tutorial/pointers/)
- [Pointer declaration](https://en.cppreference.com/w/cpp/language/pointer)

we can declare and initialize pointer at same step or in multiple line.
```
int a = 10;
  int *p = &a;
```
```
    int *p;
    p = &a;
```
But a reference is different:
```
int a=10;
int &p=a;  //it is correct
   but
int &p;
 p=a;    // it is incorrect as we should declare and initialize references at single step.
 ```

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

## References
Call by value: By default, cpp uses call by value to pass arguments, which means a function can not alter the arguments used.

- [Reference declaration](https://en.cppreference.com/w/cpp/language/reference): Declares a named variable as a reference, that is, an alias to an already-existing object or function.
```
    std::string s = "Ex";
    std::string& r1 = s;
    const std::string& r2 = s;

    r1 += "ample";           // modifies s
//  r2 += "!";               // error: cannot modify through reference to const
    std::cout << r2 << '\n'; // prints s, which now holds "Example"
```

https://www.educative.io/edpresso/differences-between-pointers-and-references-in-cpp
References can be used ,simply, by name.

    int a = 5;
    int &ref = a;

### pass-by-reference semantics in function calls
References allow modifying variable values passed to functions or to avoid passing an entire copy (value) of a large object to a function.

    void cpp_ReferenceGood(int& r) {   // reference to an integer
        r = 40;                       // modify the referenced object
    }
    void cpp_ReferencesBad(const int& r) {   // constant reference to   message
        r = 20;                     // !!!compile error!!!
    }

    void cpp_References() {
        int x = 10;
        cpp_ReferenceGood(x);
        cout << dec << x << endl;     // ensure cout not still in hex mode.
       // Output is "40".
    }
Warning - only use references for:
- 1) passing data to functions (either to modify or as an efficient way to avoid passing value), or
- 2) As return types of operator overload functions. Never keep or hold a reference or return them from functions
### Access memebers of a pointer to class object
using ->: The -> operator dereferences the pointer. The expressions e->member and (*(e)).member (where e represents a pointer) yield identical results (except when the operators -> or * are overloaded).

The member access operators . and -> are used to refer to members of struct, union, and class types. Member access expressions have the value and type of the selected member.

https://docs.microsoft.com/en-us/cpp/cpp/member-access-operators-dot-and?view=msvc-170

https://en.cppreference.com/w/cpp/language/operator_member_access#Built-in_member_access_operators

### Pointer to member operators
The pointer-to-member operators .* and ->* return the value of a specific class member for the object specified on the left side of the expression. <br>
https://docs.microsoft.com/en-us/cpp/cpp/pointer-to-member-operators-dot-star-and-star?view=msvc-170
## Difference between reference and pointer
https://www.geeksforgeeks.org/pointers-vs-references-cpp/?ref=lbp

A pointer in C++ is a variable that holds the memory address of another variable.

A reference is an alias for an already existing variable. Once a reference is initialized to a variable, it cannot be changed to refer to another variable. Hence, a reference is similar to a const pointer.

Note that the asterisk (*) used when declaring a pointer only means that it is a pointer (it is part of its type compound specifier), and should not be confused with the dereference operator seen a bit earlier, but which is also written with an asterisk (*). They are simply two different things represented with the same sign.

A pointer has itws own memory address whereas a reference shares the same memory addres with the original varable:
```
 int &p = a;
   cout << &p << endl << &a;
```

When to use What?

The performances are exactly the same, as references are implemented internally as pointers. But still you can keep some points in your mind to decide when to use what :
- Use references : In function parameters and return types.
- Use pointers:
    - Use pointers if pointer arithmetic or **passing NULL-pointer is needed**. For example for arrays (Note that array access is implemented using pointer arithmetic).
    - To implement data structures like linked list, tree, etc and their algorithms because to point different cell, we have to use the concept of pointers.
## Passing By Pointer Vs Passing By Reference in C++
1) Passing by Pointer: Here, the memory location of the variables is passed to the parameters in the function, and then the operations are performed.
2) Passing by Reference: It allows a function to modify a variable without having to create a copy of it. We have to declare reference variables. The memory location of the passed variable and parameter is the same and therefore, any change to the parameter reflects in the variable as well.

A reference is the same object, just with a different name and a reference must refer to an object. Since references can’t be NULL, they are safer to use.

- A pointer can be re-assigned while a reference cannot, and must be assigned at initialization only.
- The pointer can be assigned NULL directly, whereas the reference cannot.
- Pointers can iterate over an array, we can use increment/decrement operators to go to the next/previous item that a pointer is pointing to.
- A pointer is a variable that holds a memory address. A reference has the same memory address as the item it references.
- A pointer to a class/struct uses ‘->’ (arrow operator) to access its members whereas a reference uses a ‘.’ (dot operator)
- A pointer needs to be dereferenced with * to access the memory location it points to, whereas a reference can be used directly.
### Passing pointer to a pointer as a parmeter to function
If a pointer is passed to a function as a parameter and tried to be modified then the changes made to the pointer does not reflects back outside that function. This is because only a copy of the pointer is passed to the function. It can be said that “pass by pointer” is passing a pointer by value. In most cases, this does not present a problem. But the problem comes when you modify the pointer inside the function. Instead of modifying the variable, you are only modifying a copy of the pointer and the original pointer remains unmodified.
```
#include <iostream>

using namespace std;

int global_Var = 42;

// function to change pointer value
void changePointerValue(int* pp)
{
    pp = &global_Var;
}

int main()
{
    int var = 23;
    int* ptr_to_var = &var;

    cout << "Passing Pointer to function:" << endl;

    cout << "Before :" << *ptr_to_var << endl; // display 23

    changePointerValue(ptr_to_var);

    cout << "After :" << *ptr_to_var << endl; // display 23

    return 0;
}
```
Output:
```
Passing Pointer to function:
Before :23
After :23
```
This problem can ve resolved by passing the address of the pointer to the function
```
int global_var = 42;

// function to change pointer to pointer value
void changePointerValue(int** ptr_ptr)
{
    *ptr_ptr = &global_var;
}

int main()
{
    int var = 23;
    int* pointer_to_var = &var;

    cout << "Passing a pointer to a pointer to function " << endl;

    cout << "Before :" << *pointer_to_var << endl; // display 23

    changePointerValue(&pointer_to_var);

    cout << "After :" << *pointer_to_var << endl; // display 42

    return 0;
}
```
A reference allows called function to modify a local variable of the caller function. For example, consider the following example program where fun() is able to modify local variable x of main().
```
void fun(int &x) {
    x = 20;
}

int main() {
    int x = 10;
    fun(x);
    cout<<"New value of x is "<<x;
    return 0;
}
```
Output:
New value of x is 20

Below program shows how to pass a “Reference to a pointer” to a function:
```
// function to change Reference to pointer value
void changeReferenceValue(int*& pp)
{
    pp = &gobal_var;
}

int main()
{
    int var = 23;
    int* ptr_to_var = &var;

    cout << "Passing a Reference to a pointer to function" << endl;

    cout << "Before :" << *ptr_to_var << endl; // display 23

    changeReferenceValue(ptr_to_var);

    cout << "After :" << *ptr_to_var << endl; // display 42

    return 0;
}
```
### Returning reference from function
```
#include <iostream>

using namespace std;

int global_var = 42;

// function to return reference value
int& ReturnReference()
{
    return global_var;
}

int main()
{
    int var = 23;
    int* ptr_to_var = &var;

    cout << "Returning a Reference " << endl;

    cout << "Before :" << *ptr_to_var << endl; // display 23

    ptr_to_var = &ReturnReference();

    cout << "After :" << *ptr_to_var << endl; // display 42

    return 0;
}
```
output:
```
Returning a Reference
Before :23
After :42
```
### On References and Pointers
References are typically used to be able to modify a value passed to a function. They can also be used to pass a reference to an object instead of copying it to functions. But:
1. References can never be null which means they cannot be used for optional values
2. References do not have ownership semantics, so objects cannot be deleted using references

To have a class optionally (dynamically) contain an object of another class one must use a pointer. And pointers provide mechanisms for controlling the ownership of the dynamically created object.
There are 3 kinds of pointers:
- unique_ptr<> allows one object to own another object,
- shared_ptr<> allows an object to be shared amongst other objects, and
- weak_ptr<> is used when no ownership is needed.
Warning -Do not use legacy C-style raw pointers and deletealways use containment, unique_ptr, or shared_ptr

### New Operator
https://docs.microsoft.com/en-us/cpp/cpp/new-operator-cpp?view=msvc-170
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

    class Patient
    {
        private:
            string name;
        public:
            //The constructor accepts a name parameter
            Patient(string input);
            void setName(string input);
            string getName();
    };

    Patient::Patient(string input)
    {
        //when an object is created
        //the name must be added as a parameter
        name = input;
    }

    void Patient::setName(string input)
    {
        name = input;
    }

    string Patient::getName()
    {
        return name;
    }
    int main()
    {
        //an instance of Patient is
        //instanciated with a name
        Patient p1("Tammy Smith");
        cout<<p1.getName();
        return 0;
    }
## [Initialization](https://en.cppreference.com/w/cpp/language/initialization)
1. [Declarators](https://en.cppreference.com/w/cpp/language/declarations)
2. [new expression](https://en.cppreference.com/w/cpp/language/new)

### [Member Initialize lists](https://en.cppreference.com/w/cpp/language/constructor)
```
#include <fstream>
#include <string>
#include <mutex>
#include <iostream>
struct Base
{
    int n;
};

struct Class : public Base
{
    unsigned char x;
    unsigned char y;
    std::mutex m;
    std::lock_guard<std::mutex> lg;
    std::fstream f;
    std::string s;

    Class(int x) : Base{123}, // initialize base class
        x(x),     // x (member) is initialized with x (parameter)
        y{0},     // y initialized to 0
        f{"test.cc", std::ios::app}, // this takes place after m and lg are initialized
        s(__func__), // __func__ is available because init-list is a part of constructor
        lg(m),    // lg uses m, which is already initialized
        m{}       // m is initialized before lg even though it appears last here
    {}            // empty compound statement

    Class(double a) : y(a + 1),
        x(y), // x will be initialized before y, its value here is indeterminate
        lg(m)
    {} // base class initializer does not appear in the list, it is
       // default-initialized (not the same as if Base() were used, which is value-init)

    Class()
    try // function-try block begins before the function body, which includes init list
      : Class(0.0) // delegate constructor
    {
        // ...
    }
    catch (...)
    {
        // exception occurred on initialization
    }
};

int main()
{
    Class c;
    Class c1(65);
    Class c2(0.1);
    Base b{123};
    std::cout <<"c.x=" << c.x << std::endl;

    std::cout <<"c1.x=" << c1.x << std::endl;
    std::cout <<"c1.n=" <<c1.n<< std::endl;
    std::cout <<"c2.y=" <<c2.y<< std::endl;
}
```
Output:
- note x and y is char, 0 does not print out anything;
- in c2, x is declared first in class definition so it is initialized before y.
```
c.x=
c1.x=A
c1.n=123
c2.y=
```
### Initialization order
The order of member initializers in the list is irrelevant: the actual order of initialization is as follows:

1) If the constructor is for the most-derived class, virtual bases are initialized in the order in which they appear in depth-first left-to-right traversal of the base class declarations (left-to-right refers to the appearance in base-specifier lists)
2) Then, direct bases are initialized in left-to-right order as they appear in this class's base-specifier list
3) Then, non-static data member are initialized in order of declaration in the class definition.
4) Finally, the body of the constructor is executed
(Note: if initialization order was controlled by the appearance in the member initializer lists of different constructors, then the destructor wouldn't be able to ensure that the order of destruction is the reverse of the order of construction)

## Parameter pack
https://en.cppreference.com/w/cpp/language/parameter_pack

A variadic class template can be instantiated with any number of template arguments:
```
template<class... Types> struct Tuple {};
Tuple<> t0;           // Types contains no arguments
Tuple<int> t1;        // Types contains one argument: int
Tuple<int, float> t2; // Types contains two arguments: int and float
Tuple<0> t3;          // error: 0 is not a type
```
A variadic function template can be called with any number of function arguments (the template arguments are deduced through template argument deduction):
```
template<class... Types> void f(Types... args);
f();       // OK: args contains no arguments
f(1);      // OK: args contains one argument: int
f(2, 1.0); // OK: args contains two arguments: int and double
```
### Constructors and member initializer lists
https://en.cppreference.com/w/cpp/language/constructor

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
## This Pointer
'this' returns its own address.
## Overloading Operators
the function must specify a return type. Then it must use the keyword 'operator' followed by the '+' sign.

    //The function that overloads the '+' sign
      int operator + (Shape shapeIn)
      {
          return Area() + shapeIn.Area();
      }

      int total = sh1 + sh2;

## Generic Template
The function declaration:

    template <typename T>  //tell the compiler we are using a template

    //T represents the variable type. Since we want it to be for any type, we
    //use T
    T  functionName (T parameter1,T parameter2, ...);


The function definition:

    template <typename T>
    T functionName (T  parameter1,T  parameter2,...)
    {
        function statements;
    }
Example

    #include<iostream>

    using namespace std;

    //Our generic function
    template <typename T>  //tell the compiler we are using a template
    T findSmaller(T input1,T  input2);

    int main()
    {
        int a = 54;
        int b = 89;
        float f1 = 7.8;
        float f2 = 9.1;
        char c1 = 'f';
        char c2 = 'h';
        string s1 = "Hello";
        string s2 = "Bots are fun";

        //Wow! We can use one function for different variable types
        cout<<"\nIntegers compared: "<<findSmaller(a,b);
        cout<<"\nFloats compared: "<<findSmaller(f1,f2);
        cout<<"\nChars compared: "<<findSmaller(c1,c2);
        cout<<"\nStrings compared: "<<findSmaller(s1,s2);
        return 0;
    }

    template <typename T>
    T findSmaller(T  input1,T  input2)
    {
        if(input1 < input2)
            return input1;
        else
            return input2;
    }

## Template with different types
    template <typename T, typename U, typename V>
    T functionName (U  parameter1, V  parameter2,...)
    {
        function statements;
    }

Examples:

    #include<iostream>
    using namespace std;

    template <typename T, typename U>
    T getBigger(T input1, U input2);


    int main()
    {
        int a = 5;
        float b = 6.334;
        int bigger;
        cout<<"Between "<<a<<" and "<<b<<" "<<getBigger(a,b)<<" is bigger.\n";

        cout<<"Between "<<a<<" and "<<b<<" "<<getBigger(b,a)<<" is bigger.\n";
        return 0;
    }

    template <typename T, typename U>
    T getBigger(T input1, U input2)
    {
        if(input1 > input2)
            return input1;
        return input2;
    }

    // output
    Between 5 and 6.334 6 is bigger.
    Between 5 and 6.334 6.334 is bigger.
## Generic Classes with template
If the class is going to accept strings, we need to include the 'using namespace' compiler directive, or it will not recognize the string variable.

    //header file for main.cpp
    #include<iostream>

    //The class accepts strings, so we need to use namespace
    using namespace std;

    //tell compiler this class uses a generic value
    template <class T>
    class StudentRecord
    {
        private:
            const int size = 5;
            T grade;
            int studentId;
        public:
        //note: I used a constructor that accepts the grade input
            StudentRecord(T input);
            void setId(int idIn);
            void printGrades();
    };
**The member functions must all be treated as generic functions. You will have to add the template command to each member function.**

    template<class T>
    StudentRecord<T>::StudentRecord(T input)
    {
        grade=input;
    }

    //Notice I still add the template<class T here, even though this is not a generic //function. It is in a generic class.

    template<class T>
    void StudentRecord<T>::setId(int idIn)
    {
        studentId = idIn;
    }

    template<class T>
    void StudentRecord<T>::printGrades()
    {
        cout<<"ID# "<<studentId<<": ";
        cout<<grade<<"\n ";
        cout<<"\n";
    }

    int main()
    {
        //StudentRecord is the generic class
        //The constructor sets the grade value
        StudentRecord<int> srInt(3);
        srInt.setId(111111);
        srInt.printGrades();

        StudentRecord<char> srChar('B');
        srChar.setId(222222);
        srChar.printGrades();

        StudentRecord<float> srFloat(3.333);
        srFloat.setId(333333);
        srFloat.printGrades();

        StudentRecord<string> srString("B-");
        srString.setId(4444);
        srString.printGrades();

        return 0;
    }

    //output:
    ID# 111111: 3
    ID# 222222: B
    ID# 333333: 3.333
    ID# 4444: B-

**Generic classes with arrays will not compile without initializing the array, example error:**

    main.cpp:12:11: error: request for member ‘setId’ in ‘srInt’, which is of non-class type ‘StudentRecord<int>()’
        srInt.setId(111111);
            ^
Use a constructor to allocate memory for the array.

    //header file for main.cpp

    #include<iostream>

    using namespace std;

    const int SIZE = 5;
    template <class T>
    class StudentRecord
    {
        private:
            const int size = SIZE;
            T grades[SIZE];
            int studentId;
        public:
            StudentRecord(T defaultInput);//A default constructor with a default value
            void setGrades(T* input);
            void setId(int idIn);
            void printGrades();
    };

    template<class T>
    StudentRecord<T>::StudentRecord(T defaultInput)
    {
        //we use the default value to allocate the size of the memory
        //the array will use
        for(int i=0; i<SIZE; ++i)
            grades[i] = defaultInput;

    }


    template<class T>
    void StudentRecord<T>::setGrades(T* input)
    {
        for(int i=0; i<SIZE;++i)
        {
            grades[i] = input[i];
        }
    }

    template<class T>
    void StudentRecord<T>::setId(int idIn)
    {
        studentId = idIn;
    }

    template<class T>
    void StudentRecord<T>::printGrades()
    {
        std::cout<<"ID# "<<studentId<<": ";
        for(int i=0;i<SIZE;++i)
            std::cout<<grades[i]<<"\n ";
        std::cout<<"\n";
    }

    /*Goal: study generic classes
    **Fix the program by completing
    **the constructor. It should
    **assign a default value to each
    **element in the array.*/

    int main()
    {
        //StudentRecord is the generic class
        //The constructor sets the grade value
        StudentRecord<int> srInt(-1);//add a default value using a constructor
        srInt.setId(123456);
        int arrayInt[SIZE]={0,0,0,0};
        srInt.setGrades(arrayInt);
        srInt.printGrades();

        StudentRecord<char> srChar('U');//add a default value using a constructor
        srChar.setId(234567);
        char arrayChar[SIZE]={'F','F','F','F','E'};
        srChar.setGrades(arrayChar);
        srChar.printGrades();

        StudentRecord<float> srFloat(-1.0);//add a default value using a constructor
        srFloat.setId(345678);
        float arrayFloat[SIZE]={2.75,4.0,3.7,2.8,3.99};
        srFloat.setGrades(arrayFloat);
        srFloat.printGrades();

        StudentRecord<string> srString("U");//add a default value using a constructor
        srString.setId(456789);
        string arrayString[SIZE]={"B","B-","C+","B","A"};
        srString.setGrades(arrayString);
        srString.printGrades();

        return 0;
    }

## Class Inheritance

    /*The header file for inheritance.*/

    #include<iostream>
    #include<string>
    using namespace std;

    //The base class
    class Student
    {
        private:
            int id;
        public:
            void setId(int idIn);
            int getId();
            Student();
    };

    Student::Student()
    {
        id = 000000000;
    }

    void Student::setId(int idIn)
    {
        id = idIn;
    }

    int Student::getId()
    {
        return id;
    }

    //The derived class with Student as base class
    class GradStudent : public Student
    {
        private:
            string degree;
        public:
            GradStudent();
            void setDegree(string degreeIn);
            string getDegree();
    };

    GradStudent::GradStudent()
    {
        degree = "undelcared";
    }
    void GradStudent::setDegree(string degreeIn)
    {
        degree = degreeIn;
    }
    string GradStudent::getDegree()
    {
        return degree;
    }
### Access control

    //The derived class with Student as base class
    class GradStudent : private Student
    {
        private:
            string degree;
        public:
            GradStudent();
            void setDegree(string degreeIn);
            string getDegree();
            void setStudentId(int idIn); //need this to access Student::setId()
            int getStudentId(); //need this to access Student::getId()
    };
Now that we have a private inheritance, the Student member functions setId() and getID() are no longer available to the GradStudent class.

When we write the member functions, we must explicitly refer to the Student class.

    int GradStudent::getStudentId()
    {
        //We must access getId() as a private function
        return Student::getId();
    }
    void GradStudent::setStudentId(int idIn)
    {
        //We must access setId() as a private function
        Student::setId(idIn);
    }
### Multiple Inheritance
C++ classes can inherit from more than one class. This is known as "Multiple Inheritance".

    class DerivedClass : access BaseClass1, ... ,access BaseClassN
Example

    #include<iostream>
    #include<string>
    using namespace std;

    class Staff
    {
        private:
            string title;
        public:
            Staff();
            void setTitle(string input);
            string getTitle();
    };

    Staff::Staff()
    {
        title = "NA";
    }

    void Staff::setTitle(string input)
    {
        title = input;
    }

    string Staff::getTitle()
    {
        return title;
    }

    class GradStudent
    {
        private:
            int studentId;
        public:
            GradStudent();
            void setId(int input);
            int getId();

    };

    GradStudent::GradStudent()
    {
        studentId = 000000;
    }

    void GradStudent::setId(int input)
    {
        studentId = input;
    }

    int GradStudent::getId()
    {
        return studentId;
    }

    class TA: public Staff, public GradStudent
    {
        private:
            string supervisor;
        public:
            TA();
            void setSupervisor(string input);
            string getSupervisor();
    };

    TA::TA()
    {
        supervisor = "NA";
    }

    void TA::setSupervisor(string input)
    {
        supervisor = input;
    }

    string TA::getSupervisor()
    {
        return supervisor;
    }

## Virtual Functions
    #include "main.hpp"

    int main()
    {
        string status = "salary"; //options: hourly or salary
        string level;
        level = "hourly";
        Employee *e1; //e1 is now a pointer to Employee object

        if(status == level)
        {
            e1 = new Employee(); //we define an hourly employee
        }
        else
        {
            e1 = new Manager(); //we define a salaried employee
        }

    ...
    }

We assign an Employee pointer at the start of the program. This allocates memory to an Employee object. Then later, we define that same memory location as a manager. This should now supersede the previous definition, but it doesn't. **The term e1 is statically bound during compile.
We want it to be dynamically bound during execution.**
A [virtual function](https://docs.microsoft.com/en-us/cpp/cpp/virtual-functions?redirectedfrom=MSDN&view=msvc-170) is a member function that you expect to be redefined in derived classes. When you refer to a derived class object using a pointer or a reference to the base class, you can call a virtual function for that object and execute the derived class's version of the function. Solution:

    class Employee
    {
        private:
            float payRate;
            string name;
            int employeeNumber;
        public:
            void setPayRate(float rateIn);
            float getPayRate();
            //This is now a virtual function
            virtual float calcWeeklyPay();
    };
    void Employee::setPayRate(float rateIn)
    {
        payRate = rateIn;
    }
    float Employee::getPayRate()
    {
        return payRate;
    }
    float Employee::calcWeeklyPay()
    {
        return 40 * payRate;
    }

    //The class manager inherits from Employee
    //The only difference... managers are salary
    //employees. So the pay is calculated differently.
    class Manager : public Employee
    {
        public:
            float calcWeeklyPay();
    };

    float Manager::calcWeeklyPay()
    {
        //weekly pay is based on the yearly salary
        //divided by 52 weeks
        return Employee::getPayRate() /52;
    }
Note: we only had to add the keyword virtual in one location in the base class. Any class derived from Employee that has a function by the same name will inherit the same properties.

    virtual float calcWeeklyPay();
Pure Virtual Functions are a special case of virtual functions.

A pure virtual function is used when the base class has a function that will be defined in its derived class, but it has no meaningful definition in the base class.


```

#include<iostream>
using namespace std;

class base {
public:
    virtual void print()
    {
        cout << "print base class\n";
    }

    void show()
    {
        cout << "show base class\n";
    }
};

class derived : public base {
public:
    void print()
    {
        cout << "print derived class\n";
    }

    void show()
    {
        cout << "show derived class\n";
    }
};

int main()
{
    base *bptr;
    derived d;
    bptr = &d;

    // Virtual function, binded at runtime
    bptr->print();

    // Non-virtual function, binded at compile time
    bptr->show();

    return 0;
}
```
Runtime polymorphism is achieved only through a pointer (or reference) of base class type. Also, a base class pointer can point to the objects of base class as well as to the objects of derived class. In above code, base class pointer ‘bptr’ contains the address of object ‘d’ of derived class.
Late binding (Runtime) is done in accordance with the content of pointer (i.e. location pointed to by pointer) and Early binding (Compile time) is done according to the type of pointer, since print() function is declared with virtual keyword so it will be bound at runtime (output is print derived class as pointer is pointing to object of derived class) and show() is non-virtual so it will be bound during compile time (output is show base class as pointer is of base type).
NOTE: If we have created a virtual function in the base class and it is being overridden in the derived class then we don’t need virtual keyword in the derived class, functions are automatically considered as virtual functions in the derived class.

## Structures
Note - struct is a class with all members being public by default


    struct Person {
        string name;
        int age;
    };
Construct:

    void cpp_StructConstruct() {
        Person jon{"jon snow", 30};
        cout << jon.name << " is " << jon.age << " old" << endl;
    }
Pass by reference:

    void makeOlder(Person& p) {
        p.age += 1;
    }

    void cpp_StructByRef() {
        Person jon{"jon snow", 30};
        makeOlder(jon);
        cout << jon.name << " is " << jon.age << " old" << endl;
    }
Return as Value, meaining each return of person is a new one instead of pointing to the same object:

    Person makePerson(string first, string last, int age) {
        string name = first + " " + last;
        return Person{name, age};
    }

    void cpp_StructRetAsValue() {
    Person jon = makePerson("jon", "snow", 30);
    cout << jon.name << " is " << jon.age << " old" << endl;
    }
## Namespace
    // People/Friend.h
    #pragma once
    namespace People {
        class Friend {
            public:
                Friend(string name) : name_(name) { }
            private:
                string name_;
        };
    }
NAMESPACE EXPLICIT USAGE

    // #include "People/Friend"
    void cpp_NamespaceExplicit() {
        People::Friend bruce{"bruce wayne"};
    }
NAMESPACE IMPLICIT USAGE

    // #include "People/Friend"
    void cpp_NamespaceImplicit() {
        using namespace People;
        Friend ww{"wonder woman"};
    }
### namespace using declaration
A using declaration lets us use a name from a namespace without qualifying the name with a namespace_name:: prefix. A using declaration has the form

using namespace::name;

Once the using declaration has been made, we can access name directly:
```
#include <iostream>
// using declaration; when we use the name cin, we get the one from the namespace std
using std::cin;
int main()
{
    int i;
    cin >> i;       // ok: cin is a synonym for std::cin
    cout << i;      // error: no using declaration; we must use the full name
    std::cout << i; // ok: explicitly use cout from namepsace std
    return 0;
}
```
A Separate using Declaration Is Required for Each Name
```
#include <iostream>
// using declarations for names from the standard library
using std::cin;
using std::cout; using std::endl;
int main()
{
    cout << "Enter two numbers:" << endl;
    int v1, v2;
    cin >> v1 >> v2;
    cout << "The sum of " << v1 << " and " << v2
         << " is " << v1 + v2 << endl;
    return 0;
}
```
Headers Should Not Include using Declarations. The reason is that the contents of a header are copied into the including program’s text. If a header has a using declaration, then every program that includes that header gets that same using declaration. As a result, a program that didn’t intend to use the specified library name might encounter unexpected name conflicts.

## Deducing Types
auto and one for decltype:  The increasingly widespread application of type deduction frees you from the tyranny of spelling out types that are obvious or redundant. It makes C++ software more adaptable, because changing a type at one point in the source code automatically propagates through type deduction to other locations. However, it can render code more difficult to reason about, because the types deduced by compilers may not be as apparent as you’d like.
```
template<typename T>
void f(ParamType param);

f(expr);                // deduce T and ParamType from expr

```

## dynamic allocation (using new)
https://stackoverflow.com/questions/22146094/why-should-i-use-a-pointer-rather-than-the-object-itself

two ways of creating an object. The main difference is the storage duration of the object. When doing Object myObject; within a block, the object is created with automatic storage duration, which means it will be destroyed automatically when it goes out of scope. When you do new Object(), the object has dynamic storage duration, which means it stays alive until you explicitly delete it. You should only use dynamic storage duration when you need it. That is, you should always prefer creating objects with automatic storage duration when you can.
The main two situations in which you might require dynamic allocation:

You need the object to outlive the current scope - that specific object at that specific memory location, not a copy of it. If you're okay with copying/moving the object (most of the time you should be), you should prefer an automatic object.
You need to allocate a lot of memory, which may easily fill up the stack. It would be nice if we didn't have to concern ourselves with this (most of the time you shouldn't have to), as it's really outside the purview of C++, but unfortunately, we have to deal with the reality of the systems we're developing for.

Benefits of dynamic allocation
1. You don't have to know the size of the array in advance

One of the first problems many C++ programmers run into is that when they are accepting arbitrary input from users, you can only allocate a fixed size for a stack variable. You cannot change the size of arrays either. For example:
```

char buffer[100];
std::cin >> buffer;
// bad input = buffer overflow
```
Of course, if you used an std::string instead, std::string internally resizes itself so that shouldn't be a problem. But essentially the solution to this problem is dynamic allocation. You can allocate dynamic memory based on the input of the user, for example:
```
int * pointer;
std::cout << "How many items do you need?";
std::cin >> n;
pointer = new int[n];
```
Because the heap is much bigger than the stack, one can arbitrarily allocate/reallocate as much memory as he/she needs, whereas the stack has a limitation.
## Pointers usages
always prefer the alternatives unless you really need pointers.

1. You need reference semantics. Sometimes you want to pass an object using a pointer (regardless of how it was allocated) because you want the function to which you're passing it to have access that that specific object (not a copy of it). However, in most situations, you should prefer reference types to pointers, because this is specifically what they're designed for. Note this is not necessarily about extending the lifetime of the object beyond the current scope, as in situation 1 above. As before, if you're okay with passing a copy of the object, you don't need reference semantics.

2. You need polymorphism. You can only call functions polymorphically (that is, according to the dynamic type of an object) through a pointer or reference to the object. If that's the behavior you need, then you need to use pointers or references. Again, references should be preferred.

3. You want to represent that an object is optional by allowing a nullptr to be passed when the object is being omitted. If it's an argument, you should prefer to use default arguments or function overloads. Otherwise, you should preferably use a type that encapsulates this behavior, such as std::optional (introduced in C++17 - with earlier C++ standards, use boost::optional).

4. You want to decouple compilation units to improve compilation time. The useful property of a pointer is that you only require a forward declaration of the pointed-to type (to actually use the object, you'll need a definition). This allows you to decouple parts of your compilation process, which may significantly improve compilation time. See the Pimpl idiom.

5. You need to interface with a C library or a C-style library. At this point, you're forced to use raw pointers. The best thing you can do is make sure you only let your raw pointers loose at the last possible moment. You can get a raw pointer from a smart pointer, for example, by using its get member function. If a library performs some allocation for you which it expects you to deallocate via a handle, you can often wrap the handle up in a smart pointer with a custom deleter that will deallocate the object appropriately.

### Uninitialized Pointers
Uninitialized pointers is pointing to “aynwhere” which is of course an invalid memory location. It contains garbage data. This pointer may lead a program to behave wrongly or to crash.
```
int * Ptr;  //The pointer Ptr was declared without initialization
*Ptr = 17;
cout << *Ptr << endl;
```
### Null Pointers
We can initialize a pointer to 0 or NULL, which is pointing to nothing. Null pointer points to empty location in memory.
```
int * Ptr = 0;  // Initialize the pointer to point to nothing
int * ptr= NULL; // Also declare a NULL pointer points to nothing
```
C++11 introduces a new keyword called nullptr to represent null pointer.
int * p{nullptr}; // Also declare a nullptr pointer points to nothing
Initialize a pointer to null during declaration is a good software engineering practice.

### Why Use Pointers
Pointers can be used to pass of arrays and strings to functions more efficiently.
Pointers save the memory.
Pointers reduce the length and complexity of a program.
Pointers make possible to return more than one value from the function.
Pointers increase the processing speed. In other words, Execution time with pointers is faster because data are manipulated with the address, that is, direct access to memory location.
Memory is accessed efficiently with the pointers. The pointer assigns and releases the memory as well. Hence we can allocate memory dynamically on the heap or free store.
It can access specific address in memory.
It is useful in embedded and systems applications.
Pointer is used with data structures, which is useful for representing two-dimensional and multi-dimensional arrays.
Pointers are used for file handling.
Pointer declared to a base class could access the object of a derived class. However, a pointer to a derived class cannot access the object of a base class.
18

### Compare with Java - [source](https://hownot2code.com/2017/08/10/c-pointers-why-we-need-them-when-we-use-them-how-they-differ-from-accessing-to-object-itself/)
that pointers in Java are not used explicitly, e.g. a programmer cannot access to object in code through a pointer to it. However, in Java all types, except base, are referenced: accessing to them goes by the link, although you cannot explicitly pass the parameter by link. Besides that, new in C++ and Java or C# are different things.

In order to give a slight idea about the pointers in C++ , we’ll give two similar code fragments:

Java:
```
Object object1 = new Object();
//A new object is allocated by Java
Object object2 = new Object();
//Another new object is allocated by Java

object1 = object2;
//object1 now points to the object originally allocated for object2
//The object originally allocated for object1 is now "dead" –
//nothing points to it, so it
//will be reclaimed by the Garbage Collector.
//If either object1 or object2 is changed,
//the change will be reflected to the other
The closest equivalent to this, is:
```

C++:
```
Object * object1 = new Object();
//A new object is allocated on the heap
Object * object2 = new Object();
//Another new object is allocated on the heap
delete object1;
//Since C++ does not have a garbage collector,
//if we don't do that, the next line would
//cause a "memory leak", i.e. a piece of claimed memory that
//the app cannot use
//and that we have no way to reclaim...

object1 = object2;
//Same as Java, object1 points to object2.
```

Let’s see the alternative C++ way:
```
Object object1;
//A new object is allocated on the STACK
Object object2;
//Another new object is allocated on the STACK
object1 = object2;
//!!!! This is different!
//The CONTENTS of object2 are COPIED onto object1,
//using the "copy assignment operator", the definition of operator =.
//But, the two objects are still different.
//Change one, the other remains unchanged.
//Also, the objects get automatically destroyed
//once the function returns...
```
Pointers are usually used for the access to heap while the objects are located in stack – this is a simpler and quicker structure. First: when do we use dynamic memory allocation? Second: when is it better to use pointers?
## Stack and Heap
https://hownot2code.com/2017/08/09/programming-concepts-the-stack-and-the-heap/
- The stack is a region of RAM that gets created on every thread that your application is running on. It works in a LIFO (Last In, First Out) manner. Every time a function declares a new variable, it is “pushed” onto the stack, and after that variable falls out of scope (such as when the function closes), that variable will be deallocated from the stack automatically. Once a stack variable is freed, that region of memory becomes available for other stack variables.

Due to the pushing and popping nature of the stack, memory management is very logical and is able to be handled completely by the CPU; this makes it very quick, especially since each byte in the stack tends to be reused very frequently which means it tends to be mapped to the processor’s cache. However, there are some cons to this form of strict management. The size of the stack is a fixed value, and allocating more onto the stack than it can hold will result in a stack overflow. The size of the stack is **decided when the thread is created**, and each variable has a maximum size that it can occupy based on its data type; this prevents certain variables such as integers from ever growing beyond a certain value, and forces more complex data types such as arrays to specify their size prior to runtime since the stack won’t let them be resized. Variables allocated on the stack also are always local in nature because they are always next in line to be popped (unless more variables are pushed prior to the popping of earlier variables).

Overall, the stack really exceeds in managing memory in the most efficient way possible – but what if you need data structures that can be dynamic, such as a dynamically sized array, or what if you need global variables? This is where the heap comes into play.

- the heap is a memory store also in RAM that allows for dynamic memory allocation, and does not work on a stack-like basis, it’s more just a hub of storage for you to define your variables. Once you allocate a memory location on the heap to store a variable, that variable can be accessed at any point in time not only throughout just the thread, but throughout the application’s entire life. This is how you can define global variables. Once an application ends, all of the allocated memory locations are reclaimed by the CPU. The heap size is set on application startup, but unlike the stack there are no size restrictions on the heap (aside from the physical limitations of your machine), which means it can get ever larger as you allocate more memory to it. This is what allows you to create variables that can be dynamically resized, since the heap itself is dynamic in size.

**You interact with the heap via references typically called pointers**, which are variables whose values are the address of another variable, such as a memory location. By creating a pointer, you ‘point’ at a memory location on the heap, which is what signifies the initial location of your variable and tells the program where to access the value. Due to the dynamic nature of the heap, it is completely unmanaged by the CPU aside from initial allocation and heap resizing; in non-garbage collected languages such as C and C++, this requires you as the developer to manage memory and to manually free memory locations when they are no longer needed. Failing to do so can create memory leaks and cause memory to become fragmented, which will cause reads from the heap to take longer and makes it difficult to continuously allocate more memory onto the heap. <br>
Compared to the stack, the heap is slower to access because variables are scattered across memory instead of always sitting at the top of the stack. Improper memory management of the heap can also slow down reading from the heap; however, this shouldn’t detract from its importance – you absolutely need it to create any type of variable dynamically, or a global variable. <br>
Different languages use the stack and the heap differently; C and C++ allocate to the stack automatically, and you as the developer manually have to allocate and deallocate from the heap, where more modern languages such as Go and Java allocate to both the stack and the heap automatically, and have a garbage collector that handles heap deallocation on its own. There are even languages like Ruby and Python where everything is allocated on the heap and don’t use a stack at all.
## [The rule of three/five/zero](https://en.cppreference.com/w/cpp/language/rule_of_three)
## Array
compilers need to know what variable type and how many elements are required for an array at compile time. The information is necessary to allocate memory for the array.

## IDE
http://www.codeblocks.org/

## cheatsheet
https://en.cppreference.com/w/cpp/language/operator_incdec

## Interviews
https://www.toptal.com/c-plus-plus/interview-questions#iquestion_form


## It is a copy unless pass by reference or pointer
```
#include <iostream>
#include <vector>
void test(std::vector<int>& v ) {
    // Add two more integers to vector
    v.push_back(25);
    v.push_back(13);
    std::cout << "inside test() :";
    for (int n : v) {
        std::cout << n << ",";
    }
    std::cout << "\n";
}

int main()
{
    // Create a vector containing integers
    std::vector<int> v = { 7, 5, 16, 8 };

    test(v);

    // Print out the vector
    std::cout << "v = { ";
    for (int n : v) {
        std::cout << n << ", ";
    }
    std::cout << "}; \n";
}
```
output:
```
inside test() :7,5,16,8,25,13,
v = { 7, 5, 16, 8, 25, 13, };
```
if not using reference for test()'s paremeter result is
```
inside test() :7,5,16,8,25,13,
v = { 7, 5, 16, 8, };
```
alternative way of using pointer:
```
#include <iostream>
#include <vector>
void test(std::vector<int> * v ) {
    // Add two more integers to vector
    v->push_back(25);
    v->push_back(13);

    std::cout << "inside test() :";
    for (int n : *v) {
        std::cout << n << ",";
    }
    std::cout << "\n";
}

int main()
{
    // Create a vector containing integers
    std::vector<int> v = { 7, 5, 16, 8 };

    test(&v);

    // Print out the vector
    std::cout << "v = { ";
    for (int n : v) {
        std::cout << n << ", ";
    }
    std::cout << "}; \n";
}
```
## braces and parenthis
```
std::vector<int> v = { 7, 5, 16, 8 };  => v = { 7, 5, 16, 8};
std::vector<int> v{ 7, 5, 16, 8 };  => v = { 7, 5, 16, 8};
std::vector<int> v( 7, 5, 16, 8 );  // error: no matching function for call to 'std::vector<int>::vector(int, int, int, int)'
std::vector<int> v( 2); => {0,0}
vector<int> vect(3, 10); // Create a vector of size 3 with all values as 10.
int arr[] = { 10, 20, 30 }; int n = sizeof(arr) / sizeof(arr[0]); vector<int> vect(arr, arr + n);  //Initializing from an array
vector<int> vect1{ 10, 20, 30 }; vector<int> vect2(vect1.begin(), vect1.end());  // initialize from another vector
vector<int> vect1(10);int value = 5;fill(vect1.begin(), vect1.end(), value);  // initilize all to a value
```
## When to use pointer vs reference?
https://stackoverflow.com/questions/7058339/when-to-use-references-vs-pointers

Use reference wherever you can, pointers wherever you must.
void pointers until you can't.

The reason is that pointers make things harder to follow/read, less safe and far more dangerous manipulations than any other constructs.

So the rule of thumb is to use pointers only if there is no other choice.

For example, returning a pointer to an object is a valid option when the function can return nullptr in some cases and it is assumed it will. That said, a better option would be to use something similar to std::optional (requires C++17; before that, there's boost::optional).

Another example is to use pointers to raw memory for specific memory manipulations. That should be hidden and localized in very narrow parts of the code, to help limit the dangerous parts of the whole code base.

In your example, there is no point in using a pointer as argument because:

if you provide nullptr as the argument, you're going in undefined-behaviour-land;
the reference attribute version doesn't allow (without easy to spot tricks) the problem with 1.
the reference attribute version is simpler to understand for the user: you have to provide a valid object, not something that could be null.
If the behaviour of the function would have to work with or without a given object, then using a pointer as attribute suggests that you can pass nullptr as the argument and it is fine for the function. That's kind of a contract between the user and the implementation.
The performances are exactly the same, as references are implemented internally as pointers.
## shared_ptr snippet
```
#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <mutex>

struct Base
{
    Base() { std::cout << "  Base::Base()\n"; }
    // Note: non-virtual destructor is OK here
    ~Base() { std::cout << "  Base::~Base()\n"; }
};

struct Derived: public Base
{
    Derived() { std::cout << "  Derived::Derived()\n"; }
    ~Derived() { std::cout << "  Derived::~Derived()\n"; }
};

void thr(std::shared_ptr<Base> p)
{
    std::cout  << "before sleep, p.use_count() = " << p.use_count() << '\n';
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::shared_ptr<Base> lp = p; // thread-safe, even though the
                                  // shared use_count is incremented
    {
        static std::mutex io_mutex;
        std::lock_guard<std::mutex> lk(io_mutex);
        std::cout  << "in mutex, lp.use_count() = " << lp.use_count() << '\n';
    }
}

int main()
{
    std::shared_ptr<Base> p = std::make_shared<Derived>();

    std::cout  << "before t1, p.use_count() = " << p.use_count() << '\n';
    std::thread t1(thr, p);//, t2(thr, p), t3(thr, p);
    std::cout  << "after t1, p.use_count() = " << p.use_count() << '\n';
    p.reset(); // release ownership from main
    std::cout << "after reset(), p.use_count() = " << p.use_count() << '\n';
    t1.join(); //t2.join(); t3.join();
    std::cout << "All threads completed, the last one deleted Derived\n";
}
```
possible output: my guess is thread t1 is still holding 1 pointer even after main reset()
```
Base::Base()
  Derived::Derived()
before t1, p.use_count() = 1
after t1, p.use_count() = 2
after reset(), p.use_count() = 0
before sleep, p.use_count() = 1
in mutex, lp.use_count() = 2
  Derived::~Derived()
  Base::~Base()
All threads completed, the last one deleted Derived
```
if it is single thread?
```
#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <mutex>

struct Base
{
    Base() { std::cout << "  Base::Base()\n"; }
    // Note: non-virtual destructor is OK here
    ~Base() { std::cout << "  Base::~Base()\n"; }
};

struct Derived: public Base
{
    Derived() { std::cout << "  Derived::Derived()\n"; }
    ~Derived() { std::cout << "  Derived::~Derived()\n"; }
};

void thr(std::shared_ptr<Base> p)
{
    std::cout  << "before sleep, p.use_count() = " << p.use_count() << '\n';
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::shared_ptr<Base> lp = p; // thread-safe, even though the
                                  // shared use_count is incremented
    {
        static std::mutex io_mutex;
        std::lock_guard<std::mutex> lk(io_mutex);
        std::cout  << "in mutex, lp.use_count() = " << lp.use_count() << '\n';
    }
}

int main()
{
    std::shared_ptr<Base> p = std::make_shared<Derived>();

    std::cout  << "before t1, p.use_count() = " << p.use_count() << '\n';
    thr(p);//, t2(thr, p), t3(thr, p);
    std::cout  << "after t1, p.use_count() = " << p.use_count() << '\n';
    p.reset(); // release ownership from main
    std::cout << "after reset(), p.use_count() = " << p.use_count() << '\n';
    std::cout << "All threads completed, the last one deleted Derived\n";
}
```
after ths() function, the corresponding pointers because of the function are released, so output:
```
  Base::Base()
  Derived::Derived()
before t1, p.use_count() = 1
before sleep, p.use_count() = 2
in mutex, lp.use_count() = 3
after t1, p.use_count() = 1
  Derived::~Derived()
  Base::~Base()
after reset(), p.use_count() = 0
All threads completed, the last one deleted Derived
```
```
#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <mutex>

struct Base
{
    Base() { std::cout << "  Base::Base()\n"; }
    // Note: non-virtual destructor is OK here
    ~Base() { std::cout << "  Base::~Base()\n"; }
};

struct Derived: public Base
{
    Derived() { std::cout << "  Derived::Derived()\n"; }
    ~Derived() { std::cout << "  Derived::~Derived()\n"; }
};

void thr(std::shared_ptr<Base> p)
{
    std::cout  << "1st thr(): p.use_count() = " << p.use_count() << '\n';
    // std::this_thread::sleep_for(std::chrono::seconds(1));
    std::shared_ptr<Base> lp = p; // thread-safe, even though the
                                  // shared use_count is incremented
    {
        static std::mutex io_mutex;
        std::lock_guard<std::mutex> lk(io_mutex);
        std::cout  << "in mutex, lp.use_count() = " << lp.use_count() << '\n';
    }
}

int main()
{
    std::shared_ptr<Base> p = std::make_shared<Derived>();

    std::cout  << "before t1, p.use_count() = " << p.use_count() << '\n';
    std::thread t1(thr, p);//, t2(thr, p), t3(thr, p);
    std::cout  << "after t1, p.use_count() = " << p.use_count() << '\n';
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout  << "after sleep_for, p.use_count() = " << p.use_count() << '\n';
    p.reset(); // release ownership from main
    std::cout << "after reset(), p.use_count() = " << p.use_count() << '\n';
    t1.join(); //t2.join(); t3.join();
    std::cout << "All threads completed, the last one deleted Derived\n";
}
```
Possible output:
```
  Base::Base()
  Derived::Derived()
before t1, p.use_count() = 1
after t1, p.use_count() = 2
1st thr(): p.use_count() = 2
in mutex, lp.use_count() = 3
after sleep_for, p.use_count() = 1
  Derived::~Derived()
  Base::~Base()
after reset(), p.use_count() = 0
All threads completed, the last one deleted Derived
```
## On References and Pointers
- why do you have to use pointer? when creating object using new keyword, it returns a address allocated on heap;
- when to use new then? The new operator should only be used if the data object should remain in memory until delete is called. Otherwise if the new operator is not used, the object is automatically destroyed when it goes out of scope. In other words, the objects using new are cleaned up manually while other objects are automatically cleaned when they go out of scope.

You should use new when you wish an obj

Method 1 (using new)

Allocates memory for the object on the free store (This is frequently the same thing as the heap)
Requires you to explicitly delete your object later. (If you don't delete it, you could create a memory leak)
Memory stays allocated until you delete it. (i.e. you could return an object that you created using new)
The example in the question will leak memory unless the pointer is deleted; and it should always be deleted, regardless of which control path is taken, or if exceptions are thrown.
Method 2 (not using new)

Allocates memory for the object on the stack (where all local variables go) There is generally less memory available for the stack; if you allocate too many objects, you risk stack overflow.
You won't need to delete it later.
Memory is no longer allocated when it goes out of scope. (i.e. you shouldn't return a pointer to an object on the stack)
As far as which one to use; you choose the method that works best for you, given the above constraints.

Some easy cases:

If you don't want to worry about calling delete, (and the potential to cause memory leaks) you shouldn't use new.
If you'd like to return a pointer to your object from a function, you must use new

Anything allocated on the stack has to have a constant size, determined at compile-time (the compiler has to set the stack pointer correctly, or if the object is a member of another class, it has to adjust the size of that other class). That's why arrays in C# are reference types. They have to be, because with reference types, we can decide at runtime how much memory to ask for. And the same applies here. Only arrays with constant size (a size that can be determined at compile-time) can be allocated with automatic storage duration (on the stack). Dynamically sized arrays have to be allocated on the heap, by calling new.

```
void foo() {
  bar b;
  bar* b2 = new bar();
}
```
This function creates three values worth considering:

On line 1, it declares a variable b of type bar on the stack (automatic duration).

On line 2, it declares a bar pointer b2 on the stack (automatic duration), and calls new, allocating a bar object on the heap. (dynamic duration).
When the function returns, the following will happen: First, b2 goes out of scope (order of destruction is always opposite of order of construction). But b2 is just a pointer, so nothing happens, the memory it occupies is simply freed. And importantly, the memory it points to (the bar instance on the heap) is NOT touched. Only the pointer is freed, because only the pointer had automatic duration. Second, b goes out of scope, so since it has automatic duration, its destructor is called, and the memory is freed.

And the barinstance on the heap? It's probably still there. No one bothered to delete it, so we've leaked memory.
From this example, we can see that anything with automatic duration is guaranteed to have its destructor called when it goes out of scope. That's useful. But anything allocated on the heap lasts as long as we need it to, and can be dynamically sized, as in the case of arrays. That is also useful. We can use that to manage our memory allocations. What if the Foo class allocated some memory on the heap in its constructor, and deleted that memory in its destructor. Then we could get the best of both worlds, safe memory allocations that are guaranteed to be freed again, but without the limitations of forcing everything to be on the stack.
    - examples?
References are typically used to be able to modify a value passed to a function. They can also be used to pass a reference to an object instead of copying it to functions. But:
References can never be null which means they cannot be used for optional values
References do not have ownership semantics, so objects cannot be deleted using references
To have a class optionally (dynamically) contain an object of another class one must use a pointer. And pointers provide mechanisms for controlling the ownership of the dynamically created object.

https://www.geeksforgeeks.org/when-do-we-pass-arguments-by-reference-or-pointer/
