import numba
from SIMDprofiler import vector_print, vector_wrapper, init_diagnostics
import numpy as np

def loop1(A,B,C,D,E,n):
    a=np.copy(A)
    b=np.copy(B)
    c=np.copy(C)
    d=np.copy(D)
    e=np.copy(E)
    
    for i in range (n-1):
        a[i+1] = b[i]+c[i];
        b[i]   = c[i]*e[i];
        d[i]   = a[i]*e[i];
        
        
    return a,b,d
    
##### Solution with loop distribution
def loop1_sol(A,B,C,D,E,n):
    a=np.copy(A)
    b=np.copy(B)
    c=np.copy(C)
    d=np.copy(D)
    e=np.copy(E)
    
    for i in range (n-1):
        a[i+1] = b[i]+c[i];
        b[i]   = c[i]*e[i];
    for i in range (n-1):
        d[i]   = a[i]*e[i];
    return a,b,d    
        
def loop2(A,B,C,D,n):
    a=np.copy(A)
    b=np.copy(B)
    c=np.copy(C)
    d=np.copy(D)
    for i in range (n-1):
        a[i] = b[i-1]+c[i]
        b[i] = a[i+1]*d[i]
        
    return a,b
##### Solution with loop distribution
def loop2_sol(A,B,C,D,n):
    a=np.copy(A)
    b=np.copy(B)
    c=np.copy(C)
    d=np.copy(D)
    for i in range (n-1):
        b[i] = a[i+1]*d[i]
    for i in range (n-1):
        a[i] = b[i-1]+c[i]
    return a,b
    
def loop3(A,B,C,D,n):
    a=np.copy(A)
    b=np.copy(B)
    c=np.copy(C)
    d=np.copy(D)
    for i in range (n-1):
        a[i] = b[i] + c[i] * c[i] + b[i]*b[i] + c[i];
        d[i] = a[i] + a[i+1];
        
    return a,d
    
#### Solution with temporary array

def loop3_sol(A,B,C,D,n):
    a=np.copy(A)
    b=np.copy(B)
    c=np.copy(C)
    d=np.copy(D)
    for i in range (n-1):
        a[i] = b[i] + c[i] * c[i] + b[i]*b[i] + c[i];
        d[i] = a[i] + A[i+1];
        
    return a,d
    
def loop4(A,B,C,D,n):
    a=np.copy(A)
    b=np.copy(B)
    c=np.copy(C)
    d=np.copy(D)
    for i in range (n):
        a[i] += c[i] * d[i]
        b[i] = b[i - 1] + a[i] + d[i]
        
    return a,b
    
#### Partial vectorization with distribution
def loop4_sol(A,B,C,D,n):
    a=np.copy(A)
    b=np.copy(B)
    c=np.copy(C)
    d=np.copy(D)
    for i in range (n):
        a[i] += c[i] * d[i]
    for i in range (n):
        b[i] = b[i - 1] + a[i] + d[i]
        
    return a,b
def loop5(A,B,C,D,n):
    a=np.copy(A)
    b=np.copy(B)
    c=np.copy(C)
    d=np.copy(D)
    for i in range (n-1):
        a[i] += c[i] * d[i]
        if(i>=int(n/2)):
            b[i] = b[i - 1] + a[i] + d[i]
        else:
            b[i] = b[i + 1] + a[i] + d[i]
        
    return a,b
    
##### Partial vectorization using distribution plus loop peeling

def loop5_sol(A,B,C,D,n):
    a=np.copy(A)
    b=np.copy(B)
    c=np.copy(C)
    d=np.copy(D)
    for i in range (n-1):
        a[i] += c[i] * d[i]
    for i in range (int(n/2)):
        b[i] = b[i + 1] + a[i] + d[i]
    for i in range (int(n/2),n-1,1):
        b[i] = b[i - 1] + a[i] + d[i]

        
    return a,b
    
def loop6(AA, BB, nn):
    aa=np.copy(AA)
    bb=np.copy(BB)

    for i in range (nn):
        for j in range (1,nn):
            aa[j][i] = aa[j - 1][i] + bb[j][i];
        
    return aa
    
    
###### Vectorization with loop interchange

def loop6_sol(AA, BB, nn):
    aa=np.copy(AA)
    bb=np.copy(BB)

    for j in range (1,nn):
        for i in range (nn):
            aa[j][i] = aa[j - 1][i] + bb[j][i];
        
    return aa
    
def loop7(AA, BB, CC, nn):
    aa=np.copy(AA)
    bb=np.copy(BB)
    cc=np.copy(CC)

    for i in range(nn):
        for j in range(1,nn):
            aa[j][i] = aa[j-1][i] + cc[j][i]
        for j in range(1,nn):
            bb[j][i] = bb[j][i-1] + cc[j][i]

        
    return aa, bb
    
    
######## Loop distribution plus interchange for full vectorization

def loop7_sol(AA, BB, CC, nn):
    aa=np.copy(AA)
    bb=np.copy(BB)
    cc=np.copy(CC)
    for j in range(1,nn):
        for i in range(nn):
            aa[j][i] = aa[j-1][i] + cc[j][i]
    for i in range(nn):
        for j in range(1,nn):
            bb[j][i] = bb[j][i-1] + cc[j][i]

        
    return aa, bb
    

def loop8(A,B,n):
    a=np.copy(A)
    b=np.copy(B)
    vsum=0
    for i in range (n-1):
        vsum += a[i];
        b[i] = vsum;
        
    return vsum,b
    
    
    ##### Cannot be transformed, write the same loop

def loop8_sol(A,B,n):
    a=np.copy(A)
    b=np.copy(B)
    vsum=0
    for i in range (n-1):
        vsum += a[i];
        b[i] = vsum;
        
    return vsum,b

#### These functions are for checking for array differences. Beware, they might freeze your computer if too many indexes are different
def Compare1D(array1, array2, n, detail=False):
    s1=np.sum(array1)
    s2=np.sum(array2)
    if s1!=s2:
        if (detail):
            print("Your arrays have different values at indices:")
            for i in range(n):
                if array1[i]!=array2[i]:
                    print (i)
        else:
            print("Your arrays have different values, enable detail=True to see them")
        return False
    else:
        print("Your arrays seem to be equal, their checksum is:",s1)
        return True
def Compare2D(array1, array2, n, detail=False):
    s1=np.sum(array1)
    s2=np.sum(array2)
    if s1!=s2:
        if (detail):
            print("Your arrays have different values at indices:")
            for i in range(n):
                for j in range(n):
                    if array1[i][j]!=array2[i][j]:
                        print ("i:",i,"j:",j)
        else:
            print("Your arrays have different values, enable detail=True to see them")
        return False
    else:
        print("Your arrays seem to be equal, their checksum is:",s1)
        return True
    
    

def main():
    init_diagnostics()
    A=np.random.rand(5000)
    B = np.random.rand(5000)
    C = np.random.rand(5000)
    D=np.random.rand(5000)
    E=np.random.rand(5000)
    n = A.shape[0]
    AA=np.random.rand(500,500)
    BB=np.random.rand(500,500)
    CC=np.random.rand(500,500)
    nn=AA.shape[0]
    print("Testing loop 1")
    x,y,z=loop1(A,B,C,D,E,n)
    x2,y2,z2=loop1_sol(A,B,C,D,E,n)
    print("Comparing array A:")
    Compare1D(x,x2,n)
    print("Comparing array B:")
    Compare1D(y,y2,n)
    print("Comparing array D:")
    Compare1D(z,z2,n)
    print("---------------------------------------")
    
    
    print("Testing loop 2")
    x,y=loop2(A,B,C,D,n)
    x2,y2=loop2_sol(A,B,C,D,n)
    print("Comparing array A:")
    Compare1D(x,x2,n)
    print("Comparing array B:")
    Compare1D(y,y2,n)
    print("---------------------------------------")
    
    print("Testing loop 3")
    x,y=loop3(A,B,C,D,n)
    x2,y2=loop3_sol(A,B,C,D,n)
    print("Comparing array A:")
    Compare1D(x,x2,n)
    print("Comparing array D:")
    Compare1D(y,y2,n)
    print("---------------------------------------")
    
    
    print("Testing loop 4")
    x,y=loop4(A,B,C,D,n)
    x2,y2=loop4_sol(A,B,C,D,n)
    print("Comparing array A:")
    Compare1D(x,x2,n)
    print("Comparing array B:")
    Compare1D(y,y2,n)
    print("---------------------------------------")
    
    print("Testing loop 5")
    x,y=loop5(A,B,C,D,n)
    x2,y2=loop5_sol(A,B,C,D,n)
    print("Comparing array A:")
    Compare1D(x,x2,n)
    print("Comparing array B:")
    Compare1D(y,y2,n)
    print("---------------------------------------")
    
    
    print("Testing loop 6")
    x=loop6(AA, BB, nn)
    x2=loop6_sol(AA, BB, nn)
    print("Comparing array AA:")
    Compare2D(x,x2,nn)
    print("---------------------------------------")
    
    
    print("Testing loop 7")
    x,y=loop7(AA, BB, CC, nn)
    x2, y2=loop7_sol(AA, BB, CC, nn)
    print("Comparing array AA:")
    Compare2D(x,x2,nn)
    print("Comparing array BB:")
    Compare2D(y,y2,nn)
    print("---------------------------------------")
    
    print("Testing loop 8")
    x,y=loop8(A,B,n)
    x2,y2=loop8_sol(A,B,n)
    print("Comparing VSUM:")
    if x!=x2:
        print("VSUM is not equal")
    else:
        print("VSUM is equal")
    print("Comparing array B:")
    Compare1D(y,y2,n)
    print("---------------------------------------")
    print("Starting vectorization checks \n") 
    print("Checking the original Loop 1  \n")
    f=vector_wrapper(loop1)
    f2=vector_print(f)
    f2(A,B,C,D,E,n)
    print("\n Checking Loop1_sol  \n")
    f=vector_wrapper(loop1_sol)
    f2=vector_print(f)
    f2(A,B,C,D,E,n)
    print("---------------------------------------")
    
    
    print("Checking the original Loop 2  \n")
    f=vector_wrapper(loop2)
    f2=vector_print(f)
    f2(A,B,C,D,n)
    print("\n Checking Loop2_sol  \n")
    f=vector_wrapper(loop2_sol)
    f2=vector_print(f)
    f2(A,B,C,D,n)
    print("---------------------------------------")
    
    
    print("Checking the original Loop 3  \n")
    f=vector_wrapper(loop3)
    f2=vector_print(f)
    f2(A,B,C,D,n)
    print("\n Checking Loop3_sol  \n")
    f=vector_wrapper(loop3_sol)
    f2=vector_print(f)
    f2(A,B,C,D,n)
    print("---------------------------------------")
    
    
    print("Checking the original Loop 4  \n")
    f=vector_wrapper(loop4)
    f2=vector_print(f)
    f2(A,B,C,D,n)
    print("\n Checking Loop4_sol  \n")
    f=vector_wrapper(loop4_sol)
    f2=vector_print(f)
    f2(A,B,C,D,n)
    print("---------------------------------------")
    
    
    print("Checking the original Loop 5  \n")
    f=vector_wrapper(loop5)
    f2=vector_print(f)
    f2(A,B,C,D,n)
    print("\n Checking Loop5_sol  \n")
    f=vector_wrapper(loop5_sol)
    f2=vector_print(f)
    f2(A,B,C,D,n)
    print("---------------------------------------")
    
    
    print("Checking the original Loop 6  \n")
    f=vector_wrapper(loop6)
    f2=vector_print(f)
    f2(AA,BB,nn)
    print("\n Checking Loop6_sol  \n")
    f=vector_wrapper(loop6_sol)
    f2=vector_print(f)
    f2(AA,BB,nn)
    print("---------------------------------------")
    
    
    print("Checking the original Loop 7  \n")
    f=vector_wrapper(loop7)
    f2=vector_print(f)
    f2(AA,BB,CC,nn)
    print("\n Checking Loop7_sol  \n")
    f=vector_wrapper(loop7_sol)
    f2=vector_print(f)
    f2(AA,BB,CC, nn)
    print("---------------------------------------")
    
    
    print("Checking the original Loop 8  \n")
    f=vector_wrapper(loop8)
    f2=vector_print(f)
    f2(A, B,n)
    print("\n Checking Loop8_sol  \n")
    f=vector_wrapper(loop8_sol)
    f2=vector_print(f)
    f2(A, B,n)
    
main()