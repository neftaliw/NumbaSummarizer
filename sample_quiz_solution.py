from numba import jit
from NumbaSummarizer import vector_print, vector_wrapper, Simd_profile, Compare_loops
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
        
        
    return [a,b,d]
    
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
    return [a,d,d ]   
        
def loop2(A,B,C,D,n):
    a=np.copy(A)
    b=np.copy(B)
    c=np.copy(C)
    d=np.copy(D)
    for i in range (n-1):
        a[i] = b[i-1]+c[i]
        b[i] = a[i+1]*d[i]
        
    return [a,b]
##### Solution with loop distribution
def loop2_sol(A,B,C,D,n):
    a=np.copy(A)
    b=np.copy(B)
    c=np.copy(C)
    d=np.copy(D)
    for i in range (n-1):
        b[i] = a[i+1]*d[i]
    for j in range (n-1):
        a[j] = b[j-1]+c[j]
    return [a,b]
    
def loop3(A,B,C,D,n):
    a=np.copy(A)
    b=np.copy(B)
    c=np.copy(C)
    d=np.copy(D)
    for i in range (n-1):
        a[i] = b[i] + c[i] * c[i] + b[i]*b[i] + c[i];
        d[i] = a[i] + a[i+1];
        
    return [a,d]
    
#### Solution with temporary array

def loop3_sol(A,B,C,D,n):
    a=np.copy(A)
    b=np.copy(B)
    c=np.copy(C)
    d=np.copy(D)
    for i in range (n-1):
        a[i] = b[i] + c[i] * c[i] + b[i]*b[i] + c[i];
        d[i] = a[i] + A[i+1];
        
    return [a,d]
    
def loop4(A,B,C,D,n):
    a=np.copy(A)
    b=np.copy(B)
    c=np.copy(C)
    d=np.copy(D)
    for i in range (n):
        a[i] += c[i] * d[i]
        b[i] = b[i - 1] + a[i] + d[i]
        
    return [a,b]
    
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
        
    return [a,b]
    
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

        
    return [a,b]
    
def loop6(AA, BB, nn):
    aa=np.copy(AA)
    bb=np.copy(BB)

    for i in range (nn):
        for j in range (1,nn):
            aa[j][i] = aa[j - 1][i] + bb[j][i];
        
    return [aa,bb]
    
    
###### Vectorization with loop interchange

def loop6_sol(AA, BB, nn):
    aa=np.copy(AA)
    bb=np.copy(BB)

    for j in range (1,nn):
        for i in range (nn):
            aa[j][i] = aa[j - 1][i] + bb[j][i];
        
    return [aa,bb]
    
def loop7(AA, BB, CC, nn):
    aa=np.copy(AA)
    bb=np.copy(BB)
    cc=np.copy(CC)

    for i in range(nn):
        for j in range(1,nn):
            aa[j][i] = aa[j-1][i] + cc[j][i]
        for j in range(1,nn):
            bb[j][i] = bb[j][i-1] + cc[j][i]

        
    return [aa, bb]
    
    
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

        
    return [aa, bb]
    

def loop8(A,B,n):
    a=np.copy(A)
    b=np.copy(B)
    vsum=0
    for i in range (n-1):
        vsum += a[i];
        b[i] = vsum;
        
    return [vsum,b]
    
    
    ##### Cannot be transformed, write the same loop

def loop8_sol(A,B,n):
    a=np.copy(A)
    b=np.copy(B)
    vsum=0
    for i in range (n-1):
        vsum += a[i];
        b[i] = vsum;
        
    return [vsum,b]

  
    

def main():
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
    Compare_loops(loop1(A,B,C,D,E,n),loop1_sol(A,B,C,D,E,n))
    print("---------------------------------------")
    
    
    print("Testing loop 2")
    Compare_loops(loop2(A,B,C,D,n),loop2_sol(A,B,C,D,n))
    print("---------------------------------------")
    
    print("Testing loop 3")
    Compare_loops(loop3(A,B,C,D,n),loop3_sol(A,B,C,D,n))
    print("---------------------------------------")
    
    
    print("Testing loop 4")
    Compare_loops(loop4(A,B,C,D,n),loop4_sol(A,B,C,D,n))
    print("---------------------------------------")
    
    print("Testing loop 5")
    Compare_loops(loop5(A,B,C,D,n),loop5_sol(A,B,C,D,n))
    print("---------------------------------------")
    
    
    print("Testing loop 6")
    Compare_loops(loop6(AA, BB, nn),loop6_sol(AA, BB, nn))
    print("---------------------------------------")
    
    
    print("Testing loop 7")
    Compare_loops(loop7(AA, BB, CC, nn),loop7_sol(AA, BB, CC, nn))
    print("---------------------------------------")
    
    print("Testing loop 8")
    Compare_loops(loop8(A,B,n),loop8_sol(A,B,n))
    print("---------------------------------------")
    
    
    print("Starting vectorization checks \n") 
    print("Checking the original Loop 1  \n")
    Simd_profile(loop1)(A,B,C,D,E,n)
    print("\n Checking Loop1_sol  \n")
    Simd_profile(loop1_sol)(A,B,C,D,E,n)
    print("---------------------------------------")
    
    
    print("Checking the original Loop 2  \n")
    Simd_profile(loop2)(A,B,C,D,n)
    print("\n Checking Loop2_sol  \n")
    Simd_profile(loop2_sol)(A,B,C,D,n)
    print("---------------------------------------")
    
    
    print("Checking the original Loop 3  \n")
    Simd_profile(loop3)(A,B,C,D,n)
    print("\n Checking Loop3_sol  \n")
    Simd_profile(loop3_sol)(A,B,C,D,n)
    print("---------------------------------------")
    
    
    print("Checking the original Loop 4  \n")
    Simd_profile(loop4)(A,B,C,D,n)
    print("\n Checking Loop4_sol  \n")
    Simd_profile(loop4_sol)(A,B,C,D,n)
    print("---------------------------------------")
    
    
    print("Checking the original Loop 5  \n")
    Simd_profile(loop5)(A,B,C,D,n)
    print("\n Checking Loop5_sol  \n")
    Simd_profile(loop5_sol)(A,B,C,D,n)
    print("---------------------------------------")
    
    
    print("Checking the original Loop 6  \n")
    Simd_profile(loop6)(AA,BB,nn)
    print("\n Checking Loop6_sol  \n")
    Simd_profile(loop6_sol)(AA,BB,nn)
    print("---------------------------------------")
    
    
    print("Checking the original Loop 7  \n")
    Simd_profile(loop7)(AA,BB,CC,nn)
    print("\n Checking Loop7_sol  \n")
    Simd_profile(loop7_sol)(AA,BB,CC,nn)
    print("---------------------------------------")
    
    
    print("Checking the original Loop 8  \n")
    Simd_profile(loop8)(A, B,n)
    print("\n Checking Loop8_sol  \n")
    Simd_profile(loop8_sol)(A, B,n)
  

if __name__ == "__main__":    
    main()