import numpy as np
# define Python user-defined exceptions
class Error(Exception):
   """Base class for other exceptions"""
   pass
class OutputNotEqualError(Error):
   """Raised when the two outputs are not the same"""
   print("The output of both functions is not the same")
   pass
def Compare_loops(Orig_Output,Sol_Output):
    counter=0

    for item1,item2 in zip(Orig_Output,Sol_Output):
        counter+=1
        print("*******************************")
        print("Checking element "+str(counter)+ ":")
        #if (True):
        try:
            if (type(item1)!=type(item2)):
                print("For element "+str(counter)+ " both items are not the same type")
                raise OutputNotEqualError
                continue
            if (isinstance(item1,int) and isinstance(item1,int)) or (isinstance(item1,float) and isinstance(item1,float)):
                print("Element "+str(counter)+ " is a scalar variable")
                if(item1==item2):
                    print("Check passed")
                    continue
                else:
                    raise OutputNotEqualError
                    continue
            if (isinstance(item1,np.ndarray)):
                print("Element "+str(counter)+ " is a Numpy Array")
                if(np.array_equal(item1,item2)):
                    print("Check passed")
                    continue
                else:
                    raise OutputNotEqualError
                    continue
            print("For element "+str(counter)+ " this is not a supported type")
        except OutputNotEqualError:
            print("Check failed, both items are not equal")
        