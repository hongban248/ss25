import ctypes

mycuda_int=ctypes.CDLL("./mycuda_int.so")
mycuda_float=ctypes.CDLL("./mycuda_double.so")


class arr:
    def __init__(self,data,shape=-1):
        if shape==-1:
            shape=len(data)
        self.shape=shape
        if type(self.shape)==int:
            self.shape=(len(data),1)
        self.data=list(data)
        self.dtype=type(data[0])
        self.size=1
        for dim in self.shape:
            self.size=self.size*dim
        
        if len(data)!=self.size:
            raise ValueError("size not match number of elements!!")
        if self.dtype==int:
            self.lib=mycuda_int
            self.ctypes=ctypes.c_int
        elif self.dtype==float:
            self.lib=mycuda_float
            self.ctypes=ctypes.c_double
        else :
            raise TypeError("don't support such type of datas!!")
        
        self.data_ptr=(self.size*self.ctypes)(*data)
        ptr_type=ctypes.POINTER(self.ctypes)
        self.ptr_type=ptr_type

        #设置函数参数和返回值类型：
        self.lib.init_arr.argtypes=[ptr_type,ctypes.c_int,ctypes.c_int]
        self.lib.init_arr.restype=None

        self.lib.init_arr_zero.argtypes=[ptr_type,ctypes.c_int,ctypes.c_int]
        self.lib.init_arr_zero.restype=None

        self.lib.init_arr_one.argtypes=[ptr_type,ctypes.c_int,ctypes.c_int]
        self.lib.init_arr_one.restype=None

        self.lib.sum_arr.argtypes=[ptr_type,ctypes.c_int,ctypes.c_int]
        self.lib.sum_arr.restype=self.ctypes

        self.lib.aver_arr.argtypes=[ptr_type,ctypes.c_int,ctypes.c_int]
        self.lib.aver_arr.restype=ctypes.c_double

        self.lib.det_mat.argtypes=[ptr_type,ctypes.c_int]
        self.lib.det_mat.restype=self.ctypes

        self.lib.multiple.argtypes=[ptr_type,ptr_type,ctypes.c_int,ctypes.c_int,ctypes.c_int]
        self.lib.multiple.restype=ptr_type
    
    @property
    def show_arr(self):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                print(self.data[i*self.shape[1]+j],end=' ')
            print()
        return 1

    def __sync__(self):
        self.data=list(self.data_ptr)  #如果需要直接更改，则调用

    @property
    def init_arr(self):
        self.lib.init_arr(self.data_ptr,self.shape[0],self.shape[1])
        self.__sync__()
        return self
    @property
    def init_arr_zero(self):
        self.lib.init_arr_zero(self.data_ptr,self.shape[0],self.shape[1])
        self.__sync__() 
        return self
    @property
    def init_arr_one(self):
        self.lib.init_arr_one(self.data_ptr,self.shape[0],self.shape[1])
        self.__sync__()
        return self
    @property
    def sum_arr(self):
        return self.lib.sum_arr(self.data_ptr,self.shape[0],self.shape[1])
    @property
    def aver_arr(self):
        return self.lib.aver_arr(self.data_ptr,self.shape[0],self.shape[1])
   
    def reshape(self,new_shape):
        if type(new_shape)==int:
            new_shape=(new_shape,1)
        new_size=1
        for dim in new_shape:
            new_size=new_size*dim
        if new_size!=self.size:
            raise ValueError("can't reshape!!")
        self.shape=new_shape
        return self
    @property
    def IntToFloat(self):
        if self.dtype==int:
            new_data=[float(i) for i in self.data]  
            self.__init__(new_data,self.shape)
        return self
    @property
    def float(self):
        new_data=[float(i) for i in self.data]   
        return arr(new_data,self.shape) 
    @property
    def det_mat(self):
        if self.shape[0]!=self.shape[1]:
            raise ValueError("can not determine det!!") 
        return self.lib.det_mat(self.data_ptr,self.shape[0])

    def zeros(shape,value=0):
        if type(shape)==int:
            shape=(shape,1)
        size=1
        for dim in shape:
            size=size*dim
        return arr([value for i in range(size)],shape)
    def ones(shape):
        return arr.zeros(shape,1)

    def add(self,obj2):
        if type(obj2)!=arr:
            return arr([i+obj2 for i in self.data],self.shape)
        if self.shape==obj2.shape:
            return arr([self.data[i]+obj2.data[i] for i in range(self.size)],self.shape)
        if self.shape[0]==obj2.shape[0]:
            if obj2.shape[1]==1:
                return arr([self.data[i]+obj2.data[i//self.shape[1]] for i in range(self.size)],self.shape)
            if self.shape[1]==1:
                return obj2.add(self)
        if self.shape[1]==obj2.shape[1]:
            if obj2.shape[0]==1:
                return arr([self.data[i]+obj2.data[i%self.shape[1]] for i in range(self.size)],self.shape)
            if self.shape[0]==1:
                return obj2.add(self)
        if obj2.size==1:
            return arr([self.data[i]+obj2.data[0] for i in range(self.size)],self.shape)
        if self.size==1:
            return obj2.add(self)

        raise ValueError("those shapes can not add!")
    def __add__(self, obj2):
        return self.add(obj2)
    def __radd__(self, obj2):
        return self.add(obj2)

    def multiple(self,obj2):
        if type(obj2)!=arr:
            return arr([self.data[i]*obj2 for i in range(self.size)],self.shape)
        if self.shape[1]!=obj2.shape[0]:
            raise ValueError("can not matrix multiple!!")
        
        if self.dtype==int and obj2.dtype==int:
            outcome=self.lib.multiple(self.data_ptr,obj2.data_ptr,self.shape[0],self.shape[1],obj2.shape[1])
            result_data = [outcome[i] for i in range(self.shape[0] * obj2.shape[1])]
            out_arr = arr(result_data, (self.shape[0], obj2.shape[1]))
            return out_arr
        if self.dtype==float and obj2.dtype==float:
            outcome=self.lib.multiple(self.data_ptr,obj2.data_ptr,self.shape[0],self.shape[1],obj2.shape[1])
            result_data = [outcome[i] for i in range(self.shape[0] * obj2.shape[1])]
            out_arr = arr(result_data, (self.shape[0], obj2.shape[1]))
            return out_arr
        return multiple(self.float,obj2.float)
    def __mul__(self, other):
        return self.multiple(other)
    def __rmul__(self,other):
        return self.multiple(other)

    def __neg__(self):
        return arr([-self.data[i] for i in range(self.size)],self.shape)
    def __sub__(self,obj2):
        try:
            return add(self,-obj2)
        except:
            raise ValueError("can not submit!!")

    @property
    def T(self):
        self.lib.inverse.argtypes=[self.ptr_type,ctypes.c_int,ctypes.c_int]
        self.lib.inverse.restype=self.ptr_type
        self.lib.inverse(self.data_ptr,self.shape[0],self.shape[1])
        self.__sync__()
        self.shape=(self.shape[1],self.shape[0])
        return self

    def len(self):
        return self.size
    
    @property
    def copy(self):
        return arr(self.data,self.shape)



def add(obj1,obj2):
    if type(obj1)!=arr and type(obj2)!=arr:
        return arr([obj1+obj2],1)
    if  type(obj1)==arr  :
        return obj1.add(obj2)
    else:
        return obj2.add(obj1)

def multiple(obj1,obj2):
    if type(obj1)!=arr and type(obj2)!=arr:
        return arr([obj1*obj2],1)
    if type(obj1)==arr:
        return obj1.multiple(obj2)
    else :
        return obj2.multiple(obj1)

if __name__=='__main__':
    
    arr1=arr.zeros(5).init_arr.T
    arr2=arr1.copy.T
    
    out=arr2*arr1
    out.show_arr
    

    
